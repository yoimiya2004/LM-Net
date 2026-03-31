import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from basicsr.archs.Ublock import unetBlock

def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(in_channels, out_channels, kernel_size, padding=(kernel_size // 2), bias=bias)


class ECAAttention(nn.Module):

    def __init__(self, kernel_size=3):
        super().__init__()
        self.gap=nn.AdaptiveAvgPool2d(1)
        self.conv=nn.Conv1d(1,1,kernel_size=kernel_size,padding=(kernel_size-1)//2)
        self.sigmoid=nn.Sigmoid()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        y=self.gap(x) #bs,c,1,1
        y=y.squeeze(-1).permute(0,2,1) #bs,1,c
        y=self.conv(y) #bs,1,c
        y=self.sigmoid(y) #bs,1,c
        y=y.permute(0,2,1).unsqueeze(-1) #bs,c,1,1
        return x*y.expand_as(x)


class fft_bench(nn.Module):
    def __init__(self, n_feat):
        super(fft_bench, self).__init__()
        self.main = nn.Conv2d(n_feat, n_feat, kernel_size=3, padding=1)
        self.mag = nn.Conv2d(n_feat, n_feat, kernel_size=1, padding=0)
        self.pha = nn.Sequential(
            nn.Conv2d(n_feat, n_feat, kernel_size=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(n_feat, n_feat, kernel_size=1, padding=0),
        )

        self.eca = ECAAttention()

    def forward(self, x):
        _, _, H, W = x.shape
        fre = torch.fft.rfft2(x, norm='backward')
        mag = torch.abs(fre)
        pha = torch.angle(fre)

        pha_eca = self.eca(pha)

        mag_out = self.mag(mag)
        pooling = torch.nn.functional.adaptive_avg_pool2d(mag_out, (1, 1))
        pooling = torch.nn.functional.softmax(pooling, dim=1)
        pha1 = pha_eca * pooling
        pha1 = self.pha(pha1)
        pha_out = pha1 + pha
        real = mag_out * torch.cos(pha_out)
        imag = mag_out * torch.sin(pha_out)
        fre_out = torch.complex(real, imag)
        y = torch.fft.irfft2(fre_out, s=(H, W), norm='backward')

        return self.main(x) + y


class FullCapacityDGDFFN(nn.Module):
    """
    全容量深度门控前馈网络 (Full-Capacity DGD-FFN)
    回应缺陷一：恢复 5x5 与双 3x3 分支的复杂门控交互，最大化局部纹理的建模容量。
    """

    def __init__(self, channels, expansion_factor=2.0):
        super(FullCapacityDGDFFN, self).__init__()
        hidden_channels = int(channels * expansion_factor)

        # 第一次输入投影，扩张特征空间
        self.project_in = nn.Conv2d(channels, hidden_channels * 2, kernel_size=1, bias=False)

        # 复杂多分支深度卷积
        # 分支 1：串联两个 3x3，获取等效 5x5 的精细局部非线性
        self.dwconv_3x3_1 = nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1,
                                      groups=hidden_channels, bias=False)
        self.dwconv_3x3_2 = nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1,
                                      groups=hidden_channels, bias=False)

        # 分支 2：单层 5x5，直接获取较大的局部感知域
        self.dwconv_5x5 = nn.Conv2d(hidden_channels, hidden_channels, kernel_size=5, padding=2, groups=hidden_channels,
                                    bias=False)

        self.act = nn.GELU()

        # 第二次投影，压缩回原通道
        self.project_out = nn.Conv2d(hidden_channels, channels, kernel_size=1, bias=False)

    def forward(self, x):
        x_in = self.project_in(x)
        x_branch1, x_branch2 = x_in.chunk(2, dim=1)

        # 细粒度非线性分支 (两个 3x3)
        feat1 = self.dwconv_3x3_1(x_branch1)
        feat1 = self.dwconv_3x3_2(self.act(feat1))

        # 略大感知分支 (5x5)
        feat2 = self.dwconv_5x5(x_branch2)

        # 高维分支融合与门控乘法
        gated_feat = self.act(feat1) * feat2

        return self.project_out(gated_feat)


class AtrousContextCompletion(nn.Module):
    """
    空洞上下文补全模块 (ACCM)
    回应缺陷二：利用多膨胀率深度卷积模拟 U-Block 的多尺度上下文聚合，
    在 100% 保持空间分辨率的前提下，具备强大的“缺失结构补全/幻觉”能力。
    """

    def __init__(self, channels):
        super(AtrousContextCompletion, self).__init__()

        # 模拟 U-Net 的多尺度层级 (RF 指数级放大)
        # d=1 (模拟原分辨率), d=2 (模拟下采样1次), d=4 (模拟下采样2次), d=8 (模拟下采样3次)
        self.branch_d1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, dilation=1, groups=channels,
                                   bias=False)
        self.branch_d2 = nn.Conv2d(channels, channels, kernel_size=3, padding=2, dilation=2, groups=channels,
                                   bias=False)
        self.branch_d4 = nn.Conv2d(channels, channels, kernel_size=3, padding=4, dilation=4, groups=channels,
                                   bias=False)
        self.branch_d8 = nn.Conv2d(channels, channels, kernel_size=3, padding=8, dilation=8, groups=channels,
                                   bias=False)

        self.act = nn.GELU()

        # 跨尺度信息聚合 (1x1 卷积将所有尺度的结构特征压缩融合)
        self.aggregator = nn.Conv2d(channels * 4, channels, kernel_size=1, bias=False)

    def forward(self, x):
        feat_d1 = self.branch_d1(x)
        feat_d2 = self.branch_d2(x)
        feat_d4 = self.branch_d4(x)
        feat_d8 = self.branch_d8(x)

        # 在通道维度拼接所有尺度的结构特征
        concat_feat = torch.cat([feat_d1, feat_d2, feat_d4, feat_d8], dim=1)

        # 聚合输出
        out = self.aggregator(self.act(concat_feat))
        return out


class HCBlock(nn.Module):
    """
    最终组装：高容量无损细节保留块 (HC-LDB)
    """

    def __init__(self, channels):
        super(HCBlock, self).__init__()

        # 先利用高容量模块细化局部纹理
        self.local_refine = FullCapacityDGDFFN(channels)

        # 再利用多空洞模块补全全局上下文
        self.context_complete = AtrousContextCompletion(channels)

        self.norm1 = nn.BatchNorm2d(channels)
        self.norm2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        # 步骤 1: 局部纹理细化 + 残差
        res1 = self.local_refine(self.norm1(x)) + x

        # 步骤 2: 多尺度上下文聚合 + 残差
        out = self.context_complete(self.norm2(res1)) + res1

        return out


class Block(nn.Module):
    def __init__(self, conv, dim, kernel_size):
        super(Block, self).__init__()

        self.conv1 = conv(dim, dim, kernel_size, bias=True)
        self.act1 = nn.ReLU(inplace=True)
        self.conv2 = conv(dim, dim, kernel_size, bias=True)
        self.ublock = unetBlock(dim=dim)

    def forward(self, x):

        res = self.act1(self.conv1(x))
        res = res + x
        res = self.conv2(res)
        res = self.ublock(res)
        res += x

        return res


class Group(nn.Module):
    def __init__(self, conv, dim, kernel_size, blocks):
        super(Group, self).__init__()
        # modules = [Block(conv, dim, kernel_size) for _ in range(blocks)]
        modules = [HCBlock(dim) for _ in range(blocks)]
        modules.append(conv(dim, dim, kernel_size))
        self.gp = nn.Sequential(*modules)

    def forward(self, x):
        res = self.gp(x)
        res += x
        return res


class DENet(nn.Module):
    def __init__(self, gps, blocks, conv1=default_conv):
        super(DENet, self).__init__()
        self.gps = gps
        self.dim = 16
        kernel_size = 3

        self.fft1 = fft_bench(self.dim)
        self.fft2 = fft_bench(self.dim)
        self.fft3 = fft_bench(self.dim)

        pre_process = [conv1(3, self.dim, kernel_size)]
        # assert self.gps == 3

        self.g1 = Group(conv1, self.dim, kernel_size, blocks=blocks)
        self.g2 = Group(conv1, self.dim, kernel_size, blocks=blocks)
        self.g3 = Group(conv1, self.dim, kernel_size, blocks=blocks)

        self.ca = nn.Sequential(*[
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(self.dim * self.gps, self.dim // 4, 1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.dim // 4, self.dim * self.gps, 1, padding=0, bias=True),
            nn.Sigmoid()
        ])

        post_precess = [
            conv1(self.dim, 3, kernel_size)]

        self.pre = nn.Sequential(*pre_process)
        self.post = nn.Sequential(*post_precess)

    def forward(self, x1):

        x = self.pre(x1)

        res1 = self.g1(x)
        res1 = self.fft1(res1)

        res2 = self.g2(res1)
        res2 = self.fft2(res2)

        res3 = self.g3(res2)
        res3 = self.fft3(res3)

        w = self.ca(torch.cat([res1, res2, res3], dim=1))
        w = w.view(-1, self.gps, self.dim)[:, :, :, None, None]
        out = w[:, 0, ::] * res1 + w[:, 1, ::] * res2 + w[:, 2, ::] * res3
        x = self.post(out)
        return x + x1


if __name__ == "__main__":
    x = torch.randn(1, 3, 64, 64)
    net = DENet(3,3)
    out = net(x)
    print(out.size())