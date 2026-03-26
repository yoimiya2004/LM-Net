import torch
import torch.nn as nn
import torch.nn.functional as F

from pytorch_wavelets import DWTForward

from basicsr.archs.uvmb import UVMB


class UNetConvBlock(nn.Module):
    """UNet 下采样块：Conv(stride=2) + ReLU"""
    def __init__(self, in_chans, out_chans):
        super(UNetConvBlock, self).__init__()
        block = []
        block.append(nn.Conv2d(in_chans, out_chans, kernel_size=4, stride=2, padding=1))
        block.append(nn.ReLU())
        self.block = nn.Sequential(*block)

    def forward(self, x):
        return self.block(x)


class UNetUpBlock(nn.Module):
    """UNet 上采样块：上采样 + 融合"""
    def __init__(self, in_chans, out_chans, up_mode):
        super(UNetUpBlock, self).__init__()
        if up_mode == 'upconv':
            self.up = nn.ConvTranspose2d(in_chans, out_chans, kernel_size=2, stride=2)
        elif up_mode == 'upsample':
            self.up = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                nn.Conv2d(in_chans, out_chans, kernel_size=1),
            )

        self.conv_block = nn.Sequential(
            nn.Conv2d(out_chans * 2, out_chans, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, bridge):
        up = self.up(x)
        diff_h = bridge.size(2) - up.size(2)
        diff_w = bridge.size(3) - up.size(3)
        up = F.pad(up, [diff_w // 2, diff_w - diff_w // 2,
                        diff_h // 2, diff_h - diff_h // 2])
        out = torch.cat([up, bridge], dim=1)
        return self.conv_block(out)


def conv(in_channels, out_channels, kernel_size, bias=False, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size, padding=(kernel_size // 2), bias=bias, stride=stride)


class unetBlock(nn.Module):
    """原版 unetBlock，保留以便回退"""
    def __init__(self, up_mode='upconv', dim=16):
        super(unetBlock, self).__init__()
        assert up_mode in ('upconv', 'upsample')
        self.layer1 = UNetConvBlock(in_chans=dim, out_chans=32)
        self.layer2 = UNetConvBlock(in_chans=32, out_chans=64)
        self.layer3 = UNetConvBlock(in_chans=64, out_chans=128)
        self.layer_0 = UNetUpBlock(in_chans=32, out_chans=dim, up_mode=up_mode)
        self.layer_1 = UNetUpBlock(in_chans=64, out_chans=32, up_mode=up_mode)
        self.layer_2 = UNetUpBlock(in_chans=128, out_chans=64, up_mode=up_mode)

    def forward(self, x):
        blocks = []
        blocks.append(x)
        x1 = self.layer1(x)
        blocks.append(x1)
        x2 = self.layer2(x1)
        blocks.append(x2)
        x3 = self.layer3(x2)
        x_2 = self.layer_2(x3, blocks[-1])
        x_1 = self.layer_1(x_2, blocks[-2])
        x_0 = self.layer_0(x_1, blocks[-3])
        x_final = x_0 + x
        return x_final


class unetBlock_uvmb(nn.Module):
    """在每个尺度局部插入 UVMB 的 unetBlock"""
    def __init__(self, up_mode='upconv', dim=16, uvmb_mid_size=64, uvmb_on_deep=True):
        super(unetBlock_uvmb, self).__init__()
        assert up_mode in ('upconv', 'upsample')

        self.layer1 = UNetConvBlock(in_chans=dim, out_chans=32)
        self.layer2 = UNetConvBlock(in_chans=32, out_chans=64)
        self.layer3 = UNetConvBlock(in_chans=64, out_chans=128)
        self.layer_0 = UNetUpBlock(in_chans=32, out_chans=dim, up_mode=up_mode)
        self.layer_1 = UNetUpBlock(in_chans=64, out_chans=32, up_mode=up_mode)
        self.layer_2 = UNetUpBlock(in_chans=128, out_chans=64, up_mode=up_mode)

        # 为每个尺度创建 UVMB（通道数需匹配）
        self.uvmb0 = UVMB(in_channels=dim, mid_size=uvmb_mid_size)
        self.uvmb1 = UVMB(in_channels=32, mid_size=uvmb_mid_size)
        self.uvmb2 = UVMB(in_channels=64, mid_size=uvmb_mid_size)
        if uvmb_on_deep:
            self.uvmb3 = UVMB(in_channels=128, mid_size=uvmb_mid_size)
        else:
            self.uvmb3 = None

    def forward(self, x):
        blocks = []
        blocks.append(x)

        try:
            x = x + self.uvmb0(x)
        except Exception:
            x = x

        x1 = self.layer1(x)
        try:
            x1 = x1 + self.uvmb1(x1)
        except Exception:
            x1 = x1
        blocks.append(x1)

        x2 = self.layer2(x1)
        try:
            x2 = x2 + self.uvmb2(x2)
        except Exception:
            x2 = x2
        blocks.append(x2)

        x3 = self.layer3(x2)
        if self.uvmb3 is not None:
            try:
                x3 = x3 + self.uvmb3(x3)
            except Exception:
                x3 = x3

        x_2 = self.layer_2(x3, blocks[-1])
        x_1 = self.layer_1(x_2, blocks[-2])
        x_0 = self.layer_0(x_1, blocks[-3])

        x_final = x_0 + x
        return x_final


if __name__ == '__main__':
    # 简单测试
    x = torch.randn(1, 32, 64, 64)
    net = unetBlock_uvmb(dim=32)
    out = net(x)
    print('out.size():', out.size())
    total = sum(p.numel() for p in net.parameters())
    print('params:', total)

'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_wavelets import DWTForward

class UNetConvBlock(nn.Module):
    def __init__(self, in_chans, out_chans):
        super(UNetConvBlock, self).__init__()
        block = []

        block.append(nn.Conv2d(in_chans, out_chans, kernel_size=4, stride=2, padding=1))
        block.append(nn.ReLU())

        self.block = nn.Sequential(*block)

    def forward(self, x):
        out = self.block(x)
        return out


class UNetUpBlock(nn.Module):
    def __init__(self, in_chans, out_chans, up_mode):
        super(UNetUpBlock, self).__init__()
        if up_mode == 'upconv':
            self.up = nn.ConvTranspose2d(in_chans, out_chans, kernel_size=2, stride=2)
        elif up_mode == 'upsample':
            self.up = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                nn.Conv2d(in_chans, out_chans, kernel_size=1),
            )

        self.conv_block = nn.Sequential(
            nn.Conv2d(out_chans * 2, out_chans, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, bridge):
        # 上采样输入特征图 x
        up = self.up(x)

        diff_h = bridge.size(2) - up.size(2)
        diff_w = bridge.size(3) - up.size(3)

        up = F.pad(up, [diff_w // 2, diff_w - diff_w // 2,
                        diff_h // 2, diff_h - diff_h // 2])

        out = torch.cat([up, bridge], dim=1)

        out = self.conv_block(out)
        return out

def conv(in_channels, out_channels, kernel_size, bias=False, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size, padding=(kernel_size // 2), bias=bias, stride=stride)


class unetBlock(nn.Module):
    def __init__(self, up_mode='upconv', dim=16):  # wf=48
        super(unetBlock, self).__init__()

        assert up_mode in ('upconv', 'upsample')

        self.layer1 = UNetConvBlock(in_chans=dim, out_chans=32)
        self.layer2 = UNetConvBlock(in_chans=32, out_chans=64)
        self.layer3 = UNetConvBlock(in_chans=64, out_chans=128)
        self.layer_0 = UNetUpBlock(in_chans=32, out_chans=dim, up_mode=up_mode)
        self.layer_1 = UNetUpBlock(in_chans=64, out_chans=32, up_mode=up_mode)
        self.layer_2 = UNetUpBlock(in_chans=128, out_chans=64, up_mode=up_mode)

    def forward(self, x):

        blocks = []
        blocks.append(x)

        x1 = self.layer1(x)

        blocks.append(x1)

        x2 = self.layer2(x1)

        blocks.append(x2)

        x3 = self.layer3(x2)

        x_2 = self.layer_2(x3, blocks[-1])

        x_1 = self.layer_1(x_2, blocks[-2])

        x_0 = self.layer_0(x_1, blocks[-3])

        x_final = x_0 + x

        return x_final


if __name__ == '__main__':
    x = torch.randn(1, 32, 64, 64)
    net =unetBlock(dim=32)
    out = net(x)
    print(out.size())
'''