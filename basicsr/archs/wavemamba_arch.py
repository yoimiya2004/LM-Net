import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from collections import OrderedDict
import time
from scipy.io import savemat
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, selective_scan_ref
from einops import rearrange, repeat
from functools import partial
from timm.layers import DropPath, to_2tuple, trunc_normal_
from pdb import set_trace as stx
from typing import Optional, Callable
import math
import numbers
from timm.layers import DropPath, to_2tuple, trunc_normal_
import sys
from basicsr.utils.registry import ARCH_REGISTRY
import torch.autograd
from basicsr.archs.detail_enhance_net import DENet
from basicsr.archs.wavelet import DWT, IWT

class Attention(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, groups=1, reduction=0.0625, kernel_num=4, min_channel=16):
        super(Attention, self).__init__()
        attention_channel = max(int(in_planes * reduction), min_channel)
        self.kernel_size = kernel_size
        self.kernel_num = kernel_num
        self.temperature = 1.0

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Conv2d(in_planes, attention_channel, 1, bias=False)
        self.relu = nn.GELU()

        self.channel_fc = nn.Conv2d(attention_channel, in_planes, 1, bias=True)
        self.func_channel = self.get_channel_attention

        if in_planes == groups and in_planes == out_planes:  # depth-wise convolution
            self.func_filter = self.skip
        else:
            self.filter_fc = nn.Conv2d(attention_channel, out_planes, 1, bias=True)
            self.func_filter = self.get_filter_attention

        if kernel_size == 1:  # point-wise convolution
            self.func_spatial = self.skip
        else:
            self.spatial_fc = nn.Conv2d(attention_channel, kernel_size * kernel_size, 1, bias=True)
            self.func_spatial = self.get_spatial_attention

        if kernel_num == 1:
            self.func_kernel = self.skip
        else:
            self.kernel_fc = nn.Conv2d(attention_channel, kernel_num, 1, bias=True)
            self.func_kernel = self.get_kernel_attention

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def update_temperature(self, temperature):
        self.temperature = temperature

    @staticmethod
    def skip(_):
        return 1.0

    def get_channel_attention(self, x):
        channel_attention = torch.sigmoid(self.channel_fc(x).view(x.size(0), -1, 1, 1) / self.temperature)
        return channel_attention

    def get_filter_attention(self, x):
        filter_attention = torch.sigmoid(self.filter_fc(x).view(x.size(0), -1, 1, 1) / self.temperature)
        return filter_attention

    def get_spatial_attention(self, x):
        spatial_attention = self.spatial_fc(x).view(x.size(0), 1, 1, 1, self.kernel_size, self.kernel_size)
        spatial_attention = torch.sigmoid(spatial_attention / self.temperature)
        return spatial_attention

    def get_kernel_attention(self, x):
        kernel_attention = self.kernel_fc(x).view(x.size(0), -1, 1, 1, 1, 1)
        kernel_attention = F.softmax(kernel_attention / self.temperature, dim=1)
        return kernel_attention

    def forward(self, x):
        x = self.avgpool(x)
        x = self.fc(x)
        x = self.relu(x)
        return self.func_channel(x), self.func_filter(x), self.func_spatial(x), self.func_kernel(x)



# Layer Norm
class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_first"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class ChannelAttention(nn.Module):
    """Channel attention used in RCAN.
    Args:
        num_feat (int): Channel number of intermediate features.
        squeeze_factor (int): Channel squeeze factor. Default: 16.
    """

    def __init__(self, num_feat, squeeze_factor=16):
        super(ChannelAttention, self).__init__()
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(num_feat, num_feat // squeeze_factor, 1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_feat // squeeze_factor, num_feat, 1, padding=0),
            nn.Sigmoid())

    def forward(self, x):
        y = self.attention(x)
        return x * y


class CAB(nn.Module):

    def __init__(self, num_feat, compress_ratio=3, squeeze_factor=30):
        super(CAB, self).__init__()

        self.cab = nn.Sequential(
            nn.Conv2d(num_feat, num_feat // compress_ratio, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(num_feat // compress_ratio, num_feat, 3, 1, 1),
            ChannelAttention(num_feat, squeeze_factor)
        )

    def forward(self, x):
        return self.cab(x)


class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


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


class HFCAttention(nn.Module):
    """
    专门针对雾霾频带的频率通道注意力模块 (HFCAttention)
    创新点：利用 2D 快速傅里叶变换提取各通道的低频幅度谱（表征全局雾霾能量），
    以此引导模型自适应地为不同通道分配权重，解决雾霾在不同颜色通道衰减不一致的问题。
    """

    def __init__(self, channels, reduction=16):
        super(HFCAttention, self).__init__()
        mid_channels = max(1, channels // reduction)

        # 使用 MLP 学习不同通道在低频雾霾频带上的相互关系
        self.mlp = nn.Sequential(
            nn.Linear(channels, mid_channels, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid_channels, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, h, w = x.shape

        # 1. 2D 傅里叶变换转换到频域
        # rfft2 输出形状为 [B, C, H, W//2 + 1]
        fft_x = torch.fft.rfft2(x, norm='backward')

        # 2. 提取幅度谱 (Amplitude Spectrum)
        amp = torch.abs(fft_x)

        # 3. 低频截取 (Low-Frequency Cropping)
        # 根据 WDMamba 先验，雾霾主要在低频。rfft2 的低频集中在左上角。
        # 这里我们截取幅度谱的前 25% (即 1/4 区域) 作为核心雾霾频带
        h_low = max(1, amp.shape[2] // 4)
        w_low = max(1, amp.shape[3] // 4)
        low_freq_amp = amp[:, :, :h_low, :w_low]

        # 4. 频域全局池化：计算每个通道的“低频雾霾总能量”
        freq_energy = low_freq_amp.mean(dim=(2, 3))  # Shape: [B, C]

        # 5. 通过 MLP 生成通道注意力权重
        channel_weights = self.mlp(freq_energy)  # Shape: [B, C]
        channel_weights = channel_weights.view(b, c, 1, 1)

        # 6. 将频率感知的权重作用回原始空间域特征
        return x * channel_weights


class ffn(nn.Module):
    def __init__(self, num_feat, ffn_expand=2, activation=nn.GELU, use_attention=True):
        super(ffn, self).__init__()
        self.dw_channel = num_feat * ffn_expand
        self.conv1 = nn.Conv2d(num_feat, self.dw_channel, kernel_size=1, padding=0, stride=1)
        self.conv2 = nn.Conv2d(self.dw_channel, self.dw_channel, kernel_size=3, padding=1, stride=1,
                               groups=self.dw_channel)
        self.conv3 = nn.Conv2d(self.dw_channel // 2, num_feat, kernel_size=1, padding=0, stride=1)
        self.activation = activation()
        self.use_attention = use_attention
        if use_attention:
            # 注意：特征在经过 x.chunk(2) 之后才会送入 attention，因此通道数是 dw_channel // 2
            attention_channels = self.dw_channel // 2
            self.hfc_attention = HFCAttention(channels=attention_channels)
        # --------------------------------------------------------

    def forward(self, x):
        x = self.conv2(self.conv1(x))
        x1, x2 = x.chunk(2, dim=1)
        x = self.activation(x1) * x2
        if self.use_attention:
            x = self.hfc_attention(x)
        x = self.conv3(x)
        return x


class SS2D(nn.Module):
    def __init__(
            self,
            d_model,
            d_state=16,
            d_conv=3,
            expand=2,
            dt_rank="auto",
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            dropout=0.,
            conv_bias=True,
            bias=False,
            device=None,
            dtype=None,
            **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        self.conv2d = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )
        self.act = nn.SiLU()

        self.x_proj = (
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
        )
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))  # (K=4, N, inner)
        del self.x_proj

        self.dt_projs = (
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
        )
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0))  # (K=4, inner, rank)
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))  # (K=4, inner)
        del self.dt_projs

        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=4, merge=True)  # (K=4, D, N)
        self.Ds = self.D_init(self.d_inner, copies=4, merge=True)  # (K=4, D, N)

        self.selective_scan = selective_scan_fn

        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4,
                **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        dt_proj.bias._no_reinit = True

        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        # S4D real initialization
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D

    def forward_core(self, x: torch.Tensor):
        B, C, H, W = x.shape
        L = H * W
        K = 4

        x_hwwh = torch.stack([x.view(B, -1, L), torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L)],
                             dim=1).view(B, 2, -1, L)
        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1)  # (1, 4, 192, 3136)
        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)

        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)

        xs = xs.float().view(B, -1, L)
        dts = dts.contiguous().float().view(B, -1, L)  # (b, k * d, l)
        Bs = Bs.float().view(B, K, -1, L)
        Cs = Cs.float().view(B, K, -1, L)  # (b, k, d_state, l)
        Ds = self.Ds.float().view(-1)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1)  # (k * d)

        out_y = self.selective_scan(
            xs, dts,
            As, Bs, Cs, Ds, z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(B, K, -1, L)
        assert out_y.dtype == torch.float

        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
        wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)

        return out_y[:, 0], inv_y[:, 0], wh_y, invwh_y

    def forward(self, x: torch.Tensor, **kwargs):
        B, H, W, C = x.shape

        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)

        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.act(self.conv2d(x))
        y1, y2, y3, y4 = self.forward_core(x)
        assert y1.dtype == torch.float32
        y = y1 + y2 + y3 + y4
        y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        y = self.out_norm(y)
        y = y * F.silu(z)
        out = self.out_proj(y)
        if self.dropout is not None:
            out = self.dropout(out)
        return out


class LFSSBlock(nn.Module):
    def __init__(
            self,
            hidden_dim: int = 0,
            drop_path: float = 0,
            norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
            attn_drop_rate: float = 0,
            d_state: int = 16,
            expand: float = 2.,
            **kwargs,
    ):
        super().__init__()
        self.ln_1 = norm_layer(hidden_dim)
        self.self_attention = SS2D(d_model=hidden_dim, d_state=d_state, expand=expand, dropout=attn_drop_rate, **kwargs)
        self.drop_path = DropPath(drop_path)
        self.skip_scale = nn.Parameter(torch.ones(hidden_dim))
        self.conv_blk = ffn(hidden_dim)
        self.ln_2 = nn.LayerNorm(hidden_dim)
        self.skip_scale2 = nn.Parameter(torch.ones(hidden_dim))

    def forward(self, input, x_size):
        # x [B,HW,C]
        B, L, C = input.shape
        input = input.view(B, *x_size, C).contiguous()  # [B,H,W,C]
        x = self.ln_1(input)
        x = input * self.skip_scale + self.drop_path(self.self_attention(x))
        x = x * self.skip_scale2 + self.conv_blk(self.ln_2(x).permute(0, 3, 1, 2).contiguous()).permute(0, 2, 3, 1).contiguous()
        x = x.view(B, -1, C).contiguous()
        return x


class MambaBlock(nn.Module):
    def __init__(self, dim, n_l_blocks=1, expand=2):
        super().__init__()
        self.l_blk = nn.Sequential(*[LFSSBlock(dim) for _ in range(n_l_blocks)])

    def forward(self, x_LL):
        b, c, h, w = x_LL.shape
        x_LL = rearrange(x_LL, "b c h w -> b (h w) c").contiguous()
        for l_layer in self.l_blk:
            x_LL = l_layer(x_LL, [h, w])
        x_LL = rearrange(x_LL, "b (h w) c -> b c h w", h=h, w=w).contiguous()

        return x_LL


class UNetConvBlock(nn.Module):
    def __init__(self, in_chans, out_chans):
        super(UNetConvBlock, self).__init__()
        block = []
        block.append(HA_LWT(in_chans, out_chans))

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

        # 拼接上采样后的特征图和跳跃连接特征图
        out = torch.cat([up, bridge], dim=1)

        # 通过卷积块处理拼接后的特征图
        out = self.conv_block(out)
        return out


def conv(in_channels, out_channels, kernel_size, bias=False, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size, padding=(kernel_size // 2), bias=bias, stride=stride)


class Predictor(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Predictor, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels, bias=False),
            nn.GELU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        )

    def forward(self, x):
        return self.net(x)


class Updater(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Updater, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels, bias=False),
            nn.GELU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        )

    def forward(self, x):
        return self.net(x)


class HA_LWT(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(HA_LWT, self).__init__()
        self.unshuffle = nn.PixelUnshuffle(2)
        self.P = Predictor(in_ch, 3 * in_ch)
        self.U = Updater(3 * in_ch, in_ch)
        self.conv_bn_relu = nn.Sequential(
            nn.Conv2d(in_ch * 4, out_ch, kernel_size=1, stride=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x_split = self.unshuffle(x)
        c = x.shape[1]
        x_approx = x_split[:, :c, :, :]
        x_detail = x_split[:, c:, :, :]

        d = x_detail - self.P(x_approx)
        c_new = x_approx + self.U(d)

        out = torch.cat([c_new, d], dim=1)
        out = self.conv_bn_relu(out)
        return out


class HazeDensityEstimator(nn.Module):
    """
    空间信噪比评估器 (Spatial SNR Estimator)
    利用第一阶段的低频输出预测空间信噪比图 (SNR Map)
    输出值越接近 1 代表雾越薄（信噪比高），越接近 0 代表雾越浓（细节丢失严重）
    """

    def __init__(self, in_channels=3):
        super(HazeDensityEstimator, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=3, padding=1),
            nn.Sigmoid()  # 将权重归一化到 0~1 之间，作为软路由权重
        )

    def forward(self, low_freq_img):
        return self.net(low_freq_img)


class DynamicDetailConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(DynamicDetailConv, self).__init__()
        padding = kernel_size // 2

        # 专家 1：负责“薄雾区”的高频细节锐化与纹理提取 (Clear Expert)
        self.expert_clear = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)

        # 专家 2：负责“浓雾区”的平滑去噪与结构修复 (Dense Expert)
        # 创新点：使用 dilation=2 扩大感受野，更利于浓雾区域的上下文修复
        self.expert_dense = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, dilation=2)

        # 可选：特征融合层
        self.fuse = nn.Conv2d(out_channels, out_channels, kernel_size=1)

    def forward(self, x, snr_map):
        # snr_map 维度: [B, 1, H, W]
        feat_clear = self.expert_clear(x)
        feat_dense = self.expert_dense(x)

        # 像素级软路由匹配 (Pixel-wise Soft Routing)：
        # 信噪比高的地方倾向于用 expert_clear，低的地方倾向于用 expert_dense
        out = snr_map * feat_clear + (1.0 - snr_map) * feat_dense

        return self.fuse(out)


class SNR_DDE_Module(nn.Module):
    def __init__(self, channels=3):
        super(SNR_DDE_Module, self).__init__()
        self.estimator = HazeDensityEstimator(in_channels=channels)
        self.dynamic_conv = DynamicDetailConv(channels, channels)

    def forward(self, x_stage1, output_LL):
        # 1. 用第一阶段重建的干净低频图 (output_LL) 评估信噪比
        snr_map = self.estimator(output_LL)
        # 2. 对第一阶段的初步去雾图 (x_stage1) 进行动态特征重塑
        x_refined = self.dynamic_conv(x_stage1, snr_map)
        return x_refined, snr_map


class UNet(nn.Module):
    def __init__(self, in_chn=3, wf=16, n_l_blocks=[1, 2, 2, 4], ffn_scale=2, up_mode='upconv', conv1=conv):
        super(UNet, self).__init__()

        assert up_mode in ('upconv', 'upsample')

        self.layer0 = conv1(in_chn, wf, kernel_size=3, stride=1)

        self.layer1 = UNetConvBlock(in_chans=16, out_chans=32)
        self.layer2 = UNetConvBlock(in_chans=32, out_chans=64)
        self.layer3 = UNetConvBlock(in_chans=64, out_chans=128)
        self.layer4 = UNetConvBlock(in_chans=128, out_chans=256)
        self.layer_0 = UNetUpBlock(in_chans=32, out_chans=16, up_mode=up_mode)
        self.layer_1 = UNetUpBlock(in_chans=64, out_chans=32, up_mode=up_mode)
        self.layer_2 = UNetUpBlock(in_chans=128, out_chans=64, up_mode=up_mode)
        self.layer_3 = UNetUpBlock(in_chans=256, out_chans=128, up_mode=up_mode)

        self.last = conv1(wf, in_chn, kernel_size=3, stride=1)

        self.down_group0 = MambaBlock(16, n_l_blocks=n_l_blocks[0], expand=ffn_scale)
        self.down_group1 = MambaBlock(32, n_l_blocks=n_l_blocks[0], expand=ffn_scale)
        self.down_group2 = MambaBlock(64, n_l_blocks=n_l_blocks[1], expand=ffn_scale)
        self.down_group3 = MambaBlock(128, n_l_blocks=n_l_blocks[2], expand=ffn_scale)
        self.down_group4 = MambaBlock(256, n_l_blocks=n_l_blocks[3], expand=ffn_scale)

        # decoder of UNet-64
        self.up_group4 = MambaBlock(128, n_l_blocks=n_l_blocks[3], expand=ffn_scale)
        self.up_group3 = MambaBlock(64, n_l_blocks=n_l_blocks[2], expand=ffn_scale)
        self.up_group2 = MambaBlock(32, n_l_blocks=n_l_blocks[1], expand=ffn_scale)
        self.up_group1 = MambaBlock(16, n_l_blocks=n_l_blocks[0], expand=ffn_scale)

        self.snr_dde = SNR_DDE_Module(channels=in_chn)
        self.DE = DENet(3, 6)  # self.DE = DENet(3, 4) for real_haze


    def forward(self, x):

        dwt, idwt = DWT(), IWT()
        n, c, h, w = x.shape

        x_dwt = dwt(x)
        x_LL, x_high0 = x_dwt[:n, ...], x_dwt[n:, ...]

        blocks = []
        x0 = self.layer0(x_LL)
        x0 = self.down_group0(x0)
        blocks.append(x0)

        x1 = self.layer1(x0)
        x1 = self.down_group1(x1)
        blocks.append(x1)

        x2 = self.layer2(x1)
        x2 = self.down_group2(x2)
        blocks.append(x2)

        x3 = self.layer3(x2)
        x3 = self.down_group3(x3)
        blocks.append(x3)

        x4 = self.layer4(x3)
        x4 = self.down_group4(x4)

        x_3 = self.layer_3(x4, blocks[-0 - 1])
        x_3 = self.up_group4(x_3)

        x_2 = self.layer_2(x_3, blocks[-1 - 1])
        x_2 = self.up_group3(x_2)

        x_1 = self.layer_1(x_2, blocks[-2 - 1])
        x_1 = self.up_group2(x_1)

        x_0 = self.layer_0(x_1, blocks[-3 - 1])
        x_0 = self.up_group1(x_0)
        x_0 = self.last(x_0)

        output_LL = x_0 + x_LL

        x_stage1 = idwt(torch.cat((output_LL, x_high0), dim=0))

        x_refined, snr_map = self.snr_dde(x_stage1, output_LL)

        x_final = self.DE(x_refined)

        if self.training:
            return output_LL, x_stage1, x_final
        else:
            return x_final


@ARCH_REGISTRY.register()
class WaveMamba(nn.Module):
    def __init__(self,
                 *,
                 in_chn,
                 wf,
                 n_l_blocks=[1, 2, 2, 4],
                 ffn_scale=2.0,
                 **ignore_kwargs):
        super().__init__()
        self.restoration_network = UNet(in_chn=in_chn, wf=wf, n_l_blocks=n_l_blocks, ffn_scale=ffn_scale)

    def print_network(self, model):
        num_params = 0
        for p in model.parameters():
            num_params += p.numel()
        print(model)
        print("The number of parameters: {}".format(num_params))

    def encode_and_decode(self, input, current_iter=None):

        restoration = self.restoration_network(input)
        return restoration

    @torch.no_grad()
    def test_tile(self, input, tile_size=240, tile_pad=16):
        # return self.test(input)
        """It will first crop input images to tiles, and then process each tile.
        Finally, all the processed tiles are merged into one images.
        Modified from: https://github.com/xinntao/Real-ESRGAN/blob/master/realesrgan/utils.py
        """
        batch, channel, height, width = input.shape
        output_height = height * self.scale_factor
        output_width = width * self.scale_factor
        output_shape = (batch, channel, output_height, output_width)

        # start with black image
        output = input.new_zeros(output_shape)
        tiles_x = math.ceil(width / tile_size)
        tiles_y = math.ceil(height / tile_size)

        # loop over all tiles
        for y in range(tiles_y):
            for x in range(tiles_x):
                # extract tile from input image
                ofs_x = x * tile_size
                ofs_y = y * tile_size
                # input tile area on total image
                input_start_x = ofs_x
                input_end_x = min(ofs_x + tile_size, width)
                input_start_y = ofs_y
                input_end_y = min(ofs_y + tile_size, height)

                # input tile area on total image with padding
                input_start_x_pad = max(input_start_x - tile_pad, 0)
                input_end_x_pad = min(input_end_x + tile_pad, width)
                input_start_y_pad = max(input_start_y - tile_pad, 0)
                input_end_y_pad = min(input_end_y + tile_pad, height)

                # input tile dimensions
                input_tile_width = input_end_x - input_start_x
                input_tile_height = input_end_y - input_start_y
                tile_idx = y * tiles_x + x + 1
                input_tile = input[:, :, input_start_y_pad:input_end_y_pad, input_start_x_pad:input_end_x_pad]

                # upscale tile
                output_tile = self.test(input_tile)

                # output tile area on total image
                output_start_x = input_start_x * self.scale_factor
                output_end_x = input_end_x * self.scale_factor
                output_start_y = input_start_y * self.scale_factor
                output_end_y = input_end_y * self.scale_factor

                # output tile area without padding
                output_start_x_tile = (input_start_x - input_start_x_pad) * self.scale_factor
                output_end_x_tile = output_start_x_tile + input_tile_width * self.scale_factor
                output_start_y_tile = (input_start_y - input_start_y_pad) * self.scale_factor
                output_end_y_tile = output_start_y_tile + input_tile_height * self.scale_factor

                # put tile into output image
                output[:, :, output_start_y:output_end_y,
                output_start_x:output_end_x] = output_tile[:, :, output_start_y_tile:output_end_y_tile,
                                               output_start_x_tile:output_end_x_tile]
        return output

    def check_image_size(self, x, window_size=8):
        _, _, h, w = x.size()
        mod_pad_h = (window_size - h % (window_size)) % (
            window_size)
        mod_pad_w = (window_size - w % (window_size)) % (
            window_size)
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        return x

    @torch.no_grad()
    def test(self, input):
        _, _, h_old, w_old = input.shape

        restoration = self.encode_and_decode(input)

        output = restoration

        return output

    def forward(self, input):

        restoration = self.encode_and_decode(input)

        return restoration


if __name__ == '__main__':
    import os

    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.randn(1, 3, 1920, 1280).to(device)
    model = UNet(in_chn=3, wf=32, n_l_blocks=[1, 2, 4], n_h_blocks=[1, 1, 1], ffn_scale=2).to(device)
    #    print(model)
    inp_shape = (3, 512, 512)
    from ptflops import get_model_complexity_info

    FLOPS = 0
    macs, params = get_model_complexity_info(model, inp_shape, verbose=False, print_per_layer_stat=True)

    params = float(params[:-4])
    print('mac', macs)
    print(params)
    macs = float(macs[:-4]) + FLOPS / 10 ** 9

    print('mac', macs)
    print(f'params: {sum(map(lambda x: x.numel(), model.parameters()))}')
    # print(flop_count_table(FlopCountAnalysis(model, x), activations=ActivationCountAnalysis(model, x)))
    with torch.no_grad():
        torch.cuda.reset_max_memory_allocated(device)
        start_time = time.time()
        output = model(x)
        end_time = time.time()
        memory_used = torch.cuda.max_memory_allocated(device)
    running_time = end_time - start_time
    print(output.shape)
    print(running_time)
    print(f"Memory used: {memory_used / 1024 ** 3:.3f} GB")