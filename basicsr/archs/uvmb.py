import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm import Mamba


def conv3x3(in_channels, out_channels, bias=False):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=bias)


class UVMB(nn.Module):
    """基于 Mamba 的 UVMB"""
    def __init__(
        self,
        in_channels,
        mid_size=64,
        d_state=16,
        d_conv=4,
        expand=2,
    ):
        super().__init__()
        self.in_ch = in_channels
        self.mid_size = mid_size

        self.convb = nn.Sequential(
            conv3x3(in_channels, in_channels, bias=False),
            nn.ReLU(inplace=True),
            conv3x3(in_channels, in_channels, bias=False)
        )

        self.norm1 = nn.LayerNorm(in_channels)
        self.norm2 = nn.LayerNorm(in_channels)

        self.mamba_main = Mamba(
            d_model=in_channels,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )

        self.mamba_gate = Mamba(
            d_model=in_channels,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )

        self.smooth_conv = nn.Sequential(
            conv3x3(in_channels, in_channels, bias=False),
            nn.ReLU(inplace=True),
            conv3x3(in_channels, in_channels, bias=False)
        )

    def forward(self, x):
        identity = x
        x = self.convb(x) + x

        b, c, h, w = x.shape

        if (h, w) != (self.mid_size, self.mid_size):
            x_small = F.interpolate(
                x, size=(self.mid_size, self.mid_size),
                mode='bilinear', align_corners=False
            )
        else:
            x_small = x

        tokens = x_small.flatten(2).transpose(1, 2).contiguous()  # B, N, C

        main_feat = self.mamba_main(self.norm1(tokens))
        gate_feat = self.mamba_gate(self.norm2(tokens))

        gate = torch.softmax(gate_feat, dim=1)
        tokens = tokens + gate * main_feat

        feat = tokens.transpose(1, 2).contiguous().view(
            b, c, self.mid_size, self.mid_size
        )

        if (h, w) != (self.mid_size, self.mid_size):
            feat = F.interpolate(
                feat, size=(h, w),
                mode='bilinear', align_corners=False
            )

        out = self.smooth_conv(feat)
        return out