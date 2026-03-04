import torch
import torch.nn as nn
import torch.nn.functional as F

class SimAM(nn.Module):
    """
    Parameter-free attention (SimAM).
    """
    def __init__(self, e_lambda: float = 1e-4):
        super().__init__()
        self.e_lambda = e_lambda

    def forward(self, x):
        # x: (B,C,H,W)
        b, c, h, w = x.size()
        n = h * w - 1

        x_minus_mu = x - x.mean(dim=(2,3), keepdim=True)
        var = (x_minus_mu ** 2).sum(dim=(2,3), keepdim=True) / (n + 1e-6)
        e = (x_minus_mu ** 2) / (4 * (var + self.e_lambda)) + 0.5
        attn = torch.sigmoid(e)
        return x * attn

class ChannelAttention(nn.Module):
    def __init__(self, in_ch, reduction=16):
        super().__init__()
        mid = max(1, in_ch // reduction)
        self.mlp = nn.Sequential(
            nn.Linear(in_ch, mid),
            nn.ReLU(inplace=True),
            nn.Linear(mid, in_ch),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        avg = F.adaptive_avg_pool2d(x, 1).view(b, c)
        mx  = F.adaptive_max_pool2d(x, 1).view(b, c)
        out = self.mlp(avg) + self.mlp(mx)
        scale = torch.sigmoid(out).view(b, c, 1, 1)
        return x * scale

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)

    def forward(self, x):
        avg = x.mean(dim=1, keepdim=True)
        mx, _ = x.max(dim=1, keepdim=True)
        feat = torch.cat([avg, mx], dim=1)
        attn = torch.sigmoid(self.conv(feat))
        return x * attn

class CBAM(nn.Module):
    def __init__(self, in_ch, reduction=16, spatial_kernel=7):
        super().__init__()
        self.ca = ChannelAttention(in_ch, reduction)
        self.sa = SpatialAttention(spatial_kernel)

    def forward(self, x):
        x = self.ca(x)
        x = self.sa(x)
        return x