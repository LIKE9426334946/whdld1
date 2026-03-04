import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

from .attention import SimAM, CBAM

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)

class UpBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch, use_cbam=True):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.conv = ConvBlock(out_ch + skip_ch, out_ch)
        self.attn = CBAM(out_ch) if use_cbam else nn.Identity()

    def forward(self, x, skip):
        x = self.up(x)
        # handle odd size mismatch if any
        if x.size(-1) != skip.size(-1) or x.size(-2) != skip.size(-2):
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        x = self.conv(x)
        x = self.attn(x)
        return x

class UNetResNet34Attn(nn.Module):
    def __init__(self, num_classes: int, simam_in_encoder=True, cbam_in_decoder=True, pretrained=True):
        super().__init__()
        base = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1 if pretrained else None)

        self.stem = nn.Sequential(
            base.conv1, base.bn1, base.relu
        )  # /2
        self.maxpool = base.maxpool  # /4
        self.enc1 = base.layer1       # /4
        self.enc2 = base.layer2       # /8
        self.enc3 = base.layer3       # /16
        self.enc4 = base.layer4       # /32

        self.simam = SimAM() if simam_in_encoder else nn.Identity()

        # Channels: stem=64, enc1=64, enc2=128, enc3=256, enc4=512
        self.center = ConvBlock(512, 512)

        self.dec4 = UpBlock(512, 256, 256, use_cbam=cbam_in_decoder)  # /16
        self.dec3 = UpBlock(256, 128, 128, use_cbam=cbam_in_decoder)  # /8
        self.dec2 = UpBlock(128, 64,  64,  use_cbam=cbam_in_decoder)  # /4
        # 这里用 stem 作为 /2 skip
        self.dec1 = UpBlock(64,  64,  64,  use_cbam=cbam_in_decoder)  # /2

        self.final_up = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),  # back to /1
            nn.ReLU(inplace=True),
            nn.Conv2d(32, num_classes, kernel_size=1)
        )

    def forward(self, x):
        s0 = self.stem(x)                 # /2, 64
        x1 = self.maxpool(s0)             # /4
        e1 = self.enc1(x1)                # /4, 64
        e2 = self.enc2(e1)                # /8, 128
        e3 = self.enc3(e2)                # /16, 256
        e4 = self.enc4(e3)                # /32, 512

        # SimAM on encoder features (lightweight)
        e1 = self.simam(e1)
        e2 = self.simam(e2)
        e3 = self.simam(e3)
        e4 = self.simam(e4)

        c = self.center(e4)

        d4 = self.dec4(c, e3)
        d3 = self.dec3(d4, e2)
        d2 = self.dec2(d3, e1)
        d1 = self.dec1(d2, s0)

        out = self.final_up(d1)
        return out