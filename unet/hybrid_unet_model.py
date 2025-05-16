# File: unet/hybrid_unet_model.py

import torch
import torch.nn as nn
import torch.nn.functional as F

# --- Squeeze-and-Excitation (SE) Block ---
class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        scale = self.se(x)
        return x * scale

# --- CBAM + SE Attention Module ---
class CBAM_SE(nn.Module):
    def __init__(self, channels, reduction=16):
        super(CBAM_SE, self).__init__()
        self.se_block = SEBlock(channels, reduction)
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False),
            nn.Sigmoid()
        )
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, 1, 7, padding=3, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.se_block(x)
        ca = self.channel_attention(x)
        x = x * ca
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        sa = self.spatial_attention(torch.cat([avg_out, max_out], dim=1))
        return x * sa

# --- Residual Double Conv Block ---
class ResidualDoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, p_drop=0.2):
        super(ResidualDoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p_drop),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p_drop),
        )
        self.residual_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        return self.double_conv(x) + self.residual_conv(x)

# --- Downsampling Block ---
class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__()
        self.down = nn.Sequential(
            nn.MaxPool2d(2),
            ResidualDoubleConv(in_channels, out_channels)
        )
        self.cbam = CBAM_SE(out_channels)

    def forward(self, x):
        x = self.down(x)
        return self.cbam(x)

# --- Upsampling Block ---
class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super(Up, self).__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)
        self.conv = ResidualDoubleConv(in_channels, out_channels)
        self.cbam = CBAM_SE(out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.cbam(self.conv(x))

# --- Output Conv Layer ---
class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

# --- Full Hybrid U-Net with CBAM+SE ---
class HybridUNet(nn.Module):
    def __init__(self, n_channels=1, n_classes=1, bilinear=True, p_drop=0.1):
        super(HybridUNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        # Wider: double filters, Deeper: add 5th block
        self.inc = ResidualDoubleConv(n_channels, 128, p_drop)
        self.down1 = Down(128, 256)
        self.down2 = Down(256, 512)
        self.down3 = Down(512, 1024)
        self.down4 = Down(1024, 2048)

        factor = 2 if bilinear else 1
        self.bottleneck = nn.Sequential(
            nn.Conv2d(2048, 4096 // factor, kernel_size=3, padding=2, dilation=2),
            nn.InstanceNorm2d(4096 // factor),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p_drop),
            nn.Conv2d(4096 // factor, 4096 // factor, kernel_size=3, padding=2, dilation=2),
            nn.InstanceNorm2d(4096 // factor),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p_drop),
        )
        self.cbam_bottleneck = CBAM_SE(4096 // factor)

        self.up1 = Up(4096, 2048 // factor, bilinear)
        self.up2 = Up(2048, 1024 // factor, bilinear)
        self.up3 = Up(1024, 512 // factor, bilinear)
        self.up4 = Up(512, 256 // factor, bilinear)
        self.up5 = Up(256, 128, bilinear)

        self.outc = OutConv(128, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x6 = self.cbam_bottleneck(self.bottleneck(x5))
        x = self.up1(x6, x5)
        x = self.up2(x, x4)
        x = self.up3(x, x3)
        x = self.up4(x, x2)
        x = self.up5(x, x1)
        return self.outc(x)
