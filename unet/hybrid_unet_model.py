# File: unet/hybrid_unet_model.py

import torch
import torch.nn as nn
import torch.nn.functional as F

# --- CBAM Attention Module ---
class CBAM(nn.Module):
    def __init__(self, channels, reduction=16):
        super(CBAM, self).__init__()
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channels // reduction, channels, 1, bias=False),
            nn.Sigmoid()
        )
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, 1, 7, padding=3, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Channel Attention
        ca = self.channel_attention(x)
        x = x * ca
        # Spatial Attention
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        sa = self.spatial_attention(torch.cat([avg_out, max_out], dim=1))
        x = x * sa
        return x

# --- Basic Convolution Block with Residual ---
class ResidualDoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualDoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        if in_channels != out_channels:
            self.residual_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.residual_conv = nn.Identity()

    def forward(self, x):
        residual = self.residual_conv(x)
        x = self.double_conv(x)
        return x + residual

# --- Downsampling Block ---
class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            ResidualDoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

# --- Upsampling Block ---
class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super(Up, self).__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = ResidualDoubleConv(in_channels, out_channels)
        else:
            self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)
            self.conv = ResidualDoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

# --- Final Output Layer ---
class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

# --- Full Hybrid U-Net ---
class HybridUNet(nn.Module):
    def __init__(self, n_channels=1, n_classes=1, bilinear=True):
        super(HybridUNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = ResidualDoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)

        factor = 2 if bilinear else 1

        self.bottleneck = nn.Sequential(
            nn.Conv2d(512, 1024 // factor, kernel_size=3, padding=2, dilation=2),
            nn.BatchNorm2d(1024 // factor),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024 // factor, 1024 // factor, kernel_size=3, padding=2, dilation=2),
            nn.BatchNorm2d(1024 // factor),
            nn.ReLU(inplace=True)
        )

        # CBAM Attention after bottleneck
        self.cbam_bottleneck = CBAM(1024 // factor)

        # Decoder (Upsampling) with CBAM after each up block
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.cbam1 = CBAM(512 // factor)

        self.up2 = Up(512, 256 // factor, bilinear)
        self.cbam2 = CBAM(256 // factor)

        self.up3 = Up(256, 128 // factor, bilinear)
        self.cbam3 = CBAM(128 // factor)

        self.up4 = Up(128, 64, bilinear)
        self.cbam4 = CBAM(64)

        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.bottleneck(x4)
        x5 = self.cbam_bottleneck(x5)

        x = self.up1(x5, x4)
        x = self.cbam1(x)

        x = self.up2(x, x3)
        x = self.cbam2(x)

        x = self.up3(x, x2)
        x = self.cbam3(x)

        x = self.up4(x, x1)
        x = self.cbam4(x)

        logits = self.outc(x)
        return logits
