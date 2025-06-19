""" Parts of the U-Net model """

import torch
import torch.nn as nn
import torch.nn.functional as F


class CBAMModule(nn.Module):
    """Convolutional Block Attention Module"""
    
    def __init__(self, channels, reduction=16, kernel_size=7):
        super().__init__()
        # Channel attention (SE-like)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
        
        # Spatial attention
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Apply channel attention
        avg_out = self.fc(self.avg_pool(x).view(x.size(0), -1))
        max_out = self.fc(self.max_pool(x).view(x.size(0), -1))
        channel_out = (avg_out + max_out).view(x.size(0), x.size(1), 1, 1)
        x = x * channel_out.expand_as(x)
        
        # Apply spatial attention
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        spatial = torch.cat([avg_out, max_out], dim=1)
        spatial = self.spatial_attention(spatial)
        
        return x * spatial


class DilatedResidualBlock(nn.Module):
    """Dilated residual block with CBAM attention"""
    
    def __init__(self, in_channels, out_channels, dilation=1, use_attention=True):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                               padding=dilation, dilation=dilation, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, 
                               padding=dilation, dilation=dilation, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        # CBAM attention mechanism
        if use_attention:
            self.attention = CBAMModule(out_channels)
        else:
            self.attention = nn.Identity()
        
        # Skip connection: if input and output channels are different, use 1x1 conv
        self.skip = nn.Sequential()
        if in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        residual = self.skip(x)
        
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        
        # Apply attention
        out = self.attention(out)
        
        out += residual
        out = self.relu(out)
        
        return out


class DilatedBottleneck(nn.Module):
    """Dilated bottleneck with multiple dilation rates for multi-scale feature extraction"""
    
    def __init__(self, in_channels, out_channels, use_attention=True):
        super().__init__()
        
        # Multiple dilated convolutions with different rates
        self.dilated_conv1 = DilatedResidualBlock(in_channels, out_channels//4, 
                                                  dilation=1, use_attention=use_attention)
        self.dilated_conv2 = DilatedResidualBlock(out_channels//4, out_channels//4, 
                                                  dilation=2, use_attention=use_attention)
        self.dilated_conv3 = DilatedResidualBlock(out_channels//4, out_channels//4, 
                                                  dilation=4, use_attention=use_attention)
        self.dilated_conv4 = DilatedResidualBlock(out_channels//4, out_channels//4, 
                                                  dilation=8, use_attention=use_attention)
        
        # Final 1x1 convolution to combine features
        self.final_conv = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        # CBAM attention for the final output
        if use_attention:
            self.attention = CBAMModule(out_channels)
        else:
            self.attention = nn.Identity()

    def forward(self, x):
        # Apply dilated convolutions in parallel
        out1 = self.dilated_conv1(x)
        out2 = self.dilated_conv2(out1)
        out3 = self.dilated_conv3(out2)
        out4 = self.dilated_conv4(out3)
        
        # Concatenate all outputs
        out = torch.cat([out1, out2, out3, out4], dim=1)
        
        # Final convolution and attention
        out = self.final_conv(out)
        out = self.attention(out)
        
        return out


class ResidualBlock(nn.Module):
    """Residual block with two convolutions, skip connection, and CBAM attention"""
    
    def __init__(self, in_channels, out_channels, use_attention=True):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        # CBAM attention mechanism
        if use_attention:
            self.attention = CBAMModule(out_channels)
        else:
            self.attention = nn.Identity()
        
        # Skip connection: if input and output channels are different, use 1x1 conv
        self.skip = nn.Sequential()
        if in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        residual = self.skip(x)
        
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        
        # Apply attention
        out = self.attention(out)
        
        out += residual
        out = self.relu(out)
        
        return out


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2 with residual connection and CBAM attention"""

    def __init__(self, in_channels, out_channels, mid_channels=None, use_attention=True):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        
        # Use residual blocks with CBAM attention instead of simple double conv
        self.residual_block1 = ResidualBlock(in_channels, mid_channels, use_attention)
        self.residual_block2 = ResidualBlock(mid_channels, out_channels, use_attention)

    def forward(self, x):
        x = self.residual_block1(x)
        x = self.residual_block2(x)
        return x


class Down(nn.Module):
    """Downscaling with maxpool then double conv with residual and CBAM attention"""

    def __init__(self, in_channels, out_channels, use_attention=True):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels, use_attention=use_attention)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv with residual and CBAM attention"""

    def __init__(self, in_channels, out_channels, bilinear=True, use_attention=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2, use_attention)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels, use_attention=use_attention)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)
