""" Full assembly of the parts to form the complete ResUNet network with CBAM attention """

from .unet_parts import *


class CBRDilatedUNet(nn.Module):
    """CBR-DilatedUNet: CBAM + Residual + Dilated convolutions in bottleneck"""
    
    def __init__(self, n_channels, n_classes, bilinear=False, use_attention=True):
        super(CBRDilatedUNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.use_attention = use_attention

        # Encoder path with residual blocks and CBAM attention
        self.inc = DoubleConv(n_channels, 64, use_attention=use_attention)
        self.down1 = Down(64, 128, use_attention=use_attention)
        self.down2 = Down(128, 256, use_attention=use_attention)
        self.down3 = Down(256, 512, use_attention=use_attention)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor, use_attention=use_attention)
        
        # Dilated bottleneck for multi-scale feature extraction
        self.dilated_bottleneck = DilatedBottleneck(1024 // factor, 1024 // factor, use_attention=use_attention)
        
        # Decoder path with residual blocks and CBAM attention
        self.up1 = Up(1024, 512 // factor, bilinear, use_attention=use_attention)
        self.up2 = Up(512, 256 // factor, bilinear, use_attention=use_attention)
        self.up3 = Up(256, 128 // factor, bilinear, use_attention=use_attention)
        self.up4 = Up(128, 64, bilinear, use_attention=use_attention)
        
        # Output convolution
        self.outc = OutConv(64, n_classes)
        
        # Additional residual connections for better gradient flow
        self.residual_conv1 = nn.Conv2d(64, 64, kernel_size=1)
        self.residual_conv2 = nn.Conv2d(128, 128, kernel_size=1)
        self.residual_conv3 = nn.Conv2d(256, 256, kernel_size=1)
        self.residual_conv4 = nn.Conv2d(512, 512, kernel_size=1)

    def forward(self, x):
        # Encoder path with residual connections and CBAM attention
        x1 = self.inc(x)
        x1_res = self.residual_conv1(x1) + x1  # Residual connection
        
        x2 = self.down1(x1_res)
        x2_res = self.residual_conv2(x2) + x2  # Residual connection
        
        x3 = self.down2(x2_res)
        x3_res = self.residual_conv3(x3) + x3  # Residual connection
        
        x4 = self.down3(x3_res)
        x4_res = self.residual_conv4(x4) + x4  # Residual connection
        
        x5 = self.down4(x4_res)
        
        # Apply dilated bottleneck for multi-scale feature extraction
        x5 = self.dilated_bottleneck(x5)
        
        # Decoder path with skip connections and CBAM attention
        x = self.up1(x5, x4_res)
        x = self.up2(x, x3_res)
        x = self.up3(x, x2_res)
        x = self.up4(x, x1_res)
        
        logits = self.outc(x)
        return logits

    def use_checkpointing(self):
        self.inc = torch.utils.checkpoint(self.inc)
        self.down1 = torch.utils.checkpoint(self.down1)
        self.down2 = torch.utils.checkpoint(self.down2)
        self.down3 = torch.utils.checkpoint(self.down3)
        self.down4 = torch.utils.checkpoint(self.down4)
        self.dilated_bottleneck = torch.utils.checkpoint(self.dilated_bottleneck)
        self.up1 = torch.utils.checkpoint(self.up1)
        self.up2 = torch.utils.checkpoint(self.up2)
        self.up3 = torch.utils.checkpoint(self.up3)
        self.up4 = torch.utils.checkpoint(self.up4)
        self.outc = torch.utils.checkpoint(self.outc)


class CBAMResUNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False, use_attention=True):
        super(CBAMResUNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.use_attention = use_attention

        # Encoder path with residual blocks and CBAM attention
        self.inc = DoubleConv(n_channels, 64, use_attention=use_attention)
        self.down1 = Down(64, 128, use_attention=use_attention)
        self.down2 = Down(128, 256, use_attention=use_attention)
        self.down3 = Down(256, 512, use_attention=use_attention)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor, use_attention=use_attention)
        
        # Decoder path with residual blocks and CBAM attention
        self.up1 = Up(1024, 512 // factor, bilinear, use_attention=use_attention)
        self.up2 = Up(512, 256 // factor, bilinear, use_attention=use_attention)
        self.up3 = Up(256, 128 // factor, bilinear, use_attention=use_attention)
        self.up4 = Up(128, 64, bilinear, use_attention=use_attention)
        
        # Output convolution
        self.outc = OutConv(64, n_classes)
        
        # Additional residual connections for better gradient flow
        self.residual_conv1 = nn.Conv2d(64, 64, kernel_size=1)
        self.residual_conv2 = nn.Conv2d(128, 128, kernel_size=1)
        self.residual_conv3 = nn.Conv2d(256, 256, kernel_size=1)
        self.residual_conv4 = nn.Conv2d(512, 512, kernel_size=1)

    def forward(self, x):
        # Encoder path with residual connections and CBAM attention
        x1 = self.inc(x)
        x1_res = self.residual_conv1(x1) + x1  # Residual connection
        
        x2 = self.down1(x1_res)
        x2_res = self.residual_conv2(x2) + x2  # Residual connection
        
        x3 = self.down2(x2_res)
        x3_res = self.residual_conv3(x3) + x3  # Residual connection
        
        x4 = self.down3(x3_res)
        x4_res = self.residual_conv4(x4) + x4  # Residual connection
        
        x5 = self.down4(x4_res)
        
        # Decoder path with skip connections and CBAM attention
        x = self.up1(x5, x4_res)
        x = self.up2(x, x3_res)
        x = self.up3(x, x2_res)
        x = self.up4(x, x1_res)
        
        logits = self.outc(x)
        return logits

    def use_checkpointing(self):
        self.inc = torch.utils.checkpoint(self.inc)
        self.down1 = torch.utils.checkpoint(self.down1)
        self.down2 = torch.utils.checkpoint(self.down2)
        self.down3 = torch.utils.checkpoint(self.down3)
        self.down4 = torch.utils.checkpoint(self.down4)
        self.up1 = torch.utils.checkpoint(self.up1)
        self.up2 = torch.utils.checkpoint(self.up2)
        self.up3 = torch.utils.checkpoint(self.up3)
        self.up4 = torch.utils.checkpoint(self.up4)
        self.outc = torch.utils.checkpoint(self.outc)


# Keep AttentionResUNet as an alias for backward compatibility
class AttentionResUNet(CBAMResUNet):
    def __init__(self, n_channels, n_classes, bilinear=False, attention_type='cbam'):
        # Ignore attention_type parameter, always use CBAM
        super().__init__(n_channels, n_classes, bilinear, use_attention=True)


class ResUNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(ResUNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        # Encoder path with residual blocks
        self.inc = DoubleConv(n_channels, 64, use_attention=False)
        self.down1 = Down(64, 128, use_attention=False)
        self.down2 = Down(128, 256, use_attention=False)
        self.down3 = Down(256, 512, use_attention=False)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor, use_attention=False)
        
        # Decoder path with residual blocks
        self.up1 = Up(1024, 512 // factor, bilinear, use_attention=False)
        self.up2 = Up(512, 256 // factor, bilinear, use_attention=False)
        self.up3 = Up(256, 128 // factor, bilinear, use_attention=False)
        self.up4 = Up(128, 64, bilinear, use_attention=False)
        
        # Output convolution
        self.outc = OutConv(64, n_classes)
        
        # Additional residual connections for better gradient flow
        self.residual_conv1 = nn.Conv2d(64, 64, kernel_size=1)
        self.residual_conv2 = nn.Conv2d(128, 128, kernel_size=1)
        self.residual_conv3 = nn.Conv2d(256, 256, kernel_size=1)
        self.residual_conv4 = nn.Conv2d(512, 512, kernel_size=1)

    def forward(self, x):
        # Encoder path with residual connections
        x1 = self.inc(x)
        x1_res = self.residual_conv1(x1) + x1  # Residual connection
        
        x2 = self.down1(x1_res)
        x2_res = self.residual_conv2(x2) + x2  # Residual connection
        
        x3 = self.down2(x2_res)
        x3_res = self.residual_conv3(x3) + x3  # Residual connection
        
        x4 = self.down3(x3_res)
        x4_res = self.residual_conv4(x4) + x4  # Residual connection
        
        x5 = self.down4(x4_res)
        
        # Decoder path with skip connections
        x = self.up1(x5, x4_res)
        x = self.up2(x, x3_res)
        x = self.up3(x, x2_res)
        x = self.up4(x, x1_res)
        
        logits = self.outc(x)
        return logits

    def use_checkpointing(self):
        self.inc = torch.utils.checkpoint(self.inc)
        self.down1 = torch.utils.checkpoint(self.down1)
        self.down2 = torch.utils.checkpoint(self.down2)
        self.down3 = torch.utils.checkpoint(self.down3)
        self.down4 = torch.utils.checkpoint(self.down4)
        self.up1 = torch.utils.checkpoint(self.up1)
        self.up2 = torch.utils.checkpoint(self.up2)
        self.up3 = torch.utils.checkpoint(self.up3)
        self.up4 = torch.utils.checkpoint(self.up4)
        self.outc = torch.utils.checkpoint(self.outc)


# Keep the original UNet for backward compatibility
class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = (DoubleConv(n_channels, 64, use_attention=False))
        self.down1 = (Down(64, 128, use_attention=False))
        self.down2 = (Down(128, 256, use_attention=False))
        self.down3 = (Down(256, 512, use_attention=False))
        factor = 2 if bilinear else 1
        self.down4 = (Down(512, 1024 // factor, use_attention=False))
        self.up1 = (Up(1024, 512 // factor, bilinear, use_attention=False))
        self.up2 = (Up(512, 256 // factor, bilinear, use_attention=False))
        self.up3 = (Up(256, 128 // factor, bilinear, use_attention=False))
        self.up4 = (Up(128, 64, bilinear, use_attention=False))
        self.outc = (OutConv(64, n_classes))

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

    def use_checkpointing(self):
        self.inc = torch.utils.checkpoint(self.inc)
        self.down1 = torch.utils.checkpoint(self.down1)
        self.down2 = torch.utils.checkpoint(self.down2)
        self.down3 = torch.utils.checkpoint(self.down3)
        self.down4 = torch.utils.checkpoint(self.down4)
        self.up1 = torch.utils.checkpoint(self.up1)
        self.up2 = torch.utils.checkpoint(self.up2)
        self.up3 = torch.utils.checkpoint(self.up3)
        self.up4 = torch.utils.checkpoint(self.up4)
        self.outc = torch.utils.checkpoint(self.outc)