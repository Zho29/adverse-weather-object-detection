"""WeatherUNet restoration model with a U-Net encoder, decoder, and skip connections.

This research implementation is based on the paper included in 2407.02581v1.pdf.
Paper-equivalent behavior and paper benchmark results have not been established
by this repository. Foggy Cityscapes training normalizes RGB images to [-1, 1]
and uses a three-channel tanh output.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """
    Double Convolution block used in UNet encoder and decoder.
    Each block consists of: Conv -> BatchNorm -> ReLU -> Conv -> BatchNorm -> ReLU
    """
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """
    Downsampling block: MaxPool -> DoubleConv
    Used in the encoder (contracting path)
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """
    Upsampling block: Upsample -> DoubleConv
    Used in the decoder (expansive path)
    Includes skip connections from encoder
    """
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # Use bilinear upsampling or transposed convolution
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        """
        Args:
            x1: feature map from decoder (to be upsampled)
            x2: skip connection from encoder (same resolution as upsampled x1)
        """
        x1 = self.up(x1)
        
        # Handle input sizes that don't match exactly (padding differences)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        
        # Concatenate skip connection
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    """
    Final output convolution to map to desired number of output channels
    """
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class WeatherUNet(nn.Module):
    """
    Weather UNet (WUNet) for adverse weather image restoration.
    
    Architecture:
    - Encoder: 4 downsampling blocks (64, 128, 256, 512 channels)
    - Bottleneck: 1024 channels
    - Decoder: 4 upsampling blocks with skip connections
    - Output: Clear weather image (same resolution as input)
    
    Args:
        n_channels: Number of input channels (3 for RGB images)
        n_classes: Number of output channels (3 for RGB images)
        bilinear: Use bilinear upsampling (True) or transposed convolution (False)
    """
    def __init__(self, n_channels=3, n_classes=3, bilinear=True):
        super(WeatherUNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        # Initial convolution (no downsampling)
        self.inc = DoubleConv(n_channels, 64)
        
        # Encoder (contracting path)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        
        # Adjust factor for bilinear upsampling
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        
        # Decoder (expansive path)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        
        # Output convolution
        self.outc = OutConv(64, n_classes)
        
        # Optional: Add Tanh activation for output normalization
        self.output_activation = nn.Tanh()

    def forward(self, x):
        """
        Forward pass through Weather UNet
        
        Args:
            x: Input tensor of shape (batch_size, n_channels, height, width)
            Typically adverse weather images
        
        Returns:
            Output tensor of shape (batch_size, n_classes, height, width)
            Denoised clear weather images
        """
        # Encoder
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        # Decoder with skip connections
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        
        # Output
        logits = self.outc(x)
        
        # Apply tanh to normalize output to [-1, 1] range
        output = self.output_activation(logits)
        
        return output


def test_weather_unet():
    """
    Test function to verify the Weather UNet architecture
    """
    # Create model
    model = WeatherUNet(n_channels=3, n_classes=3, bilinear=True)
    
    # Test with random input (batch_size=2, channels=3, height=256, width=256)
    x = torch.randn(2, 3, 256, 256)
    
    # Forward pass
    output = model(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Verify output shape matches input shape
    assert output.shape == x.shape, "Output shape should match input shape"
    print("✓ Architecture test passed!")
    
    return model


if __name__ == "__main__":
    print("Testing Weather UNet (WUNet) Architecture")
    print("=" * 50)
    model = test_weather_unet()
    print("=" * 50)
    print("\nModel Summary:")
    print(model)
