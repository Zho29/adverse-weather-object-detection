"""
Feature Recovery (FR) Module for DSNet
Generates restored images from CB features for visibility enhancement
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FR_Module(nn.Module):
    """
    Feature Recovery Module - Restoration Subnetwork
    
    Takes features from CB module and generates restored image
    Following DSNet paper architecture
    """
    
    def __init__(self, in_channels=256, num_upsamples=2):
        """
        Args:
            in_channels: Input channels from CB module (256 for C2)
            num_upsamples: Number of upsampling stages (2 for C2 → original)
        """
        super(FR_Module, self).__init__()
        
        self.num_upsamples = num_upsamples
        
        # Upsampling layers (bilinear interpolation as in paper)
        # Paper found bilinear works better than deconvolution
        
        # Convolutional layers after each upsampling
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        # Final layer to output RGB image
        self.output_conv = nn.Conv2d(64, 3, kernel_size=3, padding=1)
        
        # Output activation (Tanh to get [-1, 1] or Sigmoid for [0, 1])
        self.output_activation = nn.Tanh()
        
    def forward(self, features):
        """
        Args:
            features: Features from CB module [B, in_channels, H/4, W/4]
        Returns:
            restored_img: Restored image [B, 3, H, W]
        """
        # First upsampling (×2)
        x = F.interpolate(features, scale_factor=2, mode='bilinear', align_corners=False)
        x = self.conv1(x)  # [B, 128, H/2, W/2]
        
        # Second upsampling (×2)
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        x = self.conv2(x)  # [B, 64, H, W]
        
        # Output RGB
        x = self.output_conv(x)  # [B, 3, H, W]
        
        # Output activation
        restored_img = self.output_activation(x)
        
        return restored_img


class FR_Module_Advanced(nn.Module):
    """
    Advanced FR Module with skip connections and residual blocks
    For potentially better restoration quality
    """
    
    def __init__(self, in_channels=256):
        super(FR_Module_Advanced, self).__init__()
        
        # Residual blocks before upsampling
        self.res_block1 = self._make_res_block(in_channels, in_channels)
        self.res_block2 = self._make_res_block(in_channels, in_channels)
        
        # Upsampling path
        self.up1 = nn.Sequential(
            nn.Conv2d(in_channels, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        
        self.up2 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        # Output
        self.output_conv = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 3, kernel_size=3, padding=1)
        )
        
        self.output_activation = nn.Tanh()
        
    def _make_res_block(self, in_ch, out_ch):
        """Create a residual block"""
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch)
        )
        
    def forward(self, features):
        """Forward with residual connections"""
        # Residual processing
        identity = features
        x = self.res_block1(features)
        x = x + identity  # Residual connection
        
        identity = x
        x = self.res_block2(x)
        x = x + identity  # Residual connection
        
        # Upsampling
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        x = self.up1(x)
        
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        x = self.up2(x)
        
        # Output
        x = self.output_conv(x)
        restored_img = self.output_activation(x)
        
        return restored_img


def test_fr_module():
    """Test FR module"""
    print("=" * 60)
    print("Testing FR Module")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")
    
    # Test basic FR module
    print("Testing FR_Module (basic)...")
    fr_module = FR_Module(in_channels=256).to(device)
    
    # Input from CB module (C2 output)
    cb_features = torch.randn(2, 256, 128, 128).to(device)
    print(f"Input features: {cb_features.shape}")
    
    with torch.no_grad():
        restored_img = fr_module(cb_features)
        print(f"Output image: {restored_img.shape}")
        print(f"Expected: [2, 3, 512, 512] (original size)")
        print(f"Output range: [{restored_img.min():.3f}, {restored_img.max():.3f}]")
    
    print("\n✓ Basic FR Module working!")
    
    # Test advanced FR module
    print("\n" + "-" * 60)
    print("Testing FR_Module_Advanced...")
    fr_advanced = FR_Module_Advanced(in_channels=256).to(device)
    
    with torch.no_grad():
        restored_img = fr_advanced(cb_features)
        print(f"Output image: {restored_img.shape}")
        print(f"Output range: [{restored_img.min():.3f}, {restored_img.max():.3f}]")
    
    print("\n✓ Advanced FR Module working!")
    
    # Test end-to-end with CB module
    print("\n" + "=" * 60)
    print("Testing CB + FR Pipeline...")
    
    from cb_module import CB_Module
    cb_module = CB_Module(pretrained=False).to(device)
    
    # Original input image
    input_img = torch.randn(2, 3, 512, 512).to(device)
    print(f"Input image: {input_img.shape}")
    
    with torch.no_grad():
        # CB features
        features = cb_module(input_img)
        print(f"CB features: {features.shape}")
        
        # FR restoration
        restored = fr_module(features)
        print(f"Restored image: {restored.shape}")
    
    print("\n✓ CB + FR pipeline working correctly!")
    print("=" * 60)
    
    return fr_module


if __name__ == '__main__':
    test_fr_module()
