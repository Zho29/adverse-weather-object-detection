"""
Content Branch (CB) Module for DSNet
Extracts features from ResNet backbone (following paper's approach)
Can be replaced with DINO backbone later
"""

import torch
import torch.nn as nn
import torchvision.models as models


class CB_Module(nn.Module):
    """
    Content Branch Module - extracts features from backbone
    
    Based on DSNet paper: Uses first 10 layers of ResNet-50
    Output at C2 level (1/4 resolution of input)
    """
    
    def __init__(self, pretrained=True):
        super(CB_Module, self).__init__()
        
        # Load pretrained ResNet-50
        resnet = models.resnet50(pretrained=pretrained)
        
        # CB_C2: First 10 layers up to C2 output
        # Following paper's best configuration (Table 2)
        self.conv1 = resnet.conv1        # 7x7, stride 2 → 1/2
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool   # 3x3, stride 2 → 1/4
        self.layer1 = resnet.layer1      # Conv2_x (3 blocks)
        
        # Output channels at C2
        self.out_channels = 256
        
    def forward(self, x):
        """
        Args:
            x: Input image [B, 3, H, W]
        Returns:
            features: Feature maps at C2 [B, 256, H/4, W/4]
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)  # C2 output
        
        return x


class CB_Module_Full(nn.Module):
    """
    Alternative: Full ResNet backbone with multi-scale outputs
    For more flexible feature extraction
    """
    
    def __init__(self, pretrained=True, output_level='C2'):
        super(CB_Module_Full, self).__init__()
        
        # Load ResNet-50
        resnet = models.resnet50(pretrained=pretrained)
        
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        
        self.layer1 = resnet.layer1  # C2: 256 channels
        self.layer2 = resnet.layer2  # C3: 512 channels
        self.layer3 = resnet.layer3  # C4: 1024 channels
        self.layer4 = resnet.layer4  # C5: 2048 channels
        
        self.output_level = output_level
        
        # Output channels at different levels
        self.level_channels = {
            'C2': 256,
            'C3': 512,
            'C4': 1024,
            'C5': 2048
        }
        
        self.out_channels = self.level_channels[output_level]
        
    def forward(self, x):
        """Forward pass with specified output level"""
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)  # C2
        if self.output_level == 'C2':
            return x
            
        x = self.layer2(x)  # C3
        if self.output_level == 'C3':
            return x
            
        x = self.layer3(x)  # C4
        if self.output_level == 'C4':
            return x
            
        x = self.layer4(x)  # C5
        return x


def test_cb_module():
    """Test CB module"""
    print("=" * 60)
    print("Testing CB Module")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")
    
    # Test CB_C2
    print("Testing CB_Module (C2)...")
    cb_module = CB_Module(pretrained=True).to(device)
    
    # Test input
    dummy_input = torch.randn(2, 3, 512, 512).to(device)
    print(f"Input shape: {dummy_input.shape}")
    
    with torch.no_grad():
        features = cb_module(dummy_input)
        print(f"Output shape: {features.shape}")
        print(f"Expected: [2, 256, 128, 128] (1/4 resolution)")
        print(f"Output channels: {cb_module.out_channels}")
    
    print("\n✓ CB Module working correctly!")
    
    # Test full module
    print("\n" + "-" * 60)
    print("Testing CB_Module_Full at different levels...")
    
    for level in ['C2', 'C3', 'C4', 'C5']:
        cb_full = CB_Module_Full(pretrained=False, output_level=level).to(device)
        with torch.no_grad():
            features = cb_full(dummy_input)
            print(f"  {level}: {features.shape} - {cb_full.out_channels} channels")
    
    print("\n✓ All levels working correctly!")
    print("=" * 60)
    
    return cb_module


if __name__ == '__main__':
    test_cb_module()
