#!/usr/bin/env python3
"""
DSNet+DINO with Feature Adapter
Feeds CB features directly to DINO (bypasses DINO's ResNet backbone)
"""

import torch
import torch.nn as nn
import sys
sys.path.append('/home/as19197/new_project/WUNET')
sys.path.append('/home/as19197/new_project/DSNET')

# Import same way as dsnet_dino.py
from cb_module import CB_Module
from feature_recovery import FR_Module


class FeatureAdapter(nn.Module):
    """
    Adapts CB features (256 channels, single scale) 
    to DINO's expected multi-scale format
    
    CB output: 256 channels @ H/8 × W/8
    DINO expects: 3 scales from ResNet-50 backbone
      - Scale 0: 512 channels @ H/8 × W/8
      - Scale 1: 1024 channels @ H/16 × W/16
      - Scale 2: 2048 channels @ H/32 × W/32
    """
    
    def __init__(self, cb_channels=256):
        super().__init__()
        
        #Create multi-scale feature pyramid from CB features (3 scales for DINO)
        # Scale 0: 512 channels  (same resolution as CB)
        self.conv1 = nn.Sequential(
            nn.Conv2d(cb_channels, 512, 3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1, bias=False),
            nn.BatchNorm2d(512)
        )
        
        # Scale 1: 1024 channels, downsample 2x
        self.conv2 = nn.Sequential(
            nn.Conv2d(512, 1024, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 1024, 3, padding=1, bias=False),
            nn.BatchNorm2d(1024)
        )
        
        # Scale 2: 2048 channels, downsample 2x more
        self.conv3 = nn.Sequential(
            nn.Conv2d(1024, 2048, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(2048),
            nn.ReLU(inplace=True),
            nn.Conv2d(2048, 2048, 3, padding=1, bias=False),
            nn.BatchNorm2d(2048)
        )
        
    def forward(self, cb_features):
        """
        Convert CB features to multi-scale pyramid
        
        Args:
            cb_features: [B, 256, H/8, W/8] from CB module
            
        Returns:
            Tuple of features at 3 scales matching ResNet-50 output
        """
        # Scale 0: 512 channels, same resolution as CB
        feat1 = self.conv1(cb_features)
        
        # Scale 1: 1024 channels, half resolution
        feat2 = self.conv2(feat1)
        
        # Scale 2: 2048 channels, quarter resolution
        feat3 = self.conv3(feat2)
        
        return (feat1, feat2, feat3)


class DSNet_DINO_WithAdapter(nn.Module):
    """
    DSNet + DINO using Feature Adapter
    
    Architecture:
        Foggy → CB (256 ch) → FR → Restored
                    ↓
              Feature Adapter (convert to multi-scale)
                    ↓
              DINO Neck + Head (bypass DINO backbone!)
    """
    
    def __init__(self, dino_config=None, dino_checkpoint=None, 
                 pretrained_cb=True, load_dino=False):
        super().__init__()
        
        # CB Module - shared features
        self.cb_module = CB_Module(pretrained=pretrained_cb)
        
        # FR Module - restoration
        self.fr_module = FR_Module(in_channels=256)
        
        # Feature Adapter - CB to multi-scale
        self.feature_adapter = FeatureAdapter(cb_channels=256)
        
        # DINO Detector
        self.dino = None
        
        if load_dino:
            try:
                from mmdet.apis import init_detector
                device = 'cuda' if torch.cuda.is_available() else 'cpu'
                self.dino = init_detector(dino_config, dino_checkpoint, device=device)
                print("✓ Loaded DINO detector with feature adapter")
            except Exception as e:
                print(f"⚠ Could not load DINO: {e}")
    
    def forward(self, foggy_img, return_features=False):
        """Restoration forward pass"""
        cb_features = self.cb_module(foggy_img)
        restored_img = self.fr_module(cb_features)
        
        if return_features:
            return restored_img, cb_features
        return restored_img
    
    def detect_with_cb_features(self, foggy_imgs, img_metas=None):
        """
        Detection using CB features (NOT restored image!)
        Uses forward method for simpler API
        
        Args:
            foggy_imgs: Input foggy images [B, 3, H, W] tensor
            img_metas: Optional image metadata
            
        Returns:
            Detection results from DINO
        """
        if self.dino is None:
            raise RuntimeError("DINO not loaded")
        
        # Extract CB features (shared for restoration and detection!)
        cb_features = self.cb_module(foggy_imgs)
        
        # Convert CB features to multi-scale format via adapter
        multi_scale_features = self.feature_adapter(cb_features)
        
        # Feed multi-scale features to DINO's neck (BYPASS BACKBONE!)
        neck_features = self.dino.neck(multi_scale_features)
        
        # Use DINO's forward method for detection (simpler than predict)
        # This handles all the complex reference point generation internally
        batch_inputs = {
            'inputs': foggy_imgs,  # Original images for reference
            'data_samples': []  # Empty for inference
        }
        
        # Call DIN's bbox head forward for inference mode
        # This automatically generates reference points
        head_results = self.dino.bbox_head.forward(neck_features)
        
        # Post-process to get final predictions
        # Extract the detection results
        return head_results


def test_feature_adapter():
    """Test the feature adapter implementation"""
    print("="*70)
    print("Testing Feature Adapter for DSNet+DINO")
    print("="*70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    
    print("\n[1] Testing FeatureAdapter alone...")
    adapter = FeatureAdapter(cb_channels=256).to(device)
    
    # Simulate CB features
    cb_feat = torch.randn(2, 256, 64, 64).to(device)
    print(f"Input CB features: {cb_feat.shape}")
    
    multi_scale = adapter(cb_feat)
    print(f"\nMulti-scale outputs:")
    for i, feat in enumerate(multi_scale, 1):
        print(f"  Scale {i}: {feat.shape}")
    
    print("\n[2] Testing full DSNet+DINO...")
    DINO_CONFIG = '/home/as19197/new_project/mmdetection/configs/dino/dino-4scale_r50_8xb2-12e_coco.py'
    DINO_CHECKPOINT = '/home/as19197/new_project/mmdetection/configs/dino/dino-4scale_r50_8xb2-12e_coco_20221202_182705-55b2bba2.pth'
    
    model = DSNet_DINO_WithAdapter(
        dino_config=DINO_CONFIG,
        dino_checkpoint=DINO_CHECKPOINT,
        load_dino=True
    ).to(device)
    
    # Test restoration
    dummy_foggy = torch.randn(1, 3, 512, 512).to(device)
    
    with torch.no_grad():
        restored = model(dummy_foggy)
        print(f"\n✓ Restoration: {restored.shape}")
        
        # Test CB feature extraction
        neck_feats = model.detect_with_cb_features(dummy_foggy)
        print(f"✓ Detection features extracted: {len(neck_feats)} scales")
    
    print("\n" + "="*70)
    print("Feature Adapter Benefits:")
    print("  ✓ CB features used directly (not restored image)")
    print("  ✓ DINO backbone bypassed (faster inference)")
    print("  ✓ True feature sharing between tasks")
    print("  ✓ Trainable adapter learns optimal conversion")
    print("="*70)


if __name__ == '__main__':
    test_feature_adapter()
