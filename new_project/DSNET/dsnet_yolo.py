#!/usr/bin/env python3
"""
DSNet + YOLO Architecture
YOLO detection head learns directly from CB features (not restored images!)
True feature sharing for joint restoration and detection
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
sys.path.append('/home/as19197/new_project/WUNET')

from cb_module import CB_Module
from feature_recovery import FR_Module


class YOLOFeatureAdapter(nn.Module):
    """
    Adapts CB features to YOLO's expected multi-scale format
    
    YOLOv8 expects:
    - P3: 128 channels @ H/8 × W/8
    - P4: 256 channels @ H/16 × W/16
    - P5: 512 channels @ H/32 × W/32
    """
    
    def __init__(self, cb_channels=256):
        super().__init__()
        
        # P3: Upsample to 128 channels (same resolution as CB)
        self.p3_conv = nn.Sequential(
            nn.Conv2d(cb_channels, 128, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.SiLU(inplace=True)
        )
        
        # P4: Keep at 256 channels (same as CB)
        self.p4_conv = nn.Sequential(
            nn.Conv2d(cb_channels, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.SiLU(inplace=True)
        )
        
        # P5: Downsample to 512 channels
        self.p5_conv = nn.Sequential(
            nn.Conv2d(cb_channels, 512, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.SiLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.SiLU(inplace=True)
        )
    
    def forward(self, cb_features):
        """
        Convert CB features to YOLO multi-scale format
        
        Args:
            cb_features: [B, 256, H/8, W/8]
            
        Returns:
            List of [P3, P4, P5] features
        """
        p3 = self.p3_conv(cb_features)  # [B, 128, H/8, W/8]
        p4 = self.p4_conv(cb_features)  # [B, 256, H/8, W/8]
        p5 = self.p5_conv(cb_features)  # [B, 512, H/16, W/16]
        
        return [p3, p4, p5]


class DSNet_YOLO(nn.Module):
    """
    DSNet with YOLO detection - True feature sharing!
    
    Architecture:
        Foggy Image
            ↓
        [CB Module] - Shared features
            ├──────────┬──────────┐
            ↓          ↓          ↓
        [FR Module]  [Adapter]  (CB)
            ↓          ↓
        Restored   [YOLO Head]
        Image       ↓
                 Detections
    
    Key: YOLO uses CB features directly via adapter!
    """
    
    def __init__(self, num_classes=2, pretrained_cb=True):
        super().__init__()
        
        # CB Module - shared feature extraction
        self.cb_module = CB_Module(pretrained=pretrained_cb)
        
        # FR Module - restoration branch
        self.fr_module = FR_Module(in_channels=256)
        
        # Feature adapter - CB to YOLO format
        self.feature_adapter = YOLOFeatureAdapter(cb_channels=256)
        
        # YOLO detection head
        # Using simple detection head compatible with our features
        self.num_classes = num_classes
        self.no = num_classes + 5  # outputs per anchor (x,y,w,h,conf,cls...)
        
        # Detection heads for each scale (P3, P4, P5)
        self.detect_p3 = nn.Conv2d(128, self.no * 3, 1)  # 3 anchors
        self.detect_p4 = nn.Conv2d(256, self.no * 3, 1)
        self.detect_p5 = nn.Conv2d(512, self.no * 3, 1)
    
    def forward(self, foggy_img, return_features=False):
        """
        Forward pass for both restoration and detection
        
        Args:
            foggy_img: Input foggy images [B, 3, H, W]
            return_features: If True, return CB features too
            
        Returns:
            restored_img: Restored image
            detections: Detection outputs [P3, P4, P5] or None if inference
        """
        # Extract CB features (shared for both tasks!)
        cb_features = self.cb_module(foggy_img)
        
        # Branch 1: Restoration
        restored_img = self.fr_module(cb_features)
        
        # Branch 2: Detection (uses CB features via adapter)
        yolo_features = self.feature_adapter(cb_features)
        
        # YOLO detection heads
        detections = [
            self.detect_p3(yolo_features[0]),
            self.detect_p4(yolo_features[1]),
            self.detect_p5(yolo_features[2])
        ]
        
        if return_features:
            return restored_img, detections, cb_features
        return restored_img, detections
    
    def predict(self, foggy_img):
        """Inference mode - returns processed predictions"""
        with torch.no_grad():
            restored, detections = self.forward(foggy_img)
            # Post-process detections (NMS etc)
            # For now, return raw outputs
            return restored, detections


def test_dsnet_yolo():
    """Test DSNet+YOLO architecture"""
    print("="*70)
    print("Testing DSNet+YOLO Architecture")
    print("="*70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    
    print("\n[1] Creating DSNet+YOLO model...")
    model = DSNet_YOLO(num_classes=2, pretrained_cb=True)
    model = model.to(device)
    print(f"✓ Model created")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    
    print("\n[2] Testing forward pass...")
    dummy_input = torch.randn(2, 3, 512, 512).to(device)
    
    with torch.no_grad():
        restored, detections = model(dummy_input)
    
    print(f"✓ Forward pass successful!")
    print(f"  Restored shape: {restored.shape}")
    print(f"  Detection outputs:")
    for i, det in enumerate(detections):
        print(f"    Scale {i}: {det.shape}")
    
    print("\n[3] Testing with return_features...")
    with torch.no_grad():
        restored, detections, cb_feat = model(dummy_input, return_features=True)
    print(f"✓ CB features: {cb_feat.shape}")
    
    print("\n" + "="*70)
    print("Architecture Benefits:")
    print("  ✓ CB features shared between restoration and detection")
    print("  ✓ YOLO learns directly from CB (not restored images)")
    print("  ✓ Single forward pass for both tasks")
    print("  ✓ End-to-end differentiable")
    print("="*70)
    
    return model


if __name__ == '__main__':
    model = test_dsnet_yolo()
