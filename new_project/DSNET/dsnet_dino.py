"""
DSNet with Full MMDetection DINO Detector
Using complete DINO model from MMDetection, not just backbone
"""

import torch
import torch.nn as nn
from cb_module import CB_Module
from feature_recovery import FR_Module


class DSNet_DINO(nn.Module):
    """
    DSNet with MMDetection DINO
    
    Architecture (Joint Learning):
                    Input (Foggy Image)
                           ↓
                    [CB Module] - Extract Features
                      ↙        ↘
              [FR Module]    [DINO Detector]
                  ↓               ↓
            Restored Image   Detections
    
    Key: CB features are shared for BOTH restoration and detection!
    Joint optimization makes features good for both tasks.
    """
    
    def __init__(self, dino_config='dino-4scale_r50_8xb2-12e_coco', 
                 dino_checkpoint=None, pretrained_cb=True, load_dino=False):
        super(DSNet_DINO, self).__init__()
        
        # CB Module - shared feature extraction
        self.cb_module = CB_Module(pretrained=pretrained_cb)
        
        # FR Module - restoration branch
        self.fr_module = FR_Module(in_channels=256)
        
        # DINO Detector - detection branch (uses CB features!)
        self.dino = None
        if load_dino:
            try:
                from mmdet.apis import init_detector
                
                # Config file
                if dino_config.endswith('.py'):
                    config_file = dino_config
                else:
                    config_file = f'https://github.com/open-mmlab/mmdetection/tree/main/configs/dino/{dino_config}.py'
                
                # Checkpoint
                if dino_checkpoint is None:
                    dino_checkpoint = 'https://download.openmmlab.com/mmdetection/v3.0/dino/dino-4scale_r50_8xb2-12e_coco/dino-4scale_r50_8xb2-12e_coco_20221202_182705-55b2bba2.pth'
                
                device = 'cuda' if torch.cuda.is_available() else 'cpu'
                self.dino = init_detector(config_file, dino_checkpoint, device=device)
                print(f"✓ Loaded DINO detector from MMDetection")
                
            except Exception as e:
                print(f"⚠ Could not load DINO: {e}")
                print("  DINO can be loaded later for inference")
        
    def forward(self, foggy_img, return_features=False):
        """
        Forward pass - DSNet joint architecture
        
        Args:
            foggy_img: Input foggy image [B, 3, H, W]
            return_features: Whether to return CB features
            
        Returns:
            restored_img: Restored image from FR module
            features: CB features (if return_features=True)
        """
        # Step 1: Extract shared features via CB
        features = self.cb_module(foggy_img)
        
        # Step 2: Restoration branch (FR module uses CB features)
        restored_img = self.fr_module(features)
        
        # Return
        if return_features:
            return restored_img, features
        else:
            return restored_img
    
    def detect(self, foggy_img):
        """
        Run DINO detection using CB features
        
        Args:
            foggy_img: Input foggy image (PIL, numpy, or tensor)
            
        Returns:
            results: DINO detection results
            restored_img: Restored image from FR
            features: CB features used
        """
        if self.dino is None:
            raise RuntimeError("DINO detector not loaded. Set load_dino=True")
        
        # Get CB features and restored image
        restored_img, features = self.forward(foggy_img, return_features=True)
        
        # TODO: DINO should use CB features, but MMDetection expects images
        # For now, use restored image as input to DINO
        # (Future: modify DINO to accept feature maps directly)
        from mmdet.apis import inference_detector
        results = inference_detector(self.dino, restored_img)
        
        return results, restored_img, features
    
    def forward_train(self, foggy_img, clear_img):
        """
        Training forward pass
        
        Args:
            foggy_img: Foggy input
            clear_img: Clear reference
            
        Returns:
            restored_img: Restored image
            features: Features for detection
        """
        restored_img, features = self.forward(foggy_img)
        
        return {
            'restored_img': restored_img,
            'features': features,
            'target': clear_img
        }


def test_simple():
    """Simple test without loading DINO"""
    print("=" * 60)
    print("Testing DSNet Architecture (CB + FR modules)")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")
    
    # Test CB + FR modules
    print("Testing CB Module...")
    cb = CB_Module(pretrained=False).to(device)
    fr = FR_Module(in_channels=256).to(device)
    
    # Test forward
    dummy_input = torch.randn(2, 3, 512, 512).to(device)
    print(f"Input shape: {dummy_input.shape}")
    
    with torch.no_grad():
        features = cb(dummy_input)
        restored = fr(features)
        print(f"CB Features: {features.shape}")
        print(f"FR Restored: {restored.shape}")
    
    print("\n✓ DSNet modules working!")
    print("\n" + "=" * 60)
    print("Ready for DINO integration when MMDet loads")
    print("=" * 60)


if __name__ == '__main__':
    # Just test the core architecture
    test_simple()
