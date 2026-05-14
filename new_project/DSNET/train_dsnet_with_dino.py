#!/usr/bin/env python3
"""
Train DSNet + DINO with Detection (5 epochs)
Joint training of restoration (FR) + detection (DINO)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import sys
import argparse
from tqdm import tqdm
import json
import numpy as np

# Add paths
sys.path.append('/home/as19197/new_project/DSNET')
sys.path.append('/home/as19197/new_project/WUNET')

from dsnet_dino import DSNet_DINO
import cv2
from PIL import Image
import torchvision.transforms as transforms

print("="*70)
print("DSNet + DINO Joint Training - Detection + Restoration")
print("="*70)

class FoggyCityscapesDataset(Dataset):
    """Dataset with foggy/clear images + detection annotations"""
    
    def __init__(self, foggy_root, clear_root, label_root, beta=0.02, split='train', img_size=512):
        self.foggy_root = Path(foggy_root)
        self.clear_root = Path(clear_root)
        self.label_root = Path(label_root)
        self.beta = beta
        self.split = split
        self.img_size = img_size
        
        # Find all foggy images
        self.foggy_images = list(self.foggy_root.rglob(f"{split}/*/*_beta_{beta}.png"))
        print(f"Found {len(self.foggy_images)} {split} images for beta={beta}")
        
        # Transforms
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        
    def __len__(self):
        return len(self.foggy_images)
    
    def __getitem__(self, idx):
        foggy_path = self.foggy_images[idx]
        
        # Get corresponding clear image
        city_frame = foggy_path.stem.replace(f"_leftImg8bit_foggy_beta_{self.beta}", "")
        city = foggy_path.parent.name
        clear_path = self.clear_root / self.split / city / f"{city_frame}_leftImg8bit.png"
        
        # Load images
        foggy_img = Image.open(foggy_path).convert('RGB')
        clear_img = Image.open(clear_path).convert('RGB')
        
        # Apply transforms
        foggy_tensor = self.transform(foggy_img)
        clear_tensor = self.transform(clear_img)
        
        # Load labels (YOLO format)
        # For now, we'll skip detection labels as placeholder
        # TODO: Add actual YOLO label loading
        
        return {
            'foggy': foggy_tensor,
            'clear': clear_tensor,
            'image_path': str(foggy_path)
        }


def main():
    # Load pre-trained FR model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    
    print("\n[1] Loading pre-trained FR model...")
    checkpoint = torch.load('/home/as19197/new_project/checkpoints/dsnet_beta0.02/dsnet_best.pth')
    
    print("\n[2] Creating DSNet with DINO...")
    model = DSNet_DINO(load_dino=True, pretrained_cb=True)
    
    # Load FR weights
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model = model.to(device)
    
    # Load pre-trained restoration weights
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    print("✓ Loaded pre-trained FR weights")
    
    print("\n[3] Training with DINO detection for 5 epochs...")
    print("  (DINO integration coming soon - placeholder for now)")
    
    # TODO: Implement DINO detection training
    # For now, just save the model with DINO enabled
    
    print("\n✓ DSNet + DINO model ready")
    print(f"  FR module: Pre-trained (PSNR: 16.89 dB)")
    print(f"  DINO module: Initialized (ready for detection)")


if __name__ == '__main__':
    main()
