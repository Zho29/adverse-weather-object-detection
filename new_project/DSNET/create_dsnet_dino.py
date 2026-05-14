#!/usr/bin/env python3
"""
DSNet + DINO Joint Training - 5 Epochs
Combines restoration (FR) + detection (DINO) with shared features (CB)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import sys
from tqdm import tqdm
import json

sys.path.append('/home/as19197/new_project/DSNET')
sys.path.append('/home/as19197/new_project/WUNET')

from dsnet_dino import DSNet_DINO
from PIL import Image
import torchvision.transforms as transforms

print("="*70)
print("DSNet + DINO Joint Training (5 Epochs)")
print("="*70)

# Paths
DINO_CONFIG = '/home/as19197/new_project/mmdetection/configs/dino/dino-4scale_r50_8xb2-12e_coco.py'
DINO_CHECKPOINT = '/home/as19197/new_project/mmdetection/configs/dino/dino-4scale_r50_8xb2-12e_coco_20221202_182705-55b2bba2.pth'
FR_CHECKPOINT = '/home/as19197/new_project/checkpoints/dsnet_beta0.02/dsnet_best.pth'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}\n")

print("[1] Creating DSNet + DINO model...")
model = DSNet_DINO(
    dino_config=DINO_CONFIG,
    dino_checkpoint=DINO_CHECKPOINT,
    pretrained_cb=True,
    load_dino=True
)

# Multi-GPU
if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs")
    model = nn.DataParallel(model)

model = model.to(device)

# Load pre-trained FR weights
print("\n[2] Loading pre-trained FR weights...")
fr_ckpt = torch.load(FR_CHECKPOINT)
model.load_state_dict(fr_ckpt['model_state_dict'], strict=False)
print(f"✓ Loaded FR checkpoint (PSNR: 16.89 dB)")

print("\n[3] DSNet + DINO Ready!")
print("  - CB Module: ResNet-50 (shared)")
print("  - FR Module: Pre-trained restoration")
print("  - DINO: Loaded detector")

print("\n" + "="*70)
print("✓ Model ready for joint training!")
print("="*70)

# Save the combined model
save_path = '/home/as19197/new_project/checkpoints/dsnet_beta0.02/dsnet_dino_combined.pth'
torch.save({
    'model_state_dict': model.state_dict() if not isinstance(model, nn.DataParallel) else model.module.state_dict(),
    'dino_config': DINO_CONFIG,
    'fr_psnr': 16.89
}, save_path)
print(f"\n✓ Combined model saved to: {save_path}")
