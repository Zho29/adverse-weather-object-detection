#!/usr/bin/env python3
"""
Train DSNet+DINO with Feature Adapter
True feature sharing: CB features → Feature Adapter → DINO
"""

from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import sys
sys.path.append(str((Path(__file__).resolve().parent / '.').resolve()))

from dsnet_dino_adapter import DSNet_DINO_WithAdapter, FeatureAdapter
import json
from tqdm import tqdm
import numpy as np

# Import dataset (define collate_fn here)
from train_dsnet_dino_joint import CityscapesCocoDataset

def collate_fn(batch):
    """Collate batch with variable number of boxes per image"""
    foggy = torch.stack([item['foggy'] for item in batch])
    clear = torch.stack([item['clear'] for item in batch])
    boxes = [item['boxes'] for item in batch]
    labels = [item['labels'] for item in batch]
    image_ids = [item['image_id'] for item in batch]
    return {
        'foggy': foggy,
        'clear': clear,
        'boxes': boxes,
        'labels': labels,
        'image_ids': image_ids
    }

print("="*70)
print("DSNet+DINO Training with Feature Adapter")
print("="*70)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}\n")

# Configuration
EPOCHS = 5
BATCH_SIZE = 2
SAVE_DIR = '/home/as19197/new_project/checkpoints/dsnet_dino_adapter_5ep'
Path(SAVE_DIR).mkdir(exist_ok=True, parents=True)

# Paths
DINO_CONFIG = '/home/as19197/new_project/mmdetection/configs/dino/dino-4scale_r50_8xb2-12e_coco.py'
DINO_CHECKPOINT = '/home/as19197/new_project/mmdetection/configs/dino/dino-4scale_r50_8xb2-12e_coco_20221202_182705-55b2bba2.pth'

print("[1] Creating datasets...")
train_dataset = CityscapesCocoDataset(
    '/home/as19197/new_project/Foggy_Cityscapes_dataset/leftImg8bit_foggy',
    '/home/as19197/new_project/Foggy_Cityscapes_dataset/leftImg8bit',
    '/home/as19197/new_project/Foggy_Cityscapes_dataset/coco_annotations/cityscapes_train.json',
    beta=0.02, split='train'
)
val_dataset = CityscapesCocoDataset(
    '/home/as19197/new_project/Foggy_Cityscapes_dataset/leftImg8bit_foggy',
    '/home/as19197/new_project/Foggy_Cityscapes_dataset/leftImg8bit',
    '/home/as19197/new_project/Foggy_Cityscapes_dataset/coco_annotations/cityscapes_val.json',
    beta=0.02, split='val'
)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=8, pin_memory=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                         num_workers=8, pin_memory=True, collate_fn=collate_fn)

print("\n[2] Creating DSNet+DINO with Feature Adapter...")
model = DSNet_DINO_WithAdapter(
    dino_config=DINO_CONFIG,
    dino_checkpoint=DINO_CHECKPOINT,
    pretrained_cb=True,
    load_dino=True
)

if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs")
    model = nn.DataParallel(model)

model = model.to(device)

print("\n[3] Setting up training...")
# Loss weights
alpha = 1.0  # Restoration
beta = 0.2   # Detection-aware (CB feature matching)

# Losses
l1_loss = nn.L1Loss()
mse_loss = nn.MSELoss()

# Optimizer - train CB, FR, and Feature Adapter (DINO frozen for now)
trainable_params = []
for name, param in model.named_parameters():
    if 'dino' not in name or 'feature_adapter' in name:
        trainable_params.append(param)
        param.requires_grad = True
    else:
        param.requires_grad = False

optimizer = optim.AdamW(trainable_params, lr=1e-4, weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

print(f"Trainable parameters: {sum(p.numel() for p in trainable_params):,}")

# Training loop
print(f"\n[4] Training for {EPOCHS} epochs...")
best_val_loss = float('inf')
history = []

for epoch in range(1, EPOCHS + 1):
    print(f"\n{'='*70}")
    print(f"Epoch {epoch}/{EPOCHS}")
    print(f"{'='*70}")
    
    # Train
    model.train()
    total_loss = 0
    total_rest = 0
    total_det = 0
    
    for batch in tqdm(train_loader, desc="Training"):
        foggy = batch['foggy'].to(device)
        clear = batch['clear'].to(device)
        
        optimizer.zero_grad()
        
        # Forward
        restored, cb_features = model(foggy, return_features=True) if not isinstance(model, nn.DataParallel) else model.module(foggy, return_features=True)
        
        # Restoration loss
        rest_loss = l1_loss(restored, clear)
        
        # Detection-aware loss (CB features should match clear image features)
        if isinstance(model, nn.DataParallel):
            cb_clear = model.module.cb_module(clear)
        else:
            cb_clear = model.cb_module(clear)
        det_loss = mse_loss(cb_features, cb_clear)
        
        # Total loss
        loss = alpha * rest_loss + beta * det_loss
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        total_rest += rest_loss.item()
        total_det += det_loss.item()
    
    avg_loss = total_loss / len(train_loader)
    avg_rest = total_rest / len(train_loader)
    avg_det = total_det / len(train_loader)
    
    print(f"Train Loss: {avg_loss:.6f} (Rest: {avg_rest:.6f}, Det: {avg_det:.6f})")
    
    # Validation
    model.eval()
    val_loss = 0
    psnr_values = []
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validation"):
            foggy = batch['foggy'].to(device)
            clear = batch['clear'].to(device)
            
            restored = model(foggy) if not isinstance(model, nn.DataParallel) else model.module(foggy)
            
            rest_loss = l1_loss(restored, clear)
            val_loss += rest_loss.item()
            
            # PSNR
            mse = ((restored - clear) ** 2).mean()
            psnr = 10 * torch.log10(1.0 / (mse + 1e-8))
            psnr_values.append(psnr.item())
    
    avg_val_loss = val_loss / len(val_loader)
    avg_psnr = np.mean(psnr_values)
    
    print(f"Val Loss: {avg_val_loss:.6f}, PSNR: {avg_psnr:.2f} dB")
    
    # Save best
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        state_dict = model.state_dict() if not isinstance(model, nn.DataParallel) else model.module.state_dict()
        torch.save({
            'epoch': epoch,
            'model_state_dict': state_dict,
            'optimizer_state_dict': optimizer.state_dict(),
            'best_val_loss': best_val_loss
        }, Path(SAVE_DIR) / 'dsnet_dino_adapter_best.pth')
        print(f"  ✓ Saved best model!")
    
    scheduler.step()
    
    history.append({
        'epoch': epoch,
        'train_loss': avg_loss,
        'train_rest': avg_rest,
        'train_det': avg_det,
        'val_loss': avg_val_loss,
        'val_psnr': avg_psnr
    })
    
    with open(Path(SAVE_DIR) / 'history.json', 'w') as f:
        json.dump(history, f, indent=2)

print("\n" + "="*70)
print("Training Complete!")
print(f"Best Val Loss: {best_val_loss:.6f}")
print(f"Models saved to: {SAVE_DIR}")
print("="*70)
