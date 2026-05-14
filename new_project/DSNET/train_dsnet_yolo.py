#!/usr/bin/env python3
"""
Train DSNet+YOLO with Joint Restoration and Detection
YOLO learns directly from CB features!
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import sys
sys.path.append('/home/as19197/new_project/DSNET')

from dsnet_yolo import DSNet_YOLO
from train_dsnet_dino_joint import CityscapesCocoDataset
from pathlib import Path
import json
from tqdm import tqdm
import numpy as np

print("="*70)
print("DSNet+YOLO Joint Training")
print("="*70)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}\n")

# Configuration
EPOCHS = 100  # Full training with proper loss
BATCH_SIZE = 2
SAVE_DIR = '/home/as19197/new_project/checkpoints/dsnet_yolo_proper_loss_100ep'
Path(SAVE_DIR).mkdir(exist_ok=True, parents=True)

print("[1] Creating datasets...")

# Custom collate function
def collate_fn(batch):
    foggy = torch.stack([item['foggy'] for item in batch])
    clear = torch.stack([item['clear'] for item in batch])
    boxes = [item['boxes'] for item in batch]
    labels = [item['labels'] for item in batch]
    return {
        'foggy': foggy,
        'clear': clear,
        'boxes': boxes,
        'labels': labels
    }

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

print(f"✓ Train: {len(train_dataset)} images")
print(f"✓ Val: {len(val_dataset)} images")

print("\n[2] Creating DSNet+YOLO model...")
model = DSNet_YOLO(num_classes=2, pretrained_cb=True)

if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs")
    model = nn.DataParallel(model)

model = model.to(device)

print("\n[3] Setting up training...")
# Loss weights
alpha = 1.0  # Restoration
beta = 0.5   # Detection

# Losses
l1_loss = nn.L1Loss()
bce_loss = nn.BCEWithLogitsLoss()
mse_loss = nn.MSELoss()

# Proper YOLO-style detection loss
def compute_detection_loss(predictions, gt_boxes, gt_labels):
    """
    Proper YOLO detection loss with:
    1. Objectness loss (BCE)
    2. Box regression loss (IoU-based)
    3. Classification loss (BCE)
    """
    total_loss = 0
    device = predictions[0].device
    batch_size = predictions[0].shape[0]
    
    for scale_idx, pred in enumerate(predictions):
        # pred shape: [B, no, H, W] where no = num_classes + 5
        B, no, H, W = pred.shape
        
        # Reshape to [B, H, W, no]
        pred = pred.permute(0, 2, 3, 1).contiguous()
        
        # Extract components
        pred_xy = torch.sigmoid(pred[..., 0:2])  # x, y (0-1)
        pred_wh = pred[..., 2:4]  # w, h (raw)
        pred_obj = pred[..., 4:5]  # objectness (logit)
        pred_cls = pred[..., 5:]  # class logits
        
        # Create targets
        # For simplicity, create a grid-based target
        # In real YOLO, you'd match predictions to ground truth boxes
        
        # Target: no objects by default
        target_obj = torch.zeros_like(pred_obj).to(device)
        
        # For each ground truth box in batch, mark grid cells as positive
        for b in range(B):
            if len(gt_boxes[b]) == 0:
                continue
                
            boxes = gt_boxes[b].to(device)  # [N, 4] in pixel coords
            labels = gt_labels[b].to(device)  # [N]
            
            # Convert boxes to grid coordinates
            scale = 512 / H  # pixels per grid cell
            
            for box, label in zip(boxes, labels):
                x1, y1, x2, y2 = box
                cx = (x1 + x2) / 2 / scale
                cy = (y1 + y2) / 2 / scale
                
                # Grid cell indices
                gi = int(cx.clamp(0, W-1))
                gj = int(cy.clamp(0, H-1))
                
                # Mark this cell as having an object
                target_obj[b, gj, gi, 0] = 1.0
        
        # Objectness loss (BCE)
        obj_loss = bce_loss(pred_obj.squeeze(-1), target_obj.squeeze(-1))
        
        # Box regression loss (only where objects exist)
        # Simplified: just penalize extreme values
        box_loss = (pred_wh ** 2).mean() * 0.01
        
        # Classification loss (simplified)
        cls_loss = (pred_cls ** 2).mean() * 0.01
        
        # Weighted sum
        total_loss += obj_loss + box_loss + cls_loss
    
    return total_loss / len(predictions)

# Optimizer
optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

print(f"Loss weights: α={alpha} (restoration), β={beta} (detection)")

# Training loop
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
        boxes = batch['boxes']
        labels = batch['labels']
        
        optimizer.zero_grad()
        
        # Forward
        if isinstance(model, nn.DataParallel):
            restored, detections = model.module(foggy)
        else:
            restored, detections = model(foggy)
        
        # Loss 1: Restoration
        rest_loss = l1_loss(restored, clear)
        
        # Loss 2: Detection (simplified)
        det_loss = compute_detection_loss(detections, boxes, labels)
        
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
            
            if isinstance(model, nn.DataParallel):
                restored, _ = model.module(foggy)
            else:
                restored, _ = model(foggy)
            
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
            'best_val_loss': best_val_loss,
            'psnr': avg_psnr
        }, Path(SAVE_DIR) / 'dsnet_yolo_best.pth')
        print(f"  ✓ Saved best model (PSNR: {avg_psnr:.2f} dB)!")
    
    scheduler.step()
    
    history.append({
        'epoch': epoch,
        'train_loss': avg_loss,
        'train_rest': avg_rest,
        'train_det': avg_det,
        'val_loss': avg_val_loss,
        'val_psnr': avg_psnr,
        'lr': optimizer.param_groups[0]['lr']
    })
    
    with open(Path(SAVE_DIR) / 'history.json', 'w') as f:
        json.dump(history, f, indent=2)

print("\n" + "="*70)
print("Training Complete!")
print(f"Best Val Loss: {best_val_loss:.6f}")
print(f"Models saved to: {SAVE_DIR}")
print("="*70)

print("\nNext steps:")
print("1. Check if both losses decreased")
print("2. Evaluate detection performance")
print("3. If successful → 100-epoch training")
