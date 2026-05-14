#!/usr/bin/env python3
"""
DSNet + DINO Joint Training with Detection Loss
- Restoration (FR Module): L1 loss
- Detection (DINO): Detection loss
- Shared features (CB Module)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import sys
from tqdm import tqdm
import json
import numpy as np
from PIL import Image
import torchvision.transforms as transforms

sys.path.append('/home/as19197/new_project/DSNET')

from dsnet_dino import DSNet_DINO

print("="*70)
print("DSNet + DINO Joint Training (Restoration + Detection)")
print("="*70)


class CityscapesCocoDataset(Dataset):
    """Dataset with foggy images, clear references, and COCO detection annotations"""
    
    def __init__(self, foggy_root, clear_root, coco_annotation_file, 
                 beta=0.02, split='train', img_size=512):
        self.foggy_root = Path(foggy_root)
        self.clear_root = Path(clear_root)
        self.beta = beta
        self.split = split
        self.img_size = img_size
        
        # Load COCO annotations
        with open(coco_annotation_file, 'r') as f:
            self.coco = json.load(f)
        
        # Create image index
        self.images = self.coco['images']
        self.annotations = self.coco['annotations']
        
        # Group annotations by image_id
        self.img_to_anns = {}
        for ann in self.annotations:
            img_id = ann['image_id']
            if img_id not in self.img_to_anns:
                self.img_to_anns[img_id] = []
            self.img_to_anns[img_id].append(ann)
        
        print(f"  Loaded {len(self.images)} {split} images with {len(self.annotations)} annotations")
        
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_info = self.images[idx]
        img_id = img_info['id']
        city = img_info['city']
        
        # Construct paths
        base_name = img_info['file_name'].replace('_leftImg8bit.png', '')
        foggy_path = self.foggy_root / self.split / city / f"{base_name}_leftImg8bit_foggy_beta_{self.beta}.png"
        clear_path = self.clear_root / self.split / city / f"{base_name}_leftImg8bit.png"
        
        # Load images
        foggy_img = Image.open(foggy_path).convert('RGB')
        clear_img = Image.open(clear_path).convert('RGB')
        
        # Get original size
        orig_w, orig_h = foggy_img.size
        
        # Transform
        foggy_tensor = self.transform(foggy_img)
        clear_tensor = self.transform(clear_img)
        
        # Get annotations for this image
        anns = self.img_to_anns.get(img_id, [])
        
        # Convert to format for detection
        boxes = []
        labels = []
        for ann in anns:
            bbox = ann['bbox']  # [x, y, w, h]
            # Scale bbox to resized image
            x, y, w, h = bbox
            x_scale = self.img_size / orig_w
            y_scale = self.img_size / orig_h
            
            scaled_bbox = [
                x * x_scale, 
                y * y_scale,
                (x + w) * x_scale,
                (y + h) * y_scale
            ]  # Convert to [x1, y1, x2, y2]
            
            boxes.append(scaled_bbox)
            labels.append(ann['category_id'])
        
        return {
            'foggy': foggy_tensor,
            'clear': clear_tensor,
            'boxes': torch.tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 4)),
            'labels': torch.tensor(labels, dtype=torch.int64) if labels else torch.zeros((0,), dtype=torch.int64),
            'image_id': img_id
        }


class DSNetDINOJointTrainer:
    """Joint trainer for restoration + detection"""
    
    def __init__(self, model, train_loader, val_loader, device, save_dir):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True, parents=True)
        
        # Loss weights
        self.alpha = 1.0  # Restoration
        self.beta = 0.2   # Detection-aware perceptual loss
        
        # Restoration loss
        self.l1_loss = nn.L1Loss()
        self.mse_loss = nn.MSELoss()
        
        # Optimizer (only FR and CB modules - DINO frozen)
        trainable_params = []
        for name, param in self.model.named_parameters():
            if 'dino' not in name:  # Don't train DINO
                trainable_params.append(param)
            else:
                param.requires_grad = False
        
        self.optimizer = optim.AdamW(trainable_params, lr=1e-4, weight_decay=1e-4)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=100)
        
        self.best_val_loss = float('inf')
        self.history = []
        
    def compute_detection_aware_loss(self, foggy, clear, boxes_list):
        """Detection-aware perceptual loss using CB features"""
        # Extract features from CB module
        if isinstance(self.model, nn.DataParallel):
            cb_foggy_feat = self.model.module.cb_module(foggy)
            cb_clear_feat = self.model.module.cb_module(clear)
        else:
            cb_foggy_feat = self.model.cb_module(foggy)
            cb_clear_feat = self.model.cb_module(clear)
        
        # Feature perceptual loss (features should match for detection)
        det_loss = self.mse_loss(cb_foggy_feat, cb_clear_feat)
        return det_loss
    
    def train_epoch(self, epoch):
        """Train one epoch"""
        self.model.train()
        total_rest_loss = 0
        total_det_loss = 0
        total_loss = 0
        
        for batch in tqdm(self.train_loader, desc=f"Epoch {epoch}"):
            foggy = batch['foggy'].to(self.device)
            clear = batch['clear'].to(self.device)
            boxes = batch['boxes']
            
            self.optimizer.zero_grad()
            
            # Forward pass
            restored = self.model(foggy)
            
            # Restoration loss
            restoration_loss = self.l1_loss(restored, clear)
            
            # Detection-aware perceptual loss
            detection_loss = self.compute_detection_aware_loss(foggy, clear, boxes)
            
            # Combined loss
            loss = self.alpha * restoration_loss + self.beta * detection_loss
            
            loss.backward()
            self.optimizer.step()
            
            total_rest_loss += restoration_loss.item()
            total_det_loss += detection_loss.item()
            total_loss += loss.item()
        
        avg_loss = total_loss / len(self.train_loader)
        avg_rest_loss = total_rest_loss / len(self.train_loader)
        avg_det_loss = total_det_loss / len(self.train_loader)
        return avg_loss, avg_rest_loss, avg_det_loss
    
    def validate(self, epoch):
        """Validation"""
        self.model.eval()
        total_loss = 0
        psnr_values = []
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation"):
                foggy = batch['foggy'].to(self.device)
                clear = batch['clear'].to(self.device)
                
                restored = self.model(foggy)
                
                restoration_loss = self.l1_loss(restored, clear)
                loss = self.alpha * restoration_loss
                total_loss += loss.item()
                
                # PSNR
                mse = ((restored - clear) ** 2).mean()
                psnr = 10 * torch.log10(1.0 / (mse + 1e-8))
                psnr_values.append(psnr.item())
        
        avg_loss = total_loss / len(self.val_loader)
        avg_psnr = np.mean(psnr_values)
        
        if avg_loss < self.best_val_loss:
            self.best_val_loss = avg_loss
            self.save_checkpoint(epoch, is_best=True)
            print(f"  ✓ Best model! Val Loss: {avg_loss:.6f}, PSNR: {avg_psnr:.2f} dB")
        
        return avg_loss, avg_psnr
    
    def save_checkpoint(self, epoch, is_best=False):
        """Save checkpoint"""
        state_dict = self.model.state_dict() if not isinstance(self.model, nn.DataParallel) else self.model.module.state_dict()
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': state_dict,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'history': self.history
        }
        
        torch.save(checkpoint, self.save_dir / 'dsnet_dino_joint_last.pth')
        if is_best:
            torch.save(checkpoint, self.save_dir / 'dsnet_dino_joint_best.pth')
    
    def train(self, epochs):
        """Full training loop"""
        print(f"\nStarting joint training for {epochs} epochs...")
        print(f"Loss weights: α={self.alpha} (restoration), β={self.beta} (detection)")
        
        for epoch in range(1, epochs + 1):
            print(f"\n{'='*70}")
            print(f"Epoch {epoch}/{epochs}")
            print(f"{'='*70}")
            
            train_loss, train_rest, train_det = self.train_epoch(epoch)
            print(f"Train Loss: {train_loss:.6f} (Rest: {train_rest:.6f}, Det: {train_det:.6f})")
            
            val_loss, val_psnr = self.validate(epoch)
            print(f"Val Loss: {val_loss:.6f}, PSNR: {val_psnr:.2f} dB")
            
            self.scheduler.step()
            
            self.history.append({
                'epoch': epoch,
                'train_loss': train_loss,
                'train_restoration_loss': train_rest,
                'train_detection_loss': train_det,
                'val_loss': val_loss,
                'val_psnr': val_psnr,
                'lr': self.optimizer.param_groups[0]['lr']
            })
            
            with open(self.save_dir / 'training_history.json', 'w') as f:
                json.dump(self.history, f, indent=2)
        
        print("\n" + "="*70)
        print("Training Complete!")
        print(f"Best Val Loss: {self.best_val_loss:.6f}")
        print("="*70)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch-size', type=int, default=4)
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")
    
    # Paths
    DINO_CONFIG = '/home/as19197/new_project/mmdetection/configs/dino/dino-4scale_r50_8xb2-12e_coco.py'
    DINO_CHECKPOINT = '/home/as19197/new_project/mmdetection/configs/dino/dino-4scale_r50_8xb2-12e_coco_20221202_182705-55b2bba2.pth'
    SAVE_DIR = f'/home/as19197/new_project/checkpoints/dsnet_dino_joint_{args.epochs}ep'
    
    print("[1] Creating datasets with COCO annotations...")
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
    
    # Custom collate function for variable-size boxes
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
    
    # Filter to only beta=0.02 images (2975 train, 500 val)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                               shuffle=True, num_workers=8, pin_memory=True,
                               collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                             shuffle=False, num_workers=8, pin_memory=True,
                             collate_fn=collate_fn)
    
    print("\n[2] Creating DSNet + DINO model...")
    model = DSNet_DINO(
        dino_config=DINO_CONFIG,
        dino_checkpoint=DINO_CHECKPOINT,
        pretrained_cb=True,
        load_dino=True
    )
    
    if torch.cuda.device_count() > 1:
        print(f"  Using {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)
    
    model = model.to(device)
    print("  ✓ Model created")
    
    print("\n[3] Starting joint training...")
    trainer = DSNetDINOJointTrainer(model, train_loader, val_loader, device, SAVE_DIR)
    trainer.train(args.epochs)
    
    print(f"\n✓ Training complete! Models saved to {SAVE_DIR}")


if __name__ == '__main__':
    main()
