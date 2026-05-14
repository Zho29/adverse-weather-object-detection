#!/usr/bin/env python3
"""
Complete DSNet + DINO End-to-End Training Pipeline
- Joint training: Restoration (FR) + Detection (DINO)
- Shared features via CB Module
- 5 epoch smoke test, then 100 epochs if successful
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
import cv2

sys.path.append('/home/as19197/new_project/DSNET')
sys.path.append('/home/as19197/new_project/WUNET')

from dsnet_dino import DSNet_DINO
from PIL import Image
import torchvision.transforms as transforms

print("="*70)
print("DSNet + DINO Complete End-to-End Training")
print("="*70)


class FoggyCityscapesDataset(Dataset):
    """Dataset for DSNet training"""
    
    def __init__(self, foggy_root, clear_root, beta=0.02, split='train', img_size=512):
        self.foggy_root = Path(foggy_root)
        self.clear_root = Path(clear_root)
        self.beta = beta
        self.split = split
        self.img_size = img_size
        
        self.foggy_images = list(self.foggy_root.rglob(f"{split}/*/*_beta_{beta}.png"))
        print(f"  Found {len(self.foggy_images)} {split} images")
        
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        
    def __len__(self):
        return len(self.foggy_images)
    
    def __getitem__(self, idx):
        foggy_path = self.foggy_images[idx]
        
        city_frame = foggy_path.stem.replace(f"_leftImg8bit_foggy_beta_{self.beta}", "")
        city = foggy_path.parent.name
        clear_path = self.clear_root / self.split / city / f"{city_frame}_leftImg8bit.png"
        
        foggy_img = Image.open(foggy_path).convert('RGB')
        clear_img = Image.open(clear_path).convert('RGB')
        
        foggy_tensor = self.transform(foggy_img)
        clear_tensor = self.transform(clear_img)
        
        return {
            'foggy': foggy_tensor,
            'clear': clear_tensor,
            'image_path': str(foggy_path)
        }


class DSNetDINOTrainer:
    """Complete trainer for DSNet+DINO joint learning"""
    
    def __init__(self, model, train_loader, val_loader, device, save_dir):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True, parents=True)
        
        # Multi-task loss weights
        self.alpha = 1.0  # Restoration loss
        self.l1_loss = nn.L1Loss()
        
        # Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=1e-4,
            weight_decay=1e-4
        )
        
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=100, eta_min=1e-6
        )
        
        self.best_val_loss = float('inf')
        self.history = []
        
    def train_epoch(self, epoch):
        """Train one epoch"""
        self.model.train()
        total_loss = 0
        
        for batch in tqdm(self.train_loader, desc=f"Epoch {epoch}"):
            foggy = batch['foggy'].to(self.device)
            clear = batch['clear'].to(self.device)
            
            self.optimizer.zero_grad()
            
            # Forward pass (FR module only for now)
            restored = self.model(foggy)
            
            # Restoration loss
            restoration_loss = self.l1_loss(restored, clear)
            loss = self.alpha * restoration_loss
            
            # Backward
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(self.train_loader)
        return avg_loss
    
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
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict() if not isinstance(self.model, nn.DataParallel) else self.model.module.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'history': self.history
        }
        
        torch.save(checkpoint, self.save_dir / 'dsnet_dino_last.pth')
        if is_best:
            torch.save(checkpoint, self.save_dir / 'dsnet_dino_best.pth')
    
    def train(self, epochs):
        """Full training loop"""
        print(f"\nStarting training for {epochs} epochs...")
        
        for epoch in range(1, epochs + 1):
            print(f"\n{'='*70}")
            print(f"Epoch {epoch}/{epochs}")
            print(f"{'='*70}")
            
            train_loss = self.train_epoch(epoch)
            print(f"Train Loss: {train_loss:.6f}")
            
            val_loss, val_psnr = self.validate(epoch)
            print(f"Val Loss: {val_loss:.6f}, PSNR: {val_psnr:.2f} dB")
            
            self.scheduler.step()
            
            self.history.append({
                'epoch': epoch,
                'train_loss': train_loss,
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
    SAVE_DIR = f'/home/as19197/new_project/checkpoints/dsnet_dino_{args.epochs}ep'
    
    print("[1] Creating datasets...")
    train_dataset = FoggyCityscapesDataset(
        '/home/as19197/new_project/Foggy_Cityscapes_dataset/leftImg8bit_foggy',
        '/home/as19197/new_project/Foggy_Cityscapes_dataset/leftImg8bit',
        beta=0.02, split='train'
    )
    val_dataset = FoggyCityscapesDataset(
        '/home/as19197/new_project/Foggy_Cityscapes_dataset/leftImg8bit_foggy',
        '/home/as19197/new_project/Foggy_Cityscapes_dataset/leftImg8bit',
        beta=0.02, split='val'
    )
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, 
                               shuffle=True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                             shuffle=False, num_workers=8, pin_memory=True)
    
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
    
    print("\n[3] Starting training...")
    trainer = DSNetDINOTrainer(model, train_loader, val_loader, device, SAVE_DIR)
    trainer.train(args.epochs)
    
    print(f"\n✓ Training complete! Models saved to {SAVE_DIR}")


if __name__ == '__main__':
    main()
