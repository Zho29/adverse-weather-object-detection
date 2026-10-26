#!/usr/bin/env python3
"""
Train DSNet + DINO on Foggy Cityscapes (High Fog - Beta=0.02)

Joint Semantic Learning:
- CB Module: Shared feature extraction (ResNet-50)
- FR Module: Image restoration branch
- DINO: Object detection branch (MMDetection)

Both branches optimize together with shared features!
"""

from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import sys
import argparse
from tqdm import tqdm
import json
import numpy as np

# Add DSNET to path
sys.path.append(str((Path(__file__).resolve().parent / '.').resolve()))
sys.path.append(str((Path(__file__).resolve().parent / '../WUNET').resolve()))

from dsnet_dino import DSNet_DINO
from weather_unet import WeatherUNet
import cv2
from PIL import Image
import torchvision.transforms as transforms

print("="*70)
print("DSNet + DINO Training - High Fog (Beta=0.02)")
print("="*70)

class FoggyCityscapesDataset(Dataset):
    """
    Dataset for DSNet training
    Returns: foggy image, clear image, annotations
    """
    
    def __init__(self, foggy_root, clear_root, beta=0.02, split='train', img_size=512):
        self.foggy_root = Path(foggy_root)
        self.clear_root = Path(clear_root)
        self.beta = beta
        self.split = split
        self.img_size = img_size
        
        # Find all foggy images for this beta
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
        
        return {
            'foggy': foggy_tensor,
            'clear': clear_tensor,
            'image_path': str(foggy_path)
        }


class DSNetTrainer:
    """
    Trainer for DSNet with joint loss
    """
    
    def __init__(self, model, train_loader, val_loader, device, save_dir):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True, parents=True)
        
        # Multi-task loss weights
        self.alpha = 1.0  # Restoration loss
        self.beta = 0.5   # Detection loss (placeholder for now)
        self.gamma = 0.5  # Perceptual loss
        
        # Restoration loss
        self.l1_loss = nn.L1Loss()
        
        # Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=1e-4,
            weight_decay=1e-4
        )
        
        # Scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=100,
            eta_min=1e-6
        )
        
        # Tracking
        self.best_val_loss = float('inf')
        self.history = []
        
    def train_epoch(self, epoch):
        """Train one epoch"""
        self.model.train()
        total_loss = 0
        
        for batch in tqdm(self.train_loader, desc=f"Epoch {epoch}"):
            foggy = batch['foggy'].to(self.device)
            clear = batch['clear'].to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            restored = self.model(foggy)
            
            # Compute loss (restoration only for now)
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
                
                # Forward
                restored = self.model(foggy)
                
                # Loss
                restoration_loss = self.l1_loss(restored, clear)
                loss = self.alpha * restoration_loss
                total_loss += loss.item()
                
                # PSNR
                mse = ((restored - clear) ** 2).mean()
                psnr = 10 * torch.log10(1.0 / (mse + 1e-8))
                psnr_values.append(psnr.item())
        
        avg_loss = total_loss / len(self.val_loader)
        avg_psnr = np.mean(psnr_values)
        
        # Save best model
        if avg_loss < self.best_val_loss:
            self.best_val_loss = avg_loss
            self.save_checkpoint(epoch, is_best=True)
            print(f"  ✓ New best model! Val Loss: {avg_loss:.6f}, PSNR: {avg_psnr:.2f} dB")
        
        return avg_loss, avg_psnr
    
    def save_checkpoint(self, epoch, is_best=False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'history': self.history
        }
        
        # Save last
        torch.save(checkpoint, self.save_dir / 'dsnet_last.pth')
        
        # Save best
        if is_best:
            torch.save(checkpoint, self.save_dir / 'dsnet_best.pth')
    
    def train(self, epochs):
        """Full training loop"""
        print(f"\nStarting training for {epochs} epochs...")
        print(f"Save directory: {self.save_dir}")
        
        for epoch in range(1, epochs + 1):
            print(f"\n{'='*70}")
            print(f"Epoch {epoch}/{epochs}")
            print(f"{'='*70}")
            
            # Train
            train_loss = self.train_epoch(epoch)
            print(f"Train Loss: {train_loss:.6f}")
            
            # Validate
            val_loss, val_psnr = self.validate(epoch)
            print(f"Val Loss: {val_loss:.6f}, Val PSNR: {val_psnr:.2f} dB")
            
            # Update scheduler
            self.scheduler.step()
            
            # Save history
            self.history.append({
                'epoch': epoch,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'val_psnr': val_psnr,
                'lr': self.optimizer.param_groups[0]['lr']
            })
            
            # Save history to JSON
            with open(self.save_dir / 'training_history.json', 'w') as f:
                json.dump(self.history, f, indent=2)
        
        print("\n" + "="*70)
        print("Training Complete!")
        print(f"Best validation loss: {self.best_val_loss:.6f}")
        print(f"Models saved to: {self.save_dir}")
        print("="*70)


def main():
    parser = argparse.ArgumentParser(description='Train DSNet+DINO on High Fog')
    parser.add_argument('--foggy-root', type=str,
                        default='/home/as19197/new_project/Foggy_Cityscapes_dataset/leftImg8bit_foggy',
                        help='Root directory of foggy images')
    parser.add_argument('--clear-root', type=str,
                        default='/home/as19197/new_project/Foggy_Cityscapes_dataset/leftImg8bit',
                        help='Root directory of clear images')
    parser.add_argument('--beta', type=float, default=0.02,
                        help='Fog density (0.02 for high fog)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=8,
                        help='Batch size')
    parser.add_argument('--img-size', type=int, default=512,
                        help='Image size')
    parser.add_argument('--gpus', type=str, default='0,1,2',
                        help='GPU devices')
    parser.add_argument('--save-dir', type=str,
                        default='/home/as19197/new_project/checkpoints/dsnet_beta0.02',
                        help='Save directory')
    
    args = parser.parse_args()
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    
    # Create datasets
    print("\n[1] Creating datasets...")
    train_dataset = FoggyCityscapesDataset(
        args.foggy_root, args.clear_root,
        beta=args.beta, split='train', img_size=args.img_size
    )
    val_dataset = FoggyCityscapesDataset(
        args.foggy_root, args.clear_root,
        beta=args.beta, split='val', img_size=args.img_size
    )
    
    # Dataloaders
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size,
        shuffle=True, num_workers=8, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size,
        shuffle=False, num_workers=8, pin_memory=True
    )
    
    # Create model
    print("\n[2] Creating DSNet model...")
    model = DSNet_DINO(load_dino=False, pretrained_cb=True)
    
    # Multi-GPU
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)
    
    model = model.to(device)
    print("✓ Model created")
    
    # Trainer
    print("\n[3] Creating trainer...")
    trainer = DSNetTrainer(
        model, train_loader, val_loader,
        device, args.save_dir
    )
    
    # Train
    print("\n[4] Starting training...")
    trainer.train(args.epochs)


if __name__ == '__main__':
    main()
