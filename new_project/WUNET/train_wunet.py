"""
Training Script for Weather UNet (WUNet)
Implements the exact training configuration from arXiv 2407.02581

This script provides:
1. Training loop with paper's hyperparameters
2. Data loading utilities
3. Evaluation metrics (MSE, PSNR, SSIM)
4. Model checkpointing
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from weather_unet import WeatherUNet
from pathlib import Path
import numpy as np
from PIL import Image


class WeatherDataset(Dataset):
    """
    Dataset for weather denoising task.
    
    Args:
        adverse_dir: Directory containing adverse weather images
        clear_dir: Directory containing clear weather images (ground truth)
        transform: Optional transforms to apply
    """
    def __init__(self, adverse_dir, clear_dir, transform=None):
        self.adverse_dir = Path(adverse_dir)
        self.clear_dir = Path(clear_dir)
        self.transform = transform
        
        # Assuming matching filenames
        self.image_files = sorted(list(self.adverse_dir.glob('*.png')) + 
                                  list(self.adverse_dir.glob('*.jpg')))
        
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        adverse_path = self.image_files[idx]
        clear_path = self.clear_dir / adverse_path.name
        
        adverse_img = Image.open(adverse_path).convert('RGB')
        clear_img = Image.open(clear_path).convert('RGB')
        
        if self.transform:
            adverse_img = self.transform(adverse_img)
            clear_img = self.transform(clear_img)
            
        return adverse_img, clear_img


class WUNetTrainer:
    """
    Trainer class implementing the exact configuration from the paper.
    
    Paper Configuration:
    - Optimizer: Adam
    - Learning Rate: 0.01 (RGB) or 0.001 (HSV)
    - Epochs: 200
    - Batch Size: 24
    - Loss: MSE (Mean Squared Error)
    """
    def __init__(
        self,
        model,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        learning_rate=0.01,  # Paper default for RGB
        batch_size=24,       # Paper default
        num_epochs=200,      # Paper default
    ):
        self.model = model.to(device)
        self.device = device
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        
        # Paper uses Adam optimizer and MSE loss
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()
        
        # For logging
        self.train_losses = []
        self.val_losses = []
        
    def train_epoch(self, train_loader):
        """Train for one epoch"""
        self.model.train()
        epoch_loss = 0.0
        
        for batch_idx, (adverse, clear) in enumerate(train_loader):
            adverse = adverse.to(self.device)
            clear = clear.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            output = self.model(adverse)
            
            # Compute loss
            loss = self.criterion(output, clear)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            epoch_loss += loss.item()
            
            if batch_idx % 50 == 0:
                print(f"  Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.6f}")
        
        return epoch_loss / len(train_loader)
    
    def validate(self, val_loader):
        """Validate the model"""
        self.model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for adverse, clear in val_loader:
                adverse = adverse.to(self.device)
                clear = clear.to(self.device)
                
                output = self.model(adverse)
                loss = self.criterion(output, clear)
                val_loss += loss.item()
        
        return val_loss / len(val_loader)
    
    def train(self, train_loader, val_loader=None, save_dir='checkpoints', weather='fog'):
        """
        Full training loop
        
        Args:
            train_loader: DataLoader for training data
            val_loader: Optional DataLoader for validation
            save_dir: Directory to save model checkpoints
            weather: Weather condition for naming checkpoints
        """
        # Create weather-specific checkpoint directory
        save_dir = Path(save_dir) / f'{weather}_checkpoints'
        save_dir.mkdir(exist_ok=True, parents=True)
        
        best_val_loss = float('inf')
        
        print(f"Starting training for {self.num_epochs} epochs")
        print(f"Device: {self.device}")
        print(f"Batch size: {self.batch_size}")
        print(f"Saving to: {save_dir}")
        print("=" * 60)
        
        for epoch in range(1, self.num_epochs + 1):
            print(f"\nEpoch {epoch}/{self.num_epochs}")
            print("-" * 60)
            
            # Train
            train_loss = self.train_epoch(train_loader)
            self.train_losses.append(train_loss)
            print(f"Training Loss: {train_loss:.6f}")
            
            # Validate
            if val_loader:
                val_loss = self.validate(val_loader)
                self.val_losses.append(val_loss)
                print(f"Validation Loss: {val_loss:.6f}")
                
                # Save best model
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    checkpoint_path = save_dir / f'{weather}_best_model.pth'
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'train_loss': train_loss,
                        'val_loss': val_loss,
                    }, checkpoint_path)
                    print(f"✓ Saved best model (val_loss: {val_loss:.6f})")
            
            # Save periodic checkpoint every 50 epochs
            if epoch % 50 == 0:
                checkpoint_path = save_dir / f'{weather}_checkpoint_epoch_{epoch}.pth'
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'train_loss': train_loss,
                }, checkpoint_path)
                print(f"✓ Saved checkpoint at epoch {epoch}")
        
        print("\n" + "=" * 60)
        print("Training completed!")
        print(f"Best validation loss: {best_val_loss:.6f}")
        
        return self.train_losses, self.val_losses
        """
        Full training loop
        
        Args:
            train_loader: DataLoader for training data
            val_loader: Optional DataLoader for validation
            save_dir: Directory to save model checkpoints
        """
        save_dir = Path(save_dir)
        save_dir.mkdir(exist_ok=True)
        
        best_val_loss = float('inf')
        
        print(f"Starting training for {self.num_epochs} epochs")
        print(f"Device: {self.device}")
        print(f"Batch size: {self.batch_size}")
        print("=" * 60)
        
        for epoch in range(1, self.num_epochs + 1):
            print(f"\nEpoch {epoch}/{self.num_epochs}")
            print("-" * 60)
            
            # Train
            train_loss = self.train_epoch(train_loader)
            self.train_losses.append(train_loss)
            print(f"Training Loss: {train_loss:.6f}")
            
            # Validate
            if val_loader:
                val_loss = self.validate(val_loader)
                self.val_losses.append(val_loss)
                print(f"Validation Loss: {val_loss:.6f}")
                
                # Save best model
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    checkpoint_path = save_dir / 'best_model.pth'
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'train_loss': train_loss,
                        'val_loss': val_loss,
                    }, checkpoint_path)
                    print(f"✓ Saved best model (val_loss: {val_loss:.6f})")
            
            # Save periodic checkpoint
            if epoch % 50 == 0:
                checkpoint_path = save_dir / f'checkpoint_epoch_{epoch}.pth'
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'train_loss': train_loss,
                }, checkpoint_path)
                print(f"✓ Saved checkpoint at epoch {epoch}")
        
        print("\n" + "=" * 60)
        print("Training completed!")
        print(f"Best validation loss: {best_val_loss:.6f}")
        
        return self.train_losses, self.val_losses


def calculate_psnr(img1, img2):
    """Calculate Peak Signal-to-Noise Ratio"""
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * torch.log10(1.0 / torch.sqrt(mse))


def get_transforms(img_size=256):
    """
    Get image transforms for training/validation
    
    Paper uses 256x256 or crops for efficiency
    """
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])


def main():
    """
    WUNET Training on ACDC Dataset
    
    Usage:
        python train_wunet.py --weather fog --epochs 5
    """
    import argparse
    import sys
    import os
    sys.path.append(os.path.dirname(__file__))
    from acdc_dataset import ACDCDataset
    
    parser = argparse.ArgumentParser(description='Train WUNET on ACDC Dataset')
    parser.add_argument('--weather', type=str, default='fog',
                       choices=['fog', 'rain', 'snow'],
                       help='Weather condition to train on')
    parser.add_argument('--epochs', type=int, default=5,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=8,
                       help='Batch size for training')
    parser.add_argument('--lr', type=float, default=0.01,
                       help='Learning rate')
    parser.add_argument('--img_size', type=int, default=512,
                       help='Image size for training')
    parser.add_argument('--dataset_root', type=str, 
                       default='/home/as19197/new_project/ACDC_dataset',
                       help='Path to ACDC dataset root')
    
    args = parser.parse_args()
    
    # Configuration
    CONFIG = {
        'learning_rate': args.lr,
        'batch_size': args.batch_size,
        'num_epochs': args.epochs,
        'img_size': args.img_size,
        'weather': args.weather,
        'dataset_root': args.dataset_root,
    }
    
    print("WUNET Training on ACDC Dataset")
    print("=" * 60)
    print("Configuration:")
    for key, value in CONFIG.items():
        print(f"  {key}: {value}")
    print("=" * 60)
    
    # Load ACDC datasets
    print("\nLoading ACDC datasets...")
    train_dataset = ACDCDataset(
        root_dir=CONFIG['dataset_root'],
        weather=CONFIG['weather'],
        split='train',
        img_size=CONFIG['img_size']
    )
    
    val_dataset = ACDCDataset(
        root_dir=CONFIG['dataset_root'],
        weather=CONFIG['weather'],
        split='val',
        img_size=CONFIG['img_size']
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=CONFIG['batch_size'],
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=CONFIG['batch_size'],
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )
    
    # Initialize model
    model = WeatherUNet(n_channels=3, n_classes=3, bilinear=True)
    print(f"\nModel Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Initialize trainer
    trainer = WUNetTrainer(
        model=model,
        learning_rate=CONFIG['learning_rate'],
        batch_size=CONFIG['batch_size'],
        num_epochs=CONFIG['num_epochs']
    )
    
    # Train the model
    print("\n" + "=" * 60)
    print("Starting Training")
    print("=" * 60)
    train_losses, val_losses = trainer.train(train_loader, val_loader, weather=CONFIG['weather'])
    
    # Plot training history
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss', linewidth=2, marker='o')
    plt.plot(val_losses, label='Validation Loss', linewidth=2, marker='s')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.title(f'WUNET Training History ({CONFIG["weather"]} weather, {CONFIG["num_epochs"]} epochs)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    history_path = f'training_history_{CONFIG["weather"]}_{CONFIG["num_epochs"]}ep.png'
    plt.savefig(history_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Training history saved to {history_path}")
    
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"Best model saved to: checkpoints/{CONFIG['weather']}_checkpoints/{CONFIG['weather']}_best_model.pth")
    print(f"Final training loss: {train_losses[-1]:.6f}")
    print(f"Final validation loss: {val_losses[-1]:.6f}")
    
    # Plot training history
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss', linewidth=2)
    plt.plot(val_losses, label='Validation Loss', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.title(f'WUNET Training History ({CONFIG["weather"]} weather)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('training_history.png', dpi=150, bbox_inches='tight')
    print(f"\n✓ Training history saved to training_history.png")
    
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"Best model saved to: checkpoints/best_model.pth")
    print(f"Final training loss: {train_losses[-1]:.6f}")
    print(f"Final validation loss: {val_losses[-1]:.6f}")


if __name__ == "__main__":
    main()
