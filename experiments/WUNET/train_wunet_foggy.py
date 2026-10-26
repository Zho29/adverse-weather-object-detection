"""
WUNET Training Script for Foggy Cityscapes

Train WUNET to restore foggy images to clear weather conditions.
Follows paper methodology with 3 fog densities: low (0.005), medium (0.01), high (0.02).

Usage:
    # Train on medium fog (recommended first)
    python train_wunet_foggy.py --beta 0.01 --epochs 200
    
    # Train on high fog  
    python train_wunet_foggy.py --beta 0.02 --epochs 200
    
    # Train on low fog
    python train_wunet_foggy.py --beta 0.005 --epochs 200
"""

import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import argparse
from tqdm import tqdm
import json
import time
import warnings
import matplotlib.pyplot as plt

# Import WUNET model and dataset
from weather_unet import WeatherUNet
from foggy_dataset import get_dataloaders


def calculate_psnr(img1, img2, max_val=2.0):
    """Return scalar tensor PSNR; normalized RGB in [-1, 1] has range 2."""
    if max_val <= 0:
        raise ValueError("PSNR data range must be positive")
    mse = torch.mean((img1 - img2) ** 2)
    # A perfect reconstruction naturally produces a tensor containing infinity.
    return 20 * torch.log10(max_val / torch.sqrt(mse))


class WUNETFoggyTrainer:
    """
    Trainer for WUNET on Foggy Cityscapes Dataset
    
    Args:
        model: WUNET model
        device: 'cuda' or 'cpu'
        learning_rate: Learning rate (paper: 0.01 for RGB)
        beta: Fog density (0.005, 0.01, or 0.02)
    """
    
    def __init__(self, model, device='cuda', learning_rate=0.01, beta=0.01, multi_gpu=False):
        self.device = device
        self.beta = beta
        self.multi_gpu = multi_gpu
        
        # Multi-GPU setup
        if multi_gpu and torch.device(device).type == 'cuda' and torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs for training")
            self.model = nn.DataParallel(model)
            self.model = self.model.to(device)
        else:
            self.model = model.to(device)
        
        # Optimizer and loss (from paper)
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=10
        )
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'val_psnr': [],
            'learning_rates': [],
            'psnr_data_ranges': []
        }
        
        self.best_val_loss = float('inf')
        self.epochs_no_improve = 0

    def checkpoint_state(self, epoch, val_loss, val_psnr):
        """Store enough state to continue training without restarting the schedule."""
        model = self.model.module if isinstance(self.model, nn.DataParallel) else self.model
        return {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'val_loss': val_loss,
            'val_psnr': val_psnr,
            'best_val_loss': self.best_val_loss,
            'epochs_no_improve': self.epochs_no_improve,
            'beta': self.beta,
            'history': self.history,
            'psnr_data_range': 2.0,
        }

    def load_checkpoint(self, path):
        """Restore a saved run, including checkpoints from the original trainer."""
        path = Path(path)
        with path.open('rb') as handle:
            if handle.read(80).startswith(b'version https://git-lfs.github.com/spec/v1'):
                raise ValueError(f"{path} is a Git LFS pointer; run git lfs pull first")
        checkpoint = torch.load(path, map_location=self.device, weights_only=True)
        if checkpoint.get('beta', self.beta) != self.beta:
            raise ValueError("Checkpoint fog density does not match --beta")
        state = {key.removeprefix('module.'): value
                 for key, value in checkpoint['model_state_dict'].items()}
        model = self.model.module if isinstance(self.model, nn.DataParallel) else self.model
        model.load_state_dict(state)
        if 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        else:
            warnings.warn("Legacy checkpoint has no scheduler state; its schedule restarts.")
        for key in self.history:
            if key in checkpoint.get('history', {}):
                self.history[key] = list(checkpoint['history'][key])
        if not self.history['psnr_data_ranges'] and self.history['val_psnr']:
            self.history['psnr_data_ranges'] = [
                checkpoint.get('psnr_data_range', 1.0)
            ] * len(self.history['val_psnr'])
        if checkpoint.get('psnr_data_range', 1.0) != 2.0:
            warnings.warn("Legacy PSNR used range 1 on [-1, 1] data. New PSNR uses range 2 "
                          "and is 6.02 dB higher for the same reconstruction; history is retained.")
        self.best_val_loss = checkpoint.get('best_val_loss', checkpoint.get('val_loss', float('inf')))
        self.epochs_no_improve = checkpoint.get('epochs_no_improve', 0)
        return int(checkpoint['epoch']) + 1

    def train_epoch(self, train_loader, epoch):
        """Train for one epoch"""
        if len(train_loader) == 0:
            raise ValueError('Training loader is empty')
        self.model.train()
        total_loss = 0
        total_samples = 0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch} [Train]')
        for batch_idx, batch in enumerate(pbar):
            foggy = batch['foggy'].to(self.device)
            clear = batch['clear'].to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            output = self.model(foggy)
            loss = self.criterion(output, clear)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item() * foggy.shape[0]
            total_samples += foggy.shape[0]
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_loss = total_loss / total_samples
        return avg_loss
    
    def validate(self, val_loader, epoch):
        """Validate the model"""
        if len(val_loader) == 0:
            raise ValueError('Validation loader is empty')
        self.model.eval()
        total_loss = 0
        total_psnr = 0
        total_samples = 0
        
        with torch.no_grad():
            pbar = tqdm(val_loader, desc=f'Epoch {epoch} [Val]')
            for batch in pbar:
                foggy = batch['foggy'].to(self.device)
                clear = batch['clear'].to(self.device)
                
                # Forward pass
                output = self.model(foggy)
                loss = self.criterion(output, clear)
                psnr = torch.stack([calculate_psnr(pred, target)
                                    for pred, target in zip(output, clear)]).mean()
                
                total_loss += loss.item() * foggy.shape[0]
                total_psnr += psnr.item() * foggy.shape[0]
                total_samples += foggy.shape[0]
                
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'psnr': f'{psnr.item():.2f}'
                })
        
        avg_loss = total_loss / total_samples
        avg_psnr = total_psnr / total_samples
        
        return avg_loss, avg_psnr
    
    def visualize_restoration(self, val_loader, epoch, save_dir, num_samples=4):
        """Save visual comparison of foggy/restored/clear images"""
        self.model.eval()
        save_dir = Path(save_dir) / "visualizations"
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Get one batch from val loader
        if num_samples <= 0:
            raise ValueError('num_samples must be positive')
        try:
            batch = next(iter(val_loader))
        except StopIteration as exc:
            raise ValueError('Validation loader is empty') from exc
        num_samples = min(num_samples, len(batch['foggy']))
        if num_samples == 0:
            raise ValueError('Validation batch is empty')
        foggy = batch['foggy'][:num_samples].to(self.device)
        clear = batch['clear'][:num_samples].to(self.device)
        
        # Generate restored images
        with torch.no_grad():
            restored = self.model(foggy)
        
        # Move to CPU and denormalize
        def denorm(tensor):
            # Denormalize from [-1,1] to [0,1]
            return (tensor + 1) / 2
        
        foggy_cpu = denorm(foggy.cpu())
        clear_cpu = denorm(clear.cpu())
        restored_cpu = denorm(restored.cpu())
        
        # Create comparison grid
        fig, axes = plt.subplots(num_samples, 3, figsize=(12, num_samples*4))
        if num_samples == 1:
            axes = axes.reshape(1, -1)
        
        for i in range(num_samples):
            # Foggy image
            axes[i, 0].imshow(foggy_cpu[i].permute(1, 2, 0).clamp(0, 1))
            axes[i, 0].set_title('Foggy Input')
            axes[i, 0].axis('off')
            
            # Restored image
            axes[i, 1].imshow(restored_cpu[i].permute(1, 2, 0).clamp(0, 1))
            axes[i, 1].set_title('WUNET Restored')
            axes[i, 1].axis('off')
            
            # Clear ground truth
            axes[i, 2].imshow(clear_cpu[i].permute(1, 2, 0).clamp(0, 1))
            axes[i, 2].set_title('Clear Ground Truth')
            axes[i, 2].axis('off')
        
        plt.tight_layout()
        save_path = save_dir / f'epoch_{epoch:03d}_beta{self.beta}.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ Visualization saved: {save_path}")
    
    def train(self, train_loader, val_loader, num_epochs, save_dir, save_every=10, start_epoch=1):
        """Train up to num_epochs total, continuing at start_epoch when resuming."""
        if save_every <= 0 or start_epoch <= 0 or num_epochs < start_epoch:
            raise ValueError('Require save_every > 0 and 1 <= start_epoch <= num_epochs')
        if len(train_loader) == 0 or len(val_loader) == 0:
            raise ValueError('Training and validation loaders must contain samples')
        save_dir = Path(save_dir)
        if start_epoch == 1 and save_dir.exists() and any(save_dir.iterdir()):
            raise FileExistsError('A new run needs an empty save directory; choose another path or resume a checkpoint.')
        save_dir.mkdir(parents=True, exist_ok=True)
        
        print("=" * 70)
        print(f"Training WUNET on Foggy Cityscapes (beta={self.beta})")
        print("=" * 70)
        print(f"Device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"Training samples: {len(train_loader.dataset)}")
        print(f"Validation samples: {len(val_loader.dataset)}")
        print(f"Epochs: {num_epochs}")
        print(f"Save directory: {save_dir}")
        print("=" * 70)
        
        start_time = time.time()
        
        for epoch in range(start_epoch, num_epochs + 1):
            print(f"\nEpoch {epoch}/{num_epochs}")
            print("-" * 70)
            
            # Train
            train_loss = self.train_epoch(train_loader, epoch)
            
            # Validate
            val_loss, val_psnr = self.validate(val_loader, epoch)
            
            # Learning rate scheduling
            self.scheduler.step(val_loss)
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['val_psnr'].append(val_psnr)
            self.history['learning_rates'].append(current_lr)
            self.history['psnr_data_ranges'].append(2.0)
            
            # Print epoch summary
            print(f"\nSummary:")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val Loss: {val_loss:.4f}")
            print(f"  Val PSNR: {val_psnr:.2f} dB")
            print(f"  Learning Rate: {current_lr:.6f}")
            
            # Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.epochs_no_improve = 0
                best_path = save_dir / f'wunet_foggy_beta{self.beta}_best.pth'
                torch.save(self.checkpoint_state(epoch, val_loss, val_psnr), best_path)
                print(f"  ✓ New best model saved! (val_loss: {val_loss:.4f})")
            else:
                self.epochs_no_improve += 1
            
            # Save periodic checkpoint and visualizations
            if epoch % save_every == 0:
                checkpoint_path = save_dir / f'wunet_foggy_beta{self.beta}_epoch{epoch}.pth'
                torch.save(self.checkpoint_state(epoch, val_loss, val_psnr), checkpoint_path)
                print(f"  ✓ Checkpoint saved (epoch {epoch})")
                
                # Save visual comparison
                self.visualize_restoration(val_loader, epoch, save_dir, num_samples=4)
            
            # Also save visualization at epoch 1 to see initial quality
            if epoch == 1 and epoch % save_every != 0:
                self.visualize_restoration(val_loader, epoch, save_dir, num_samples=4)
            
            # Save training history
            history_path = save_dir / f'training_history_beta{self.beta}.json'
            with open(history_path, 'w') as f:
                json.dump(self.history, f, indent=2)
        
        # Training complete
        total_time = time.time() - start_time
        print("\n" + "=" * 70)
        print("Training Complete!")
        print(f"Total time: {total_time/3600:.2f} hours")
        print(f"Best validation loss: {self.best_val_loss:.4f}")
        print(f"Final model saved to: {save_dir}")
        print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description='Train WUNET on Foggy Cityscapes')
    parser.add_argument('--root-dir', type=str, 
                        default=str(Path(__file__).resolve().parents[1] / 'Foggy_Cityscapes_dataset'),
                        help='Root directory of Foggy Cityscapes dataset')
    parser.add_argument('--beta', type=float, default=0.01,
                        choices=[0.005, 0.01, 0.02],
                        help='Fog density: 0.005 (low), 0.01 (medium), 0.02 (high)')
    parser.add_argument('--epochs', type=int, default=200,
                        help='Number of epochs (paper: 200)')
    parser.add_argument('--batch-size', type=int, default=24,
                        help='Batch size (paper: 24)')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='Learning rate (paper: 0.01 for RGB)')
    parser.add_argument('--img-size', type=int, default=256,
                        help='Image size for training')
    parser.add_argument('--save-dir', type=str, default=str(Path(__file__).resolve().parents[2] / 'outputs' / 'wunet_foggy'),
                        help='Directory to save checkpoints')
    parser.add_argument('--save-every', type=int, default=10,
                        help='Save checkpoint every N epochs')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='Number of dataloader workers')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--multi-gpu', action='store_true',
                        help='Use multiple GPUs (DataParallel)')
    
    args = parser.parse_args()
    
    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nUsing device: {device}")
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
    
    # Create model
    print("\nInitializing WUNET model...")
    model = WeatherUNet(n_channels=3, n_classes=3, bilinear=True)
    
    # Create dataloaders
    print(f"\nCreating dataloaders for beta={args.beta}...")
    train_loader, val_loader = get_dataloaders(
        root_dir=args.root_dir,
        beta=args.beta,
        batch_size=args.batch_size,
        img_size=args.img_size,
        num_workers=args.num_workers
    )
    
    # Create trainer
    trainer = WUNETFoggyTrainer(
        model=model,
        device=device,
        learning_rate=args.lr,
        beta=args.beta,
        multi_gpu=args.multi_gpu
    )
    
    # Restore optimizer, scheduler, history, and the next epoch before training.
    start_epoch = trainer.load_checkpoint(args.resume) if args.resume else 1

    # Train
    trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=args.epochs,
        save_dir=args.save_dir,
        save_every=args.save_every,
        start_epoch=start_epoch
    )
    
    print(f"\n✓ Training complete! Best model saved to:")
    print(f"  {args.save_dir}/wunet_foggy_beta{args.beta}_best.pth")


if __name__ == '__main__':
    main()
