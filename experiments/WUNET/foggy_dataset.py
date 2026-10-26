"""
Foggy Cityscapes Dataset Loader for WUNET Training

This dataset pairs foggy images with their corresponding clear weather images:
- Foggy images: leftImg8bit_foggy/{split}/{city}/{basename}_leftImg8bit_foggy_beta_{beta}.png
- Clear images: leftImg8bit/{split}/{city}/{basename}_leftImg8bit.png

Supports three fog densities:
- Low fog: beta = 0.005
- Medium fog: beta = 0.01  
- High fog: beta = 0.02
"""

import torch
from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
import random
import warnings


class FoggyCityscapesDataset(Dataset):
    """
    Dataset for WUNET training on Foggy Cityscapes.
    
    Args:
        root_dir: Root directory of Foggy Cityscapes dataset
        split: 'train', 'val', or 'test'
        beta: Fog density (0.005, 0.01, or 0.02)
        transform: Optional transform to apply to images
        img_size: Resize images to this size (default: 256x256)
    """
    
    def __init__(self, root_dir, split='train', beta=0.01, transform=None, img_size=256):
        if beta not in (0.005, 0.01, 0.02):
            raise ValueError('beta must be 0.005, 0.01, or 0.02')
        if img_size <= 0:
            raise ValueError('img_size must be positive')
        self.root_dir = Path(root_dir)
        self.split = split
        self.beta = beta
        self.img_size = img_size
        
        # Directories
        self.foggy_dir = self.root_dir / "leftImg8bit_foggy" / split
        self.clear_dir = self.root_dir / "leftImg8bit" / split
        
        for directory in (self.foggy_dir, self.clear_dir):
            if not directory.is_dir():
                raise FileNotFoundError(f'Missing dataset directory: {directory}')

        # Find all foggy images for this beta value
        self.foggy_images = sorted(self.foggy_dir.rglob(f"*_beta_{beta}.png"))
        
        # Verify corresponding clear images exist
        self.valid_pairs = []
        for foggy_path in self.foggy_images:
            clear_path = self._get_clear_path(foggy_path)
            if clear_path.exists():
                self.valid_pairs.append((foggy_path, clear_path))
        
        if not self.valid_pairs:
            raise ValueError(f'No foggy/clear image pairs found for {split}, beta={beta} in {self.root_dir}')
        if len(self.valid_pairs) < len(self.foggy_images):
            warnings.warn(f'{len(self.foggy_images) - len(self.valid_pairs)} foggy images have no clear pair')

        print(f"Found {len(self.valid_pairs)} valid foggy/clear pairs for {split} (beta={beta})")
        
        # Default transform if none provided
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                # Normalize to [-1, 1] to match WUNET output activation
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])
        else:
            self.transform = transform
    
    def _get_clear_path(self, foggy_path):
        """
        Convert foggy image path to corresponding clear image path.
        
        Example:
        leftImg8bit_foggy/train/aachen/aachen_000000_000019_leftImg8bit_foggy_beta_0.01.png
        -> leftImg8bit/train/aachen/aachen_000000_000019_leftImg8bit.png
        """
        # Extract city and filename
        city = foggy_path.parent.name
        filename = foggy_path.stem
        
        # Remove fog-specific suffixes to get base name
        # aachen_000000_000019_leftImg8bit_foggy_beta_0.01 -> aachen_000000_000019_leftImg8bit
        base_name = filename.split('_foggy_beta_')[0]
        
        # Construct clear image path
        clear_path = self.clear_dir / city / f"{base_name}.png"
        return clear_path
    
    def __len__(self):
        return len(self.valid_pairs)
    
    def __getitem__(self, idx):
        foggy_path, clear_path = self.valid_pairs[idx]
        
        # Load images
        with Image.open(foggy_path) as image:
            foggy_img = image.convert('RGB')
        with Image.open(clear_path) as image:
            clear_img = image.convert('RGB')
        
        # Apply transforms
        if self.transform:
            # Both images must receive identical random crops/flips. Restore each
            # generator before transforming the clear image so the pair stays aligned.
            torch_state = torch.get_rng_state()
            numpy_state = np.random.get_state()
            python_state = random.getstate()
            foggy_img = self.transform(foggy_img)
            torch.set_rng_state(torch_state)
            np.random.set_state(numpy_state)
            random.setstate(python_state)
            clear_img = self.transform(clear_img)
        
        return {
            'foggy': foggy_img,
            'clear': clear_img,
            'foggy_path': str(foggy_path),
            'clear_path': str(clear_path)
        }


def get_dataloaders(root_dir, beta=0.01, batch_size=24, img_size=256, num_workers=4):
    """
    Create train and validation dataloaders for WUNET training.
    
    Args:
        root_dir: Root directory of Foggy Cityscapes
        beta: Fog density (0.005, 0.01, or 0.02)
        batch_size: Batch size (default: 24 as per paper)
        img_size: Image size for resizing (default: 256)
        num_workers: Number of dataloader workers
        
    Returns:
        train_loader, val_loader
    """
    if batch_size <= 0 or num_workers < 0:
        raise ValueError('batch_size must be positive and num_workers nonnegative')

    # Create datasets
    train_dataset = FoggyCityscapesDataset(
        root_dir=root_dir,
        split='train',
        beta=beta,
        img_size=img_size
    )
    
    val_dataset = FoggyCityscapesDataset(
        root_dir=root_dir,
        split='val',
        beta=beta,
        img_size=img_size
    )
    
    # Create dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    
    return train_loader, val_loader


if __name__ == "__main__":
    # Test the dataset
    print("Testing Foggy Cityscapes Dataset Loader")
    print("=" * 60)
    
    root_dir = "/mnt/Data/as19197/new_project/Foggy_Cityscapes_dataset"
    
    # Test all three fog densities
    for beta in [0.005, 0.01, 0.02]:
        print(f"\nTesting beta={beta}:")
        dataset = FoggyCityscapesDataset(root_dir, split='train', beta=beta, img_size=256)
        
        if len(dataset) > 0:
            sample = dataset[0]
            print(f"  Foggy image shape: {sample['foggy'].shape}")
            print(f"  Clear image shape: {sample['clear'].shape}")
            print(f"  Foggy path: {sample['foggy_path']}")
            print(f"  Clear path: {sample['clear_path']}")
        
    # Test dataloader
    print("\n" + "=" * 60)
    print("Testing DataLoader (beta=0.01):")
    train_loader, val_loader = get_dataloaders(root_dir, beta=0.01, batch_size=4)
    
    batch = next(iter(train_loader))
    print(f"  Train batch - Foggy: {batch['foggy'].shape}, Clear: {batch['clear'].shape}")
    
    batch = next(iter(val_loader))
    print(f"  Val batch - Foggy: {batch['foggy'].shape}, Clear: {batch['clear'].shape}")
    
    print("\n✓ Dataset loader test passed!")
