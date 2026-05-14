"""
Train YOLOv8 on WUNET-Restored Foggy Cityscapes Images

This script implements the two-stage pipeline:
1. Apply pretrained WUNET to foggy images to generate clear weather images
2. Train YOLOv8 on the restored images for person and vehicle detection

Usage:
    # Train YOLO on WUNET-restored images (medium fog)
    python train_yolo_with_wunet.py --wunet-checkpoint ../checkpoints/wunet_foggy/wunet_foggy_beta0.01_best.pth
    
    # Resume training
    python train_yolo_with_wunet.py --wunet-checkpoint ../checkpoints/wunet_foggy/wunet_foggy_beta0.01_best.pth --resume runs/train/exp/weights/last.pt
"""

import torch
import sys
from pathlib import Path
import argparse
from tqdm import tqdm
import shutil

# Add WUNET to path
sys.path.append(str(Path(__file__).parent / '../WUNET'))
from weather_unet import WeatherUNet

# Import ultralytics YOLOv8
try:
    from ultralytics import YOLO
except ImportError:
    print("Error: ultralytics not installed. Installing...")
    import subprocess
    subprocess.run(['pip', 'install', 'ultralytics', '-q'])
    from ultralytics import YOLO

from PIL import Image
import torchvision.transforms as transforms


class WUNETImageRestorer:
    """Apply WUNET to restore foggy images to clear weather"""
    
    def __init__(self, checkpoint_path, device='cuda', use_multi_gpu=True):
        self.device = device
        
        # Load WUNET model
        wunet_model = WeatherUNet(n_channels=3, n_classes=3, bilinear=True)
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Handle DataParallel checkpoint (strip 'module.' prefix)
        state_dict = checkpoint['model_state_dict']
        if list(state_dict.keys())[0].startswith('module.'):
            # Remove 'module.' prefix from all keys
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        
        wunet_model.load_state_dict(state_dict)
        
        # Use DataParallel for multi-GPU acceleration
        if use_multi_gpu and torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs for WUNET restoration")
            self.model = torch.nn.DataParallel(wunet_model)
        else:
            self.model = wunet_model
            
        self.model = self.model.to(device)
        self.model.eval()
        
        print(f"✓ Loaded WUNET from: {checkpoint_path}")
        print(f"  Beta: {checkpoint.get('beta', 'unknown')}")
        print(f"  Val Loss: {checkpoint.get('val_loss', 'unknown'):.4f}")
        print(f"  Val PSNR: {checkpoint.get('val_psnr', 'unknown'):.2f} dB")
        
        # Transform for WUNET input
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        
        self.inverse_transform = transforms.Compose([
            transforms.Normalize(mean=[-1, -1, -1], std=[2, 2, 2])  # Denormalize from [-1,1] to [0,1]
        ])
    
    def restore_image(self, img_path, save_path=None):
        """
        Restore a single foggy image using WUNET.
        
        Args:
            img_path: Path to foggy image
            save_path: Optional path to save restored image
            
        Returns:
            PIL Image of restored image
        """
        # Load and preprocess
        img = Image.open(img_path).convert('RGB')
        original_size = img.size
        
        # Apply WUNET
        with torch.no_grad():
            img_tensor = self.transform(img).unsqueeze(0).to(self.device)
            restored_tensor = self.model(img_tensor)
            restored_tensor = self.inverse_transform(restored_tensor.squeeze(0).cpu())
            restored_tensor = torch.clamp(restored_tensor, 0, 1)
        
        # Convert back to PIL
        restored_img = transforms.ToPILImage()(restored_tensor)
        restored_img = restored_img.resize(original_size, Image.BILINEAR)
        
        # Save if requested
        if save_path:
            restored_img.save(save_path)
        
        return restored_img
    
    def restore_dataset(self, yolo_dataset_root, output_root):
        """
        Restore all images in a YOLO dataset.
        
        Args:
            yolo_dataset_root: Root of YOLO dataset (contains images/ and labels/)
            output_root: Output root for restored dataset
        """
        yolo_root = Path(yolo_dataset_root)
        output_root = Path(output_root)
        
        print(f"\nRestoring YOLO dataset with WUNET...")
        print(f"  Input: {yolo_root}")
        print(f"  Output: {output_root}")
        
        # Copy labels directory (unchanged)
        print("\n  Copying labels...")
        labels_src = yolo_root / "labels"
        labels_dst = output_root / "labels"
        if labels_dst.exists():
            shutil.rmtree(labels_dst)
        shutil.copytree(labels_src, labels_dst)
        
        # Restore images for each split
        for split in ['train', 'val', 'test']:
            images_src = yolo_root / "images" / split
            images_dst = output_root / "images" / split
            
            if not images_src.exists():
                continue
            
            images_dst.mkdir(parents=True, exist_ok=True)
            
            image_files = list(images_src.glob("*.png")) + list(images_src.glob("*.jpg"))
            print(f"\n  Restoring {split} images ({len(image_files)} images)...")
            
            # Batch processing for multi-GPU speedup
            batch_size = 12 if torch.cuda.device_count() > 1 else 4
            for i in tqdm(range(0, len(image_files), batch_size), desc=f"  {split}"):
                batch_paths = image_files[i:i+batch_size]
                batch_images = []
                batch_sizes = []
                
                # Load batch
                for img_path in batch_paths:
                    img = Image.open(img_path).convert('RGB')
                    batch_sizes.append(img.size)
                    batch_images.append(self.transform(img))
                
                # Process batch
                with torch.no_grad():
                    batch_tensor = torch.stack(batch_images).to(self.device)
                    restored_batch = self.model(batch_tensor)
                    restored_batch = self.inverse_transform(restored_batch.cpu())
                    restored_batch = torch.clamp(restored_batch, 0, 1)
                
                # Save batch
                for j, img_path in enumerate(batch_paths):
                    save_path = images_dst / img_path.name
                    restored_img = transforms.ToPILImage()(restored_batch[j])
                    restored_img = restored_img.resize(batch_sizes[j], Image.BILINEAR)
                    restored_img.save(save_path)
        
        # Copy data.yaml  
        yaml_src = yolo_root / "data.yaml"
        yaml_dst = output_root / "data.yaml"
        
        # Update paths in yaml
        with open(yaml_src, 'r') as f:
            yaml_content = f.read()
        
        yaml_content = yaml_content.replace(str(yolo_root), str(output_root))
        
        with open(yaml_dst, 'w') as f:
            f.write(yaml_content)
        
        print(f"\n✓ Restored dataset saved to: {output_root}")


def train_yolo(data_yaml, epochs=100, batch_size=16, img_size=640, model_name='yolov8n.pt', resume=None, gpus='0'):
    """
    Train YOLOv8 on the dataset.
    
    Args:
        data_yaml: Path to data.yaml config
        epochs: Number of training epochs
        batch_size: Batch size
        img_size: Image size for training
        model_name: YOLO model variant (yolov8n, yolov8s, yolov8m, etc.)
        resume: Path to checkpoint to resume from
        gpus: GPU devices to use (e.g., '0', '0,1,2' for multi-GPU)
    """
    print("\n" + "=" * 70)
    print("Training YOLOv8 on WUNET-Restored Images")
    print("=" * 70)
    
    # Load YOLO model
    if resume:
        print(f"\nResuming from: {resume}")
        model = YOLO(resume)
    else:
        print(f"\nInitializing YOLOv8 model: {model_name}")
        model = YOLO(model_name)
    
    # Train (YOLOv8 automatically handles multi-GPU if multiple devices specified)
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        batch=batch_size,
        imgsz=img_size,
        device=gpus,  # Use specified GPU(s)
        project='../yolo_training/runs',
        name='wunet_yolo',
        exist_ok=True,
        verbose=True
    )
    
    print("\n" + "=" * 70)
    print("Training Complete!")
    print("=" * 70)
    
    return model, results


def main():
    parser = argparse.ArgumentParser(description='Train YOLO on WUNET-restored images')
    parser.add_argument('--wunet-checkpoint', type=str, required=True,
                        help='Path to trained WUNET checkpoint')
    parser.add_argument('--yolo-dataset', type=str,
                        default='/mnt/Data/as19197/new_project/yolo_dataset',
                        help='Path to YOLO dataset (foggy images)')
    parser.add_argument('--restored-dataset', type=str,
                        default='/mnt/Data/as19197/new_project/yolo_dataset_restored',
                        help='Path to save restored dataset')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=16,
                        help='Batch size')
    parser.add_argument('--img-size', type=int, default=640,
                        help='Image size for YOLO training')
    parser.add_argument('--model', type=str, default='yolov8n.pt',
                        choices=['yolov8n.pt', 'yolov8s.pt', 'yolov8m.pt', 'yolov8l.pt', 'yolov8x.pt'],
                        help='YOLO model variant')
    parser.add_argument('--resume', type=str, default=None,
                        help='Resume from checkpoint')
    parser.add_argument('--skip-restore', action='store_true',
                        help='Skip restoration step (use existing restored dataset)')
    parser.add_argument('--gpus', type=str, default='0,1,2',
                        help='GPU devices to use (e.g., "0" for single GPU, "0,1,2" for 3 GPUs)')
    
    args = parser.parse_args()
    
    # Step 1: Restore foggy images using WUNET
    if not args.skip_restore:
        restorer = WUNETImageRestorer(
            checkpoint_path=args.wunet_checkpoint,
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )
        
        restorer.restore_dataset(
            yolo_dataset_root=args.yolo_dataset,
            output_root=args.restored_dataset
        )
    else:
        print("\n⚠️  Skipping restoration step, using existing dataset")
    
    # Step 2: Train YOLOv8 on restored images
    data_yaml = Path(args.restored_dataset) / "data.yaml"
    
    model, results = train_yolo(
        data_yaml=str(data_yaml),
        epochs=args.epochs,
        batch_size=args.batch_size,
        img_size=args.img_size,
        model_name=args.model,
        resume=args.resume,
        gpus=args.gpus  # Pass GPU configuration
    )
    
    print(f"\n✓ YOLOv8 model trained successfully!")
    print(f"  Model saved to: yolo_training/runs/wunet_yolo/weights/best.pt")


if __name__ == '__main__':
    main()
