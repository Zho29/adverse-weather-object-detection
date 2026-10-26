"""
Visualize YOLO Detection Results

Creates visualization samples showing detections on:
1. Original foggy images
2. WUNET-restored images  
3. Detection results overlaid on restored images

This helps evaluate the effectiveness of the WUNET+YOLO pipeline.
"""

import torch
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import numpy as np
import sys

# Add paths
sys.path.append(str(Path(__file__).parent / '../WUNET'))
from weather_unet import WeatherUNet
import torchvision.transforms as transforms

try:
    from ultralytics import YOLO
except ImportError:
    print("Ultralytics not found, installing...")
    import subprocess
    subprocess.run(['pip', 'install', 'ultralytics', '-q'])
    from ultralytics import YOLO


class WUNETYOLOVisualizer:
    """Visualize WUNET+YOLO pipeline results"""
    
    def __init__(self, wunet_checkpoint, yolo_checkpoint, device='cuda'):
        self.device = device
        
        # Load WUNET
        self.wunet = WeatherUNet(n_channels=3, n_classes=3, bilinear=True)
        checkpoint = torch.load(wunet_checkpoint, map_location=device)
        state_dict = checkpoint['model_state_dict']
        if list(state_dict.keys())[0].startswith('module.'):
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        self.wunet.load_state_dict(state_dict)
        self.wunet = self.wunet.to(device)
        self.wunet.eval()
        
        # Load YOLO
        self.yolo = YOLO(yolo_checkpoint)
        
        # Transform for WUNET
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        
        print(f"✓ Loaded WUNET from: {wunet_checkpoint}")
        print(f"✓ Loaded YOLO from: {yolo_checkpoint}")
    
    def restore_image(self, img_pil):
        """Restore foggy image using WUNET"""
        # Preprocess
        img_tensor = self.transform(img_pil).unsqueeze(0).to(self.device)
        
        # Apply WUNET
        with torch.no_grad():
            restored_tensor = self.wunet(img_tensor)
            # Denormalize from [-1,1] to [0,1]
            restored_tensor = (restored_tensor + 1) / 2
            restored_tensor = torch.clamp(restored_tensor, 0, 1)
        
        # Convert back to PIL
        restored = transforms.ToPILImage()(restored_tensor.squeeze(0).cpu())
        return restored
    
    def detect_objects(self, img_pil):
        """Run YOLO detection"""
        results = self.yolo(img_pil, verbose=False)
        return results[0]
    
    def draw_detections(self, img_pil, results, conf_threshold=0.25):
        """Draw bounding boxes on image"""
        img_draw = img_pil.copy()
        draw = ImageDraw.Draw(img_draw)
        
        # Class names
        class_names = {0: 'person', 1: 'vehicle'}
        colors = {0: (255, 0, 0), 1: (0, 255, 0)}  # Red for person, green for vehicle
        
        boxes = results.boxes
        for box in boxes:
            conf = box.conf.item()
            if conf < conf_threshold:
                continue
            
            cls = int(box.cls.item())
            xyxy = box.xyxy[0].cpu().numpy()
            
            # Draw box
            draw.rectangle(xyxy.tolist(), outline=colors.get(cls, (255, 255, 0)), width=3)
            
            # Draw label
            label = f"{class_names.get(cls, 'unknown')} {conf:.2f}"
            draw.text((xyxy[0], xyxy[1]-15), label, fill=colors.get(cls, (255, 255, 0)))
        
        return img_draw
    
    def visualize_sample(self, foggy_img_path, save_path):
        """Create visualization for one image"""
        # Load foggy image
        foggy = Image.open(foggy_img_path).convert('RGB')
        
        # Restore with WUNET
        restored = self.restore_image(foggy)
        
        # Run detection on foggy
        results_foggy = self.detect_objects(foggy)
        foggy_detected = self.draw_detections(foggy, results_foggy)
        
        # Run detection on restored
        results_restored = self.detect_objects(restored)
        restored_detected = self.draw_detections(restored, results_restored)
        
        # Create comparison grid
        fig, axes =plt.subplots(2, 2, figsize=(16, 12))
        
        axes[0, 0].imshow(foggy)
        axes[0, 0].set_title('Original Foggy Image', fontsize=14, fontweight='bold')
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(foggy_detected)
        num_detections_foggy = len([b for b in results_foggy.boxes if b.conf.item() >= 0.25])
        axes[0, 1].set_title(f'Detections on Foggy ({num_detections_foggy} objects)', fontsize=14, fontweight='bold')
        axes[0, 1].axis('off')
        
        axes[1, 0].imshow(restored)
        axes[1, 0].set_title('WUNET Restored Image', fontsize=14, fontweight='bold')
        axes[1, 0].axis('off')
        
        axes[1, 1].imshow(restored_detected)
        num_detections_restored = len([b for b in results_restored.boxes if b.conf.item() >= 0.25])
        axes[1, 1].set_title(f'Detections on Restored ({num_detections_restored} objects)', fontsize=14, fontweight='bold')
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Saved visualization: {save_path}")
        print(f"  Foggy detections: {num_detections_foggy}, Restored detections: {num_detections_restored}")
        
        return num_detections_foggy, num_detections_restored


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Visualize WUNET+YOLO pipeline')
    parser.add_argument('--wunet-checkpoint', type=str, required=True,
                        help='Path to WUNET checkpoint')
    parser.add_argument('--yolo-checkpoint', type=str, required=True,
                        help='Path to YOLO checkpoint')
    parser.add_argument('--foggy-dir', type=str,
                        default='Foggy_Cityscapes_dataset/leftImg8bit_foggy/val',
                        help='Directory with foggy validation images')
    parser.add_argument('--output-dir', type=str,
                        default='yolo_visualizations',
                        help='Output directory for visualizations')
    parser.add_argument('--num-samples', type=int, default=10,
                        help='Number of samples to visualize')
    parser.add_argument('--beta', type=float, default=0.01,
                        help='Fog density to visualize')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize visualizer
    viz = WUNETYOLOVisualizer(
        wunet_checkpoint=args.wunet_checkpoint,
        yolo_checkpoint=args.yolo_checkpoint,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    # Find foggy images
    foggy_dir = Path(args.foggy_dir)
    foggy_images = list(foggy_dir.rglob(f"*_beta_{args.beta}.png"))[:args.num_samples]
    
    print(f"\nCreating {len(foggy_images)} visualization samples...")
    print("=" * 70)
    
    total_foggy_detections = 0
    total_restored_detections = 0
    
    for i, img_path in enumerate(foggy_images, 1):
        save_path = output_dir / f"sample_{i:02d}_{img_path.stem}.png"
        print(f"\n[{i}/{len(foggy_images)}] Processing: {img_path.name}")
        
        try:
            num_foggy, num_restored = viz.visualize_sample(img_path, save_path)
            total_foggy_detections += num_foggy
            total_restored_detections += num_restored
        except Exception as e:
            print(f"  ✗ Error: {e}")
    
    print("\n" + "=" * 70)
    print("Visualization Complete!")
    print("=" * 70)
    print(f"\nSummary:")
    print(f"  Total samples: {len(foggy_images)}")
    print(f"  Foggy detections: {total_foggy_detections} (avg: {total_foggy_detections/len(foggy_images):.1f} per image)")
    print(f"  Restored detections: {total_restored_detections} (avg: {total_restored_detections/len(foggy_images):.1f} per image)")
    print(f"  Improvement: +{total_restored_detections - total_foggy_detections} detections ({((total_restored_detections/max(total_foggy_detections,1)-1)*100):.1f}%)")
    print(f"\n✓ Visualizations saved to: {output_dir}")


if __name__ == '__main__':
    main()
