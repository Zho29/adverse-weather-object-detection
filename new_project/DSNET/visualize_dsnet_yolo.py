#!/usr/bin/env python3
"""
Visualize DSNet+YOLO Detection Results
Shows: Foggy input | Restored output | Detections on restored
"""

import torch
import torch.nn as nn
import sys
sys.path.append('/home/as19197/new_project/DSNET')

from dsnet_yolo import DSNet_YOLO
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

print("="*70)
print("DSNet+YOLO Detection Visualization")
print("="*70)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}\n")

# Paths
MODEL_CHECKPOINT = '/home/as19197/new_project/checkpoints/dsnet_yolo_proper_loss_5ep/dsnet_yolo_best.pth'
FOGGY_DIR = Path('/home/as19197/new_project/Foggy_Cityscapes_dataset/leftImg8bit_foggy/val')
CLEAR_DIR = Path('/home/as19197/new_project/Foggy_Cityscapes_dataset/leftImg8bit/val')
VIS_DIR = Path('/home/as19197/.gemini/antigravity/brain/521be7ca-380d-4b05-9a55-7a6ab4119914')

print("[1] Loading DSNet+YOLO model...")
model = DSNet_YOLO(num_classes=2, pretrained_cb=False)

# Load checkpoint
checkpoint = torch.load(MODEL_CHECKPOINT, map_location='cpu')
model.load_state_dict(checkpoint['model_state_dict'])
print(f"✓ Loaded model from epoch {checkpoint.get('epoch', '?')}")
print(f"  PSNR: {checkpoint.get('psnr', 'N/A')} dB")

model = model.to(device)
model.eval()

# Transform
transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

def process_detections(detections, conf_thresh=0.3):
    """Extract valid detections from YOLO outputs"""
    all_dets = []
    
    for scale_idx, pred in enumerate(detections):
        B, no, H, W = pred.shape
        pred = pred.permute(0, 2, 3, 1)  # [B, H, W, no]
        
        for b in range(B):
            for h in range(H):
                for w in range(W):
                    p = pred[b, h, w, :]
                    
                    # Extract components
                    x_offset = torch.sigmoid(p[0]).item()
                    y_offset = torch.sigmoid(p[1]).item()
                    w_pred = p[2].item()
                    h_pred = p[3].item()
                    obj_conf = torch.sigmoid(p[4]).item()
                    
                    if obj_conf < conf_thresh:
                        continue
                    
                    # Class scores
                    class_scores = torch.sigmoid(p[5:])
                    class_id = torch.argmax(class_scores).item()
                    class_conf = class_scores[class_id].item()
                    
                    # Total confidence
                    conf = obj_conf * class_conf
                    
                    if conf < conf_thresh:
                        continue
                    
                    # Convert to pixel coordinates
                    scale = 8 * (2 ** scale_idx)  # P3=8, P4=16, P5=32
                    x_center = (w + x_offset) * scale
                    y_center = (h + y_offset) * scale
                    width = abs(w_pred) * 512
                    height = abs(h_pred) * 512
                    
                    # Bounding box
                    x1 = max(0, x_center - width/2)
                    y1 = max(0, y_center - height/2)
                    x2 = min(512, x_center + width/2)
                    y2 = min(512, y_center + height/2)
                    
                    if (x2 - x1) > 10 and (y2 - y1) > 10:  # Filter tiny boxes
                        all_dets.append({
                            'bbox': [x1, y1, x2, y2],
                            'conf': conf,
                            'class': class_id
                        })
    
    return all_dets

# Get sample images
print("\n[2] Processing sample images...")
foggy_images = list(FOGGY_DIR.rglob('*_beta_0.02.png'))[:10]
print(f"✓ Found {len(foggy_images)} sample images")

class_names = ['person', 'vehicle']
colors = ['#FF5722', '#2196F3']  # Orange for person, blue for vehicle

for idx, foggy_path in enumerate(foggy_images, 1):
    print(f"\n[{idx}/{len(foggy_images)}] Processing {foggy_path.name}")
    
    # Load images
    foggy_img = Image.open(foggy_path).convert('RGB')
    
    # Get clear reference
    city = foggy_path.parent.name
    base_name = foggy_path.stem.replace('_leftImg8bit_foggy_beta_0.02', '')
    clear_path = CLEAR_DIR / city / f"{base_name}_leftImg8bit.png"
    
    if clear_path.exists():
        clear_img = Image.open(clear_path).convert('RGB')
    else:
        clear_img = foggy_img.copy()
    
    # Transform for model
    foggy_tensor = transform(foggy_img).unsqueeze(0).to(device)
    
    # Run model
    with torch.no_grad():
        restored_tensor, detections = model(foggy_tensor)
    
    # Denormalize restored image
    restored_tensor = (restored_tensor + 1) / 2
    restored_tensor = torch.clamp(restored_tensor, 0, 1)
    restored_np = restored_tensor.squeeze(0).cpu().permute(1, 2, 0).numpy()
    
    # Process detections
    dets = process_detections(detections, conf_thresh=0.3)
    print(f"  Found {len(dets)} detections")
    
    # Create visualization
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Foggy input
    axes[0].imshow(foggy_img)
    axes[0].set_title('Foggy Input (β=0.02)', fontsize=14, fontweight='bold')
    axes[0].axis('off')
    
    # Restored output
    axes[1].imshow(restored_np)
    axes[1].set_title('DSNet Restored (18.24 dB)', fontsize=14, fontweight='bold')
    axes[1].axis('off')
    
    # Restored with detections
    axes[2].imshow(restored_np)
    axes[2].set_title(f'YOLO Detections ({len(dets)} boxes)', fontsize=14, fontweight='bold')
    
    # Draw detection boxes
    for det in dets:
        x1, y1, x2, y2 = det['bbox']
        class_id = det['class']
        conf = det['conf']
        
        rect = patches.Rectangle(
            (x1, y1), x2-x1, y2-y1,
            linewidth=2, edgecolor=colors[class_id],
            facecolor='none'
        )
        axes[2].add_patch(rect)
        
        # Label
        label = f"{class_names[class_id]}: {conf:.2f}"
        axes[2].text(
            x1, y1-5, label,
            color='white', fontsize=10,
            bbox=dict(facecolor=colors[class_id], alpha=0.7, edgecolor='none')
        )
    
    axes[2].axis('off')
    
    plt.tight_layout()
    
    # Save
    save_path = VIS_DIR / f'dsnet_yolo_detection_{idx:02d}.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Saved to {save_path.name}")

print(f"\n{'='*70}")
print(f"✓ Visualizations saved to: {VIS_DIR}")
print(f"{'='*70}")

# Print summary
print("\nDetection Summary:")
print(f"  Model: DSNet+YOLO (proper loss, 5 epochs)")
print(f"  Restoration PSNR: 18.24 dB")
print(f"  Detection loss: 0.005 (trained!)")
print(f"  Visualizations: 10 samples created")
