#!/usr/bin/env python3
"""
Test DSNet+YOLO on Clear Images
Run the DSNet restoration + YOLO detection model on clear images
"""

import torch
import sys
sys.path.append('/mnt/Data/as19197/new_project/DSNET')
sys.path.append('/mnt/Data/as19197/new_project/WUNET')

from dsnet_yolo import DSNet_YOLO
from pathlib import Path
import cv2
import json
import numpy as np
from PIL import Image
import torchvision.transforms as transforms

print("="*80)
print("TESTING DSNet+YOLO ON CLEAR IMAGES")
print("="*80)

# Paths
MODEL_PATH = '/mnt/Data/as19197/new_project/checkpoints/dsnet_yolo_proper_loss_100ep/dsnet_yolo_best.pth'
CLEAR_DIR = Path('/mnt/Data/as19197/new_project/Foggy_Cityscapes_dataset/leftImg8bit/val')
OUTPUT_DIR = Path('/mnt/Data/as19197/new_project/clear_image_comparison/dsnet_yolo_on_clear')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\nDevice: {device}")

# Load model
print(f"\n[1] Loading DSNet+YOLO model...")
print(f"Path: {MODEL_PATH}")
model = DSNet_YOLO(num_classes=2, pretrained_cb=False)
checkpoint = torch.load(MODEL_PATH, map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model = model.to(device)
model.eval()
print(f"✓ Model loaded (Epoch {checkpoint.get('epoch', 'N/A')})")

# Image transform
transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor()
])

# Get the same clear images
print(f"\n[2] Loading clear validation images...")
clear_images = sorted(list(CLEAR_DIR.rglob('*_leftImg8bit.png')))[:10]
print(f"✓ Selected {len(clear_images)} clear images")

print(f"\n[3] Processing clear images with DSNet+YOLO...")
detection_results = []

for idx, img_path in enumerate(clear_images, 1):
    print(f"  [{idx}/10] Processing {img_path.name}")
    
    # Load and preprocess image
    img_pil = Image.open(img_path).convert('RGB')
    img_tensor = transform(img_pil).unsqueeze(0).to(device)
    
    # Run DSNet+YOLO
    with torch.no_grad():
        restored, detections = model(img_tensor)
    
    # Save restored image
    restored_np = restored[0].cpu().permute(1, 2, 0).numpy()
    restored_np = np.clip(restored_np * 255, 0, 255).astype(np.uint8)
    restored_np = cv2.cvtColor(restored_np, cv2.COLOR_RGB2BGR)
    
    # For detection visualization, we'll show the restored image
    # (DSNet+YOLO doesn't use standard YOLO format, so we'll just count detection activations)
    
    # Count strong detection activations (simple proxy for detections)
    det_count = 0
    for det_scale in detections:
        # Get confidence predictions (every 7th channel for 2 classes + 5 bbox params)
        det_np = det_scale[0].cpu().numpy()
        # Reshape to [num_anchors, num_outputs, H, W]
        B, C, H, W = det_scale.shape
        num_outputs = 7  # x,y,w,h,conf,cls1,cls2
        num_anchors = C // num_outputs
        # Count activations above threshold
        conf_channels = [i*num_outputs + 4 for i in range(num_anchors)]
        for ch in conf_channels:
            if ch < C:
                conf_map = det_np[ch]
                det_count += np.sum(conf_map > 0.5)
    
    # Save visualization (restored image)
    output_path = OUTPUT_DIR / f'dsnet_yolo_clear_{idx:02d}.jpg'
    cv2.imwrite(str(output_path), restored_np)
    
    detection_results.append({
        'image': img_path.name,
        'strong_activations': int(det_count),
        'note': 'DSNet+YOLO uses custom detection format - count represents strong feature activations'
    })
    
    print(f"    ✓ Saved: {output_path.name} ({det_count} strong activations)")

# Save detection summary
summary_path = OUTPUT_DIR / 'detection_summary.json'
with open(summary_path, 'w') as f:
    json.dump(detection_results, f, indent=2)

# Calculate statistics
total_acts = sum(r['strong_activations'] for r in detection_results)
avg_acts = total_acts / len(detection_results)

print(f"\n{'='*80}")
print(f"✓ DSNet+YOLO processing complete")
print(f"✓ Saved to: {OUTPUT_DIR}")
print(f"\nStatistics:")
print(f"  Total strong activations: {total_acts}")
print(f"  Average per image: {avg_acts:.1f}")
print(f"  Note: DSNet+YOLO detections are in custom format")
print(f"{'='*80}")
