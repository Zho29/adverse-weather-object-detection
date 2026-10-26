#!/usr/bin/env python3
"""
Generate 10 Sample Detection Visualizations for Baseline YOLO
Shows detection boxes on foggy images
"""

from ultralytics import YOLO
from pathlib import Path
import cv2
import numpy as np

print("="*70)
print("Baseline YOLO Detection Samples (High Fog β=0.02)")
print("="*70)

# Paths
MODEL_PATH = '/home/as19197/new_project/yolo_training/runs/baseline_foggy_beta0.02/weights/best.pt'
FOGGY_DIR = Path('/mnt/Data/as19197/new_project/Foggy_Cityscapes_dataset/leftImg8bit_foggy/val')
OUTPUT_DIR = Path('/home/as19197/.gemini/antigravity/brain/521be7ca-380d-4b05-9a55-7a6ab4119914')

print(f"\n[1] Loading Baseline YOLO model...")
print(f"Model: {MODEL_PATH}")
model = YOLO(MODEL_PATH)
print(f"✓ Model loaded (mAP@50: 64.5%)")

print(f"\n[2] Finding foggy validation images...")
foggy_images = list(FOGGY_DIR.rglob('*_beta_0.02.png'))[:10]
print(f"✓ Selected {len(foggy_images)} images")

print(f"\n[3] Running detection and creating visualizations...")

for idx, img_path in enumerate(foggy_images, 1):
    print(f"  [{idx}/10] Processing {img_path.name}")
    
    # Run detection
    results = model.predict(
        source=str(img_path),
        conf=0.25,
        iou=0.45,
        imgsz=640,
        verbose=False
    )
    
    # Get annotated image
    annotated = results[0].plot()
    
    # Save
    output_path = OUTPUT_DIR / f'baseline_yolo_detection_{idx:02d}.jpg'
    cv2.imwrite(str(output_path), annotated)
    
    # Count detections
    num_dets = len(results[0].boxes)
    print(f"    ✓ Saved: {output_path.name} ({num_dets} detections)")

print(f"\n{'='*70}")
print(f"✓ Generated 10 detection samples")
print(f"✓ Saved to: {OUTPUT_DIR}")
print(f"\nBaseline YOLO Performance:")
print(f"  mAP@50: 64.5%")
print(f"  mAP@50-95: 41.7%")
print(f"  Precision: 78.9%")
print(f"  Recall: 53.0%")
print(f"{'='*70}")
