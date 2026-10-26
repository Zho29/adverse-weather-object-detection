#!/usr/bin/env python3
"""
Run YOLO Baseline on Clear (Non-Foggy) Cityscapes Images
This establishes the baseline performance on clean images
"""

from ultralytics import YOLO
from pathlib import Path
import cv2
import numpy as np
import json

print("="*70)
print("YOLO Baseline on Clear Images")
print("="*70)

# Paths
MODEL_PATH = '/mnt/Data/as19197/new_project/yolov8n.pt'  # Pretrained YOLO model
CLEAR_DIR = Path('/mnt/Data/as19197/new_project/Foggy_Cityscapes_dataset/leftImg8bit/val')
OUTPUT_DIR = Path('/mnt/Data/as19197/new_project/baseline_yolo_clear_detections')
OUTPUT_DIR.mkdir(exist_ok=True)

print(f"\n[1] Loading YOLO baseline model...")
print(f"Model: {MODEL_PATH}")
model = YOLO(MODEL_PATH)
print(f"✓ Model loaded (YOLOv8n pretrained on COCO)")

print(f"\n[2] Finding clear validation images...")
clear_images = sorted(list(CLEAR_DIR.rglob('*_leftImg8bit.png')))[:10]
print(f"✓ Selected {len(clear_images)} clear images")

print(f"\n[3] Running detection on clear images...")

detection_results = []

for idx, img_path in enumerate(clear_images, 1):
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
    
    # Save visualization
    output_path = OUTPUT_DIR / f'clear_baseline_detection_{idx:02d}.jpg'
    cv2.imwrite(str(output_path), annotated)
    
    # Count detections by class
    boxes = results[0].boxes
    num_dets = len(boxes)
    
    # Get class counts
    class_counts = {}
    if num_dets > 0:
        classes = boxes.cls.cpu().numpy().astype(int)
        for cls in classes:
            cls_name = model.names[cls]
            class_counts[cls_name] = class_counts.get(cls_name, 0) + 1
    
    detection_results.append({
        'image': img_path.name,
        'num_detections': num_dets,
        'class_counts': class_counts
    })
    
    print(f"    ✓ Saved: {output_path.name} ({num_dets} detections)")

# Save detection summary
summary_path = OUTPUT_DIR / 'detection_summary.json'
with open(summary_path, 'w') as f:
    json.dump(detection_results, f, indent=2)

# Calculate statistics
total_dets = sum(r['num_detections'] for r in detection_results)
avg_dets = total_dets / len(detection_results)

print(f"\n{'='*70}")
print(f"✓ Completed YOLO baseline on clear images")
print(f"✓ Saved to: {OUTPUT_DIR}")
print(f"\nDetection Statistics:")
print(f"  Total detections: {total_dets}")
print(f"  Average per image: {avg_dets:.1f}")
print(f"  Summary saved to: {summary_path.name}")
print(f"{'='*70}")
