#!/usr/bin/env python3
"""
Test Foggy-Trained Models on Clear Images
Compare how models trained on foggy/restored images perform on clear images
"""

from ultralytics import YOLO
from pathlib import Path
import cv2
import json
import numpy as np

print("="*80)
print("TESTING FOGGY-TRAINED MODELS ON CLEAR IMAGES")
print("="*80)

# Model configurations
MODELS = {
    'baseline_yolo_foggy': {
        'path': '/mnt/Data/as19197/new_project/yolo_training/runs/baseline_foggy_beta0.02/weights/best.pt',
        'description': 'YOLO trained on raw foggy images (β=0.02)',
        'output_dir': 'foggy_yolo_on_clear'
    },
    'wunet_yolo': {
        'path': '/mnt/Data/as19197/new_project/yolo_training/runs/wunet_yolo_beta0.02/weights/best.pt',
        'description': 'YOLO trained on WUNET-restored images (β=0.02)',
        'output_dir': 'wunet_yolo_on_clear'
    }
}

# Paths
CLEAR_DIR = Path('/mnt/Data/as19197/new_project/Foggy_Cityscapes_dataset/leftImg8bit/val')
BASE_OUTPUT_DIR = Path('/mnt/Data/as19197/new_project/clear_image_comparison')
BASE_OUTPUT_DIR.mkdir(exist_ok=True)

# Get the same clear images used in baseline
print(f"\n[1] Loading clear validation images...")
clear_images = sorted(list(CLEAR_DIR.rglob('*_leftImg8bit.png')))[:10]
print(f"✓ Selected {len(clear_images)} clear images")

# Store all results
all_results = {}

# Test each model
for model_name, config in MODELS.items():
    print(f"\n{'='*80}")
    print(f"Testing: {model_name}")
    print(f"Description: {config['description']}")
    print(f"{'='*80}")
    
    # Create output directory
    output_dir = BASE_OUTPUT_DIR / config['output_dir']
    output_dir.mkdir(exist_ok=True)
    
    # Load model
    print(f"\n[2] Loading model...")
    print(f"Path: {config['path']}")
    model = YOLO(config['path'])
    print(f"✓ Model loaded")
    
    # Run detection
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
        output_path = output_dir / f'{model_name}_clear_{idx:02d}.jpg'
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
    summary_path = output_dir / 'detection_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(detection_results, f, indent=2)
    
    # Calculate statistics
    total_dets = sum(r['num_detections'] for r in detection_results)
    avg_dets = total_dets / len(detection_results)
    
    all_results[model_name] = {
        'total_detections': total_dets,
        'average_per_image': avg_dets,
        'description': config['description']
    }
    
    print(f"\n✓ Results for {model_name}:")
    print(f"  Total detections: {total_dets}")
    print(f"  Average per image: {avg_dets:.1f}")

# Print comparison summary
print(f"\n{'='*80}")
print("COMPARISON SUMMARY")
print(f"{'='*80}")
print("\nAll models tested on the SAME 10 clear images:\n")

print(f"{'Model':<30} {'Avg Det/Image':<15} {'Total':<10}")
print("-" * 80)
for model_name, stats in all_results.items():
    print(f"{model_name:<30} {stats['average_per_image']:<15.1f} {stats['total_detections']:<10}")

# Save comparison summary
comparison_path = BASE_OUTPUT_DIR / 'comparison_summary.json'
with open(comparison_path, 'w') as f:
    json.dump(all_results, f, indent=2)

print(f"\n{'='*80}")
print(f"✓ All results saved to: {BASE_OUTPUT_DIR}")
print(f"✓ Comparison summary: {comparison_path}")
print(f"{'='*80}")
