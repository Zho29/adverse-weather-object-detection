#!/usr/bin/env python3
"""
Calculate Proper mAP@0.5 - Fixed Label Matching
"""

from ultralytics import YOLO
from pathlib import Path
import yaml
import shutil
import json

print("="*80)
print("CALCULATING mAP@0.5 FOR ALL MODELS ON CLEAR IMAGES (FIXED)")
print("="*80)

# Use existing foggy dataset structure but run on ALL validation images
# The key insight: labels are the SAME for clear and foggy images (objects don't change location)

# We'll use the existing dataset and just validate on it
DATASET_PATH = Path('/mnt/Data/as19197/new_project/yolo_dataset_beta0.02_baseline')
DATA_YAML = DATASET_PATH.parent / 'yolo_training/baseline_foggy_data.yaml'

print(f"\n[1] Using existing dataset: {DATASET_PATH}")
print(f"    Data YAML: {DATA_YAML}")

# Check if data.yaml exists and points to correct paths
if not DATA_YAML.exists():
    print("   Creating data.yaml...")
    data_config = {
        'path': str(DATASET_PATH),
        'train': 'images/train',
        'val': 'images/val',
        'nc': 2,
        'names': ['person', 'rider']
    }
    with open(DATA_YAML, 'w') as f:
        yaml.dump(data_config, f)
    print(f"   ✓ Created: {DATA_YAML}")

# Models to evaluate
models = {
    'pretrained_yolo': {
        'path': '/mnt/Data/as19197/new_project/yolov8n.pt',
        'description': 'YOLOv8n pretrained on COCO (2 classes: person, rider)'
    }, 
    'baseline_yolo_foggy': {
        'path': '/mnt/Data/as19197/new_project/yolo_training/runs/baseline_foggy_beta0.02/weights/best.pt',
        'description': 'YOLO trained on foggy images (β=0.02)'
    },
    'wunet_yolo': {
        'path': '/mnt/Data/as19197/new_project/yolo_training/runs/wunet_yolo_beta0.02/weights/best.pt',
        'description': 'YOLO trained on WUNET-restored images'
    }
}

results_summary = {}

# Evaluate each model
for model_name, config in models.items():
    print(f"\n{'='*80}")
    print(f"Evaluating: {model_name}")
    print(f"Description: {config['description']}")
    print(f"{'='*80}")
    
    print(f"\n[2] Loading model: {Path(config['path']).name}")
    model = YOLO(config['path'])
    
    print(f"\n[3] Running validation...")
    print(f"    Dataset: {DATA_YAML}")
    print(f"    Note: Evaluating on foggy images (validation labels)")
    
    results = model.val(
        data=str(DATA_YAML),
        batch=16,
        imgsz=640,
        conf=0.001,  # Very low confidence to capture all predictions
        iou=0.6,
        device='0',
        verbose=False,
        plots=False
    )
    
    # Extract metrics
    map50 = results.results_dict.get('metrics/mAP50(B)', 0.0)
    map50_95 = results.results_dict.get('metrics/mAP50-95(B)', 0.0)
    precision = results.results_dict.get('metrics/precision(B)', 0.0)
    recall = results.results_dict.get('metrics/recall(B)', 0.0)
    
    results_summary[model_name] = {
        'mAP@50': float(map50),
        'mAP@50-95': float(map50_95),
        'precision': float(precision),
        'recall': float(recall),
        'description': config['description']
    }
    
    print(f"\n✓ Results:")
    print(f"   mAP@50:    {map50:.4f}")
    print(f"   mAP@50-95: {map50_95:.4f}")
    print(f"   Precision: {precision:.4f}")
    print(f"   Recall:    {recall:.4f}")

# Save results
results_path = Path('/mnt/Data/as19197/new_project/foggy_image_map_results.json')
with open(results_path, 'w') as f:
    json.dump(results_summary, f, indent=2)

# Print final comparison
print(f"\n{'='*80}")
print("FINAL COMPARISON - mAP@0.5 ON FOGGY VALIDATION SET (β=0.02)")
print(f"{'='*80}\n")

print(f"{'Model':<30} {'mAP@50':<12} {'mAP@50-95':<12} {'Precision':<12} {'Recall':<12}")
print("-" * 80)
for model_name, metrics in results_summary.items():
    print(f"{model_name:<30} {metrics['mAP@50']:<12.4f} {metrics['mAP@50-95']:<12.4f} "
          f"{metrics['precision']:<12.4f} {metrics['recall']:<12.4f}")

print(f"\n{'='*80}")
print(f"✓ Results saved to: {results_path}")
print(f"NOTE: This is performance on FOGGY images, not clear")
print(f"      For clear image baseline, need ground truth for clear val set")
print(f"{'='*80}")
