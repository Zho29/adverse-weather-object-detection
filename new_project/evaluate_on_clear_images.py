#!/usr/bin/env python3
"""
Evaluate All Models on CLEAR Images - Proper Baseline
Ground truth labels are the same for clear and foggy (synthetic fog added to clear)
"""

from ultralytics import YOLO
from pathlib import Path
import yaml
import shutil
import json

print("="*80)
print("BASELINE EVALUATION: mAP@0.5 ON CLEAR IMAGES")
print("="*80)

# Create clear image dataset with proper labels
CLEAR_EVAL_DIR = Path('/mnt/Data/as19197/new_project/clear_image_baseline_dataset')
CLEAR_EVAL_DIR.mkdir(exist_ok=True)

# Paths
CLEAR_IMG_DIR = Path('/mnt/Data/as19197/new_project/Foggy_Cityscapes_dataset/leftImg8bit/val')
FOGGY_LABELS = Path('/mnt/Data/as19197/new_project/yolo_dataset_beta0.02_baseline/labels/val')

print("\n[1] Setting up clear image evaluation dataset...")

# Create images/val directory
val_img_dir = CLEAR_EVAL_DIR / 'images' / 'val'
val_img_dir.mkdir(parents=True, exist_ok=True)

# Create labels/val directory  
val_label_dir = CLEAR_EVAL_DIR / 'labels' / 'val'
val_label_dir.mkdir(parents=True, exist_ok=True)

# Copy clear images
print("   Copying clear validation images...")
image_count = 0
for city_dir in CLEAR_IMG_DIR.iterdir():
    if city_dir.is_dir():
        for img_path in city_dir.glob('*_leftImg8bit.png'):
            dst_img = val_img_dir / img_path.name
            if not dst_img.exists():
                shutil.copy2(img_path, dst_img)
                image_count += 1

print(f"   ✓ Copied {image_count} clear images")

# Copy corresponding labels (same as foggy - objects don't move)
print("   Copying ground truth labels...")
label_count = 0
for label_path in FOGGY_LABELS.glob('*.txt'):
    # Map foggy label name to clear image name
    # frankfurt_000000_000294_frankfurt_beta0.01.txt -> frankfurt_000000_000294_leftImg8bit.txt
    parts = label_path.stem.split('_')
    if len(parts) >= 3:
        # Get base: city_number_number
        base = '_'.join(parts[:3])
        clear_label_name = f"{base}_leftImg8bit.txt"
        
        dst_label = val_label_dir / clear_label_name
        if not dst_label.exists():
            shutil.copy2(label_path, dst_label)
            label_count += 1

print(f"   ✓ Copied {label_count} labels")

# Create data.yaml
data_config = {
    'path': str(CLEAR_EVAL_DIR),
    'train': 'images/val',  # Not used
    'val': 'images/val',
    'nc': 2,
    'names': ['person', 'rider']
}

data_yaml_path = CLEAR_EVAL_DIR / 'data.yaml'
with open(data_yaml_path, 'w') as f:
    yaml.dump(data_config, f)

print(f"   ✓ Created data.yaml: {data_yaml_path}")

# Models to evaluate
models = {
    'pretrained_yolo': {
        'path': '/mnt/Data/as19197/new_project/yolov8n.pt',
        'description': 'YOLOv8n pretrained on COCO (never seen fog)'
    },
    'foggy_yolo': {
        'path': '/mnt/Data/as19197/new_project/yolo_training/runs/baseline_foggy_beta0.02/weights/best.pt',
        'description': 'YOLO trained on foggy images (β=0.02)'
    },
    'wunet_yolo': {
        'path': '/mnt/Data/as19197/new_project/yolo_training/runs/wunet_yolo_beta0.02/weights/best.pt',
        'description': 'YOLO trained on WUNET-restored images'
    }
}

results_summary = {}

# Evaluate each model on CLEAR images
for model_name, config in models.items():
    print(f"\n{'='*80}")
    print(f"Evaluating: {model_name}")
    print(f"Description: {config['description']}")
    print(f"{'='*80}")
    
    print(f"\n[2] Loading model: {Path(config['path']).name}")
    model = YOLO(config['path'])
    
    print(f"\n[3] Running validation on CLEAR images...")
    results = model.val(
        data=str(data_yaml_path),
        batch=16,
        imgsz=640,
        conf=0.001,
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
    
    print(f"\n✓ Results on CLEAR images:")
    print(f"   mAP@50:    {map50:.4f}")
    print(f"   mAP@50-95: {map50_95:.4f}")
    print(f"   Precision: {precision:.4f}")
    print(f"   Recall:    {recall:.4f}")

# Save results
results_path = Path('/mnt/Data/as19197/new_project/CLEAR_IMAGE_mAP_BASELINE.json')
with open(results_path, 'w') as f:
    json.dump(results_summary, f, indent=2)

# Print final comparison
print(f"\n{'='*80}")
print("BASELINE: mAP@0.5 ON CLEAR IMAGES (Upper Bound Performance)")
print(f"{'='*80}\n")

print(f"{'Model':<30} {'mAP@50':<12} {'mAP@50-95':<12} {'Precision':<12} {'Recall':<12}")
print("-" * 80)
for model_name, metrics in results_summary.items():
    print(f"{model_name:<30} {metrics['mAP@50']:<12.4f} {metrics['mAP@50-95']:<12.4f} "
          f"{metrics['precision']:<12.4f} {metrics['recall']:<12.4f}")

print(f"\n{'='*80}")
print(f"✓ CLEAR IMAGE BASELINE saved to: {results_path}")
print(f"This represents the UPPER BOUND - performance without fog degradation")
print(f"{'='*80}")
