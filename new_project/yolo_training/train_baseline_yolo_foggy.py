#!/usr/bin/env python3
"""
Train Baseline YOLO on Raw Foggy Images (NO WUNET Restoration) - FIXED
"""

from ultralytics import YOLO
import yaml
from pathlib import Path
import shutil

print("="*70)
print("Baseline YOLO Training - Raw Foggy Images (NO Restoration)")
print("="*70)

# Use the FOGGY dataset directly with existing labels
# Point data.yaml to foggy images instead of restored images

baseline_yolo_dir = Path('/home/as19197/new_project/yolo_training')
foggy_dataset_yaml = baseline_yolo_dir / 'baseline_foggy_data.yaml'

# Create data.yaml pointing to foggy images
print("\n[1] Creating baseline dataset config...")

# Copy images to a clean location
baseline_img_dir = Path('/mnt/Data/as19197/new_project/yolo_baseline_foggy_beta0.02')
baseline_img_dir.mkdir(exist_ok=True, parents=True)

for split in ['train', 'val', 'test']:
    # Create image directory
    img_dir = baseline_img_dir / 'images' / split
    img_dir.mkdir(exist_ok=True, parents=True)
    
    # Copy foggy images
    src_dir = Path(f'/home/as19197/new_project/Foggy_Cityscapes_dataset/leftImg8bit_foggy/{split}')
    count = 0
    for city_dir in src_dir.iterdir():
        if city_dir.is_dir():
            for img in city_dir.glob('*_beta_0.02.png'):
                dst = img_dir / img.name
                if not dst.exists():
                    shutil.copy2(img, dst)
                    count += 1
    print(f"  ✓ Copied {count} {split} foggy images")
    
    # Copy labels (same as WUNET dataset)
    label_dir = baseline_img_dir / 'labels' / split
    if not label_dir.exists():
        src_labels = Path(f'/mnt/Data/as19197/new_project/yolo_dataset_beta0.02/labels/{split}')
        shutil.copytree(src_labels, label_dir)
        print(f"  ✓ Copied {split} labels")

# Create data.yaml
data_config = {
    'path': str(baseline_img_dir),
    'train': 'images/train',
    'val': 'images/val',
    'test': 'images/test',
    'nc': 2,
    'names': ['person', 'rider']
}

with open(foggy_dataset_yaml, 'w') as f:
    yaml.dump(data_config, f)

print(f"✓ Baseline dataset ready: {baseline_img_dir}")
print(f"✓ Config saved: {foggy_dataset_yaml}")

# Train YOLO on raw foggy images
print("\n[2] Training baseline YOLO on foggy images (100 epochs)...")
model = YOLO('yolov8n.pt')

results = model.train(
    data=str(foggy_dataset_yaml),
    epochs=100,
    batch=16,
    imgsz=640,
    device='0,1,2',
    project='/home/as19197/new_project/yolo_training/runs',
    name='baseline_foggy_beta0.02',
    exist_ok=True,
    verbose=True,
    patience=20,
    save=True,
    plots=True
)

print("\n" + "="*70)
print("✓ Baseline YOLO Training Complete!")
print(f"Best mAP@50: {results.results_dict.get('metrics/mAP50(B)', 'N/A')}")
print("="*70)
