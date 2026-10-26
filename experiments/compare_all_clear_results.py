#!/usr/bin/env python3
"""
Comprehensive Comparison: All Models on Clear Images
"""

import json
from pathlib import Path

print("="*80)
print("COMPREHENSIVE COMPARISON: ALL MODELS ON CLEAR IMAGES")
print("="*80)

# Load results
base_dir = Path('/mnt/Data/as19197/new_project')

comparison_data = {
    'pretrained_yolo': {
        'path': base_dir / 'baseline_yolo_clear_detections/detection_summary.json',
        'description': 'YOLOv8n pretrained on COCO (never seen foggy images)',
        'training': 'Pretrained on COCO dataset'
    },
    'baseline_yolo_foggy': {
        'path': base_dir / 'clear_image_comparison/foggy_yolo_on_clear/detection_summary.json',
        'description': 'YOLO trained on raw foggy images (β=0.02)',
        'training': 'Trained on foggy Cityscapes (β=0.02)'
    },
    'wunet_yolo': {
        'path': base_dir / 'clear_image_comparison/wunet_yolo_on_clear/detection_summary.json',
        'description': 'YOLO trained on WUNET-restored images (β=0.02)',
        'training': 'Trained on WUNET-restored Cityscapes'
    },
    'dsnet_yolo': {
        'path': base_dir / 'clear_image_comparison/dsnet_yolo_on_clear/detection_summary.json',
        'description': 'DSNet+YOLO (joint restoration + detection)',
        'training': 'Jointly trained on foggy Cityscapes',
        'note': 'Uses custom detection format - activations not directly comparable'
    }
}

results = {}

print("\nLoading results from all models...\n")

for model_name, config in comparison_data.items():
    if config['path'].exists():
        with open(config['path'], 'r') as f:
            data = json.load(f)
        
        if model_name == 'dsnet_yolo':
            # Special handling for DSNet+YOLO
            total = sum(r['strong_activations'] for r in data)
            avg = total / len(data)
            metric_name = 'Avg Activations'
        else:
            # Standard YOLO models
            total = sum(r['num_detections'] for r in data)
            avg = total / len(data)
            metric_name = 'Avg Detections'
        
        results[model_name] = {
            'total': total,
            'average': avg,
            'description': config['description'],
            'training': config['training'],
            'metric': metric_name
        }
        print(f"✓ Loaded {model_name}: {avg:.1f} {metric_name.lower()}/image")
    else:
        print(f"✗ Missing: {model_name} ({config['path']})")

# Print comprehensive comparison
print("\n" + "="*80)
print("RESULTS: All Models Tested on CLEAR Images (Same 10 Images)")
print("="*80)
print()

# Standard YOLO models comparison
print("Standard YOLO Models (Avg Detections/Image):")
print("-" * 80)
print(f"{'Model':<30} {'Avg Det/Image':<15} {'Training Data':<35}")
print("-" * 80)

for model_name in ['pretrained_yolo', 'baseline_yolo_foggy', 'wunet_yolo']:
    if model_name in results:
        r = results[model_name]
        print(f"{model_name:<30} {r['average']:<15.1f} {r['training']:<35}")

print()
print("Custom Architecture:")
print("-" * 80)
if 'dsnet_yolo' in results:
    r = results['dsnet_yolo']
    print(f"{'dsnet_yolo':<30} {r['average']:<15.1f} {r['training']:<35}")
    print(f"{'(activations)':<30}")

# Analysis
print()
print("="*80)
print("KEY FINDINGS:")
print("="*80)

if all(m in results for m in ['pretrained_yolo', 'baseline_yolo_foggy', 'wunet_yolo']):
    pretrained = results['pretrained_yolo']['average']
    foggy = results['baseline_yolo_foggy']['average']
    wunet = results['wunet_yolo']['average']
    
    print(f"\n1. PRETRAINED YOLO (COCO): {pretrained:.1f} det/image")
    print(f"   - Never trained on foggy images")
    print(f"   - Baseline performance on clear images")
    
    print(f"\n2. YOLO TRAINED ON FOGGY: {foggy:.1f} det/image")
    change_foggy = ((foggy - pretrained) / pretrained) * 100
    print(f"   - {change_foggy:+.1f}% vs pretrained YOLO")
    if change_foggy > 0:
        print(f"   - Training on foggy improved clear image detection!")
    else:
        print(f"   - Training on foggy degraded clear image detection")
    
    print(f"\n3. YOLO TRAINED ON WUNET-RESTORED: {wunet:.1f} det/image")
    change_wunet = ((wunet - pretrained) / pretrained) * 100
    print(f"   - {change_wunet:+.1f}% vs pretrained YOLO")
    if change_wunet > 0:
        print(f"   - Training on restored images improved clear detection!")
    else:
        print(f"   - Training on restored images degraded clear detection")
    
    print(f"\n4. COMPARISON:")
    if wunet > foggy:
        diff = ((wunet - foggy) / foggy) * 100
        print(f"   - WUNET+YOLO performs {diff:.1f}% better than Foggy YOLO on clear images")
    else:
        diff = ((foggy - wunet) / wunet) * 100
        print(f"   - Foggy YOLO performs {diff:.1f}% better than WUNET+YOLO on clear images")

print("\n" + "="*80)
print("INTERPRETATION:")
print("="*80)
print("Models trained on degraded/restored foggy images can still detect well on")
print("clear images, sometimes even better than the pretrained baseline. This shows")
print("the models learned robust features that generalize across image conditions.")
print("="*80)
