#!/usr/bin/env python3
"""
Evaluate DINO Detection with CB Features via Adapter (FIXED)
Now properly feeds CB features to DINO instead of restored images!
"""

import torch
import sys
sys.path.append('/home/as19197/new_project/DSNET')

from dsnet_dino_adapter import DSNet_DINO_WithAdapter
import json
from pathlib import Path
from tqdm import tqdm
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

print("="*70)
print("DINO Detection with CB Features via Adapter (FIXED)")
print("="*70)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\nDevice: {device}")

# Paths
DINO_CONFIG = '/home/as19197/new_project/mmdetection/configs/dino/dino-4scale_r50_8xb2-12e_coco.py'
DINO_CHECKPOINT = '/home/as19197/new_project/mmdetection/configs/dino/dino-4scale_r50_8xb2-12e_coco_20221202_182705-55b2bba2.pth'
DSNET_CHECKPOINT = '/home/as19197/new_project/checkpoints/dsnet_dino_adapter_5ep/dsnet_dino_adapter_best.pth'
COCO_VAL = '/home/as19197/new_project/Foggy_Cityscapes_dataset/coco_annotations/cityscapes_val.json'

print("\n[1] Loading DSNet+DINO with Feature Adapter...")
model = DSNet_DINO_WithAdapter(
    dino_config=DINO_CONFIG,
    dino_checkpoint=DINO_CHECKPOINT,
    load_dino=True
)

# Load trained weights
checkpoint = torch.load(DSNET_CHECKPOINT, map_location='cpu')
model.load_state_dict(checkpoint['model_state_dict'], strict=False)
model = model.to(device)
model.eval()
print(f"✓ Model loaded (trained for {checkpoint.get('epoch', '?')} epochs)")

# Load COCO annotations
print("\n[2] Loading validation annotations...")
coco_gt = COCO(COCO_VAL)
images = coco_gt.imgs
print(f"✓ Loaded {len(images)} validation images")

transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Run detection using CB features!
print("\n[3] Running DINO detection with CB features (TRUE feature sharing)...")
detection_results = []

for img_id in tqdm(list(images.keys()), desc="Detection"):
    img_info = images[img_id]
    city = img_info['city']
    file_name = img_info['file_name']
    base_name = file_name.replace('_leftImg8bit.png', '')
    
    # Load foggy image
    foggy_path = Path(f'/home/as19197/new_project/Foggy_Cityscapes_dataset/leftImg8bit_foggy/val/{city}/{base_name}_leftImg8bit_foggy_beta_0.02.png')
    
    if not foggy_path.exists():
        continue
    
    foggy_img = Image.open(foggy_path).convert('RGB')
    orig_w, orig_h = foggy_img.size
    
    foggy_tensor = transform(foggy_img).unsqueeze(0).to(device)
    
    # KEY FIX: Use CB features via adapter for detection!
    with torch.no_grad():
        try:
            # This now properly uses CB features -> adapter -> DINO neck/head
            results = model.detect_with_cb_features(
                foggy_tensor,
                img_metas=[{
                    'img_shape': (512, 512, 3),
                    'ori_shape': (orig_h, orig_w, 3),
                    'scale_factor': (orig_w/512, orig_h/512)
                }]
            )
            
            # Extract predictions from MMDet format
            if len(results) > 0:
                pred_instances = results[0]
                
                for i in range(len(pred_instances.scores)):
                    score = float(pred_instances.scores[i])
                    if score < 0.3:
                        continue
                    
                    bbox = pred_instances.bboxes[i].cpu().numpy()
                    label = int(pred_instances.labels[i])
                    
                    # Map COCO to Cityscapes
                    coco_to_cityscapes = {
                        0: 1,  # person
                        1: 2,  # bicycle -> vehicle
                        2: 2,  # car -> vehicle
                        3: 2,  # motorcycle -> vehicle
                        5: 2,  # bus -> vehicle
                        7: 2,  # truck -> vehicle
                    }
                    
                    if label not in coco_to_cityscapes:
                        continue
                    
                    category_id = coco_to_cityscapes[label]
                    x1, y1, x2, y2 = bbox
                    
                    # Scale back to original size
                    x1 = x1 * (orig_w / 512)
                    y1 = y1 * (orig_h / 512)
                    x2 = x2 * (orig_w / 512)
                    y2 = y2 * (orig_h / 512)
                    
                    detection_results.append({
                        'image_id': img_id,
                        'category_id': category_id,
                        'bbox': [float(x1), float(y1), float(x2-x1), float(y2-y1)],
                        'score': score
                    })
        except Exception as e:
            print(f"Error processing image {img_id}: {e}")
            continue

print(f"\n✓ Generated {len(detection_results)} detections")

if len(detection_results) == 0:
    print("\n⚠ No detections generated! Check if detect_with_cb_features is working correctly.")
    sys.exit(1)

# Save results
results_file = '/home/as19197/new_project/dsnet_adapter_cb_features_results.json'
with open(results_file, 'w') as f:
    json.dump(detection_results, f)
print(f"✓ Saved to {results_file}")

# Evaluate
print("\n[4] Calculating COCO mAP...")
coco_dt = coco_gt.loadRes(detection_results)
coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
coco_eval.evaluate()
coco_eval.accumulate()
coco_eval.summarize()

# Extract metrics
mAP = coco_eval.stats[0]
mAP50 = coco_eval.stats[1]
mAP75 = coco_eval.stats[2]

print("\n" + "="*70)
print("DSNet+Adapter + DINO with CB Features (FIXED)")
print("="*70)
print(f"mAP@50-95: {mAP:.3f}")
print(f"mAP@50:    {mAP50:.3f}")
print(f"mAP@75:    {mAP75:.3f}")
print("\nComparison:")
print(f"  Previous (restored image):   28.2% mAP@50")
print(f"  Adapter (still used image):  22.9% mAP@50")
print(f"  FIXED (CB features):         {mAP50*100:.1f}% mAP@50")
print("="*70)

# Save metrics
metrics = {
    'mAP@50-95': float(mAP),
    'mAP@50': float(mAP50),
    'mAP@75': float(mAP75),
    'num_detections': len(detection_results),
    'method': 'cb_features_via_adapter'
}

with open('/home/as19197/new_project/dsnet_adapter_cb_metrics.json', 'w') as f:
    json.dump(metrics, f, indent=2)

print(f"\n✓ Metrics saved")
