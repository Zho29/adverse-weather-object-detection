#!/usr/bin/env python3
"""
Complete DSNet+DINO Detection Evaluation
- Restore images with DSNet
- Run DINO detection
- Calculate COCO mAP
"""

from pathlib import Path

import torch
import sys
sys.path.append(str((Path(__file__).resolve().parent / '.').resolve()))

from dsnet_dino import DSNet_DINO
import json
from tqdm import tqdm
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
import cv2
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

print("="*70)
print("DSNet + DINO Detection Evaluation")
print("="*70)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\nDevice: {device}")

# Paths
DINO_CONFIG = '/home/as19197/new_project/mmdetection/configs/dino/dino-4scale_r50_8xb2-12e_coco.py'
DINO_CHECKPOINT = '/home/as19197/new_project/mmdetection/configs/dino/dino-4scale_r50_8xb2-12e_coco_20221202_182705-55b2bba2.pth'
DSNET_CHECKPOINT = '/home/as19197/new_project/checkpoints/dsnet_dino_joint_100ep/dsnet_dino_joint_best.pth'
COCO_VAL = '/home/as19197/new_project/Foggy_Cityscapes_dataset/coco_annotations/cityscapes_val.json'

print("\n[1] Loading DSNet+DINO model...")
model = DSNet_DINO(
    dino_config=DINO_CONFIG,
    dino_checkpoint=DINO_CHECKPOINT,
    load_dino=True
)

checkpoint = torch.load(DSNET_CHECKPOINT, map_location='cpu')
model.load_state_dict(checkpoint['model_state_dict'], strict=False)
model = model.to(device)
model.eval()
print("✓ Model loaded")

# Load annotations
print("\n[2] Loading COCO annotations...")
coco_gt = COCO(COCO_VAL)
images = coco_gt.imgs
print(f"✓ Loaded {len(images)} validation images")

# Prepare transform
transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Run detection
print("\n[3] Running DSNet restoration + DINO detection...")
detection_results = []

# Process all validation images
from mmdet.apis import inference_detector

for img_id in tqdm(list(images.keys()), desc="Detection"):
    img_info = images[img_id]
    city = img_info['city']
    file_name = img_info['file_name']
    base_name = file_name.replace('_leftImg8bit.png', '')
    
    # Load foggy image
    foggy_path = Path(f'/home/as19197/new_project/Foggy_Cityscapes_dataset/leftImg8bit_foggy/val/{city}/{base_name}_leftImg8bit_foggy_beta_0.02.png')
    
    if not foggy_path.exists():
        continue
    
    # Load and transform
    foggy_img = Image.open(foggy_path).convert('RGB')
    orig_w, orig_h = foggy_img.size
    
    foggy_tensor = transform(foggy_img).unsqueeze(0).to(device)
    
    # DSNet restoration
    with torch.no_grad():
        restored_tensor = model(foggy_tensor)
    
    # Denormalize and convert to image
    restored_tensor = (restored_tensor + 1) / 2
    restored_tensor = torch.clamp(restored_tensor, 0, 1)
    restored_np = restored_tensor.squeeze(0).cpu().permute(1, 2, 0).numpy()
    restored_np = (restored_np * 255).astype(np.uint8)
    restored_np = cv2.cvtColor(restored_np, cv2.COLOR_RGB2BGR)
    
    # Resize back to original size for DINO
    restored_np = cv2.resize(restored_np, (orig_w, orig_h))
    
    # Run DINO detection
    result = inference_detector(model.dino if not hasattr(model, 'module') else model.module.dino, restored_np)
    
    # Convert DINO results to COCO format
    # DINO returns pred_instances with bboxes, scores, labels
    pred_instances = result.pred_instances
    
    for i in range(len(pred_instances.scores)):
        score = float(pred_instances.scores[i])
        if score < 0.3:  # Confidence threshold
            continue
        
        bbox = pred_instances.bboxes[i].cpu().numpy()
        label = int(pred_instances.labels[i])
        
        # DINO is trained on COCO (80 classes), map to our 2 classes
        # COCO person=0, map to our person=1
        # COCO car/truck/bus/motorcycle/bicycle -> our vehicle=2
        coco_to_cityscapes = {
            0: 1,  # person -> person
            1: 2,  # bicycle -> vehicle
            2: 2,  # car -> vehicle
            3: 2,  # motorcycle -> vehicle
            5: 2,  # bus -> vehicle
            7: 2,  # truck -> vehicle
        }
        
        if label not in coco_to_cityscapes:
            continue
        
        category_id = coco_to_cityscapes[label]
        
        # COCO format: [x, y, width, height]
        x1, y1, x2, y2 = bbox
        detection_results.append({
            'image_id': img_id,
            'category_id': category_id,
            'bbox': [float(x1), float(y1), float(x2-x1), float(y2-y1)],
            'score': score
        })

print(f"\n✓ Generated {len(detection_results)} detections")

# Save results
results_file = '/home/as19197/new_project/dsnet_dino_detection_results.json'
with open(results_file, 'w') as f:
    json.dump(detection_results, f)
print(f"✓ Saved results to {results_file}")

# Evaluate with COCO metrics
print("\n[4] Calculating COCO mAP...")
coco_dt = coco_gt.loadRes(detection_results)
coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
coco_eval.evaluate()
coco_eval.accumulate()
coco_eval.summarize()

# Extract key metrics
mAP = coco_eval.stats[0]  # mAP @ IoU=0.50:0.95
mAP50 = coco_eval.stats[1]  # mAP @ IoU=0.50
mAP75 = coco_eval.stats[2]  # mAP @ IoU=0.75

print("\n" + "="*70)
print("DSNet + DINO Detection Results")
print("="*70)
print(f"mAP@50-95: {mAP:.3f}")
print(f"mAP@50:    {mAP50:.3f}")
print(f"mAP@75:    {mAP75:.3f}")
print("="*70)

# Save metrics
metrics = {
    'mAP@50-95': float(mAP),
    'mAP@50': float(mAP50),
    'mAP@75': float(mAP75),
    'num_detections': len(detection_results),
    'num_images': len(images)
}

metrics_file = '/home/as19197/new_project/dsnet_dino_metrics.json'
with open(metrics_file, 'w') as f:
    json.dump(metrics, f, indent=2)

print(f"\n✓ Metrics saved to {metrics_file}")
