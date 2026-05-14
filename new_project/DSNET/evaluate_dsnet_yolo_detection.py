#!/usr/bin/env python3
"""
Evaluate DSNet+YOLO Detection Performance
Measures mAP of YOLO head learning from CB features
"""

import torch
import torch.nn as nn
import sys
sys.path.append('/home/as19197/new_project/DSNET')

from dsnet_yolo import DSNet_YOLO
import json
from pathlib import Path
from tqdm import tqdm
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

print("="*70)
print("DSNet+YOLO Detection Evaluation")
print("="*70)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\nDevice: {device}")

# Paths
MODEL_CHECKPOINT = '/home/as19197/new_project/checkpoints/dsnet_yolo_100ep/dsnet_yolo_best.pth'
COCO_VAL = '/home/as19197/new_project/Foggy_Cityscapes_dataset/coco_annotations/cityscapes_val.json'
FOGGY_DIR = Path('/home/as19197/new_project/Foggy_Cityscapes_dataset/leftImg8bit_foggy/val')

print("\n[1] Loading DSNet+YOLO model...")
model = DSNet_YOLO(num_classes=2, pretrained_cb=False)

# Load trained weights
checkpoint = torch.load(MODEL_CHECKPOINT, map_location='cpu')
if 'model_state_dict' in checkpoint:
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"✓ Loaded model from epoch {checkpoint.get('epoch', '?')}")
    print(f"  PSNR: {checkpoint.get('psnr', 'N/A')} dB")
else:
    model.load_state_dict(checkpoint)
    print("✓ Loaded model weights")

model = model.to(device)

# Use all available GPUs
if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs for faster evaluation")
    model = nn.DataParallel(model)

model.eval()

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

# Helper function to process YOLO outputs
def process_yolo_outputs(detections, orig_h, orig_w, conf_thresh=0.3):
    """
    Convert YOLO outputs to COCO format detections
    
    Args:
        detections: List of [P3, P4, P5] outputs from YOLO head
        orig_h, orig_w: Original image dimensions
        conf_thresh: Confidence threshold
        
    Returns:
        List of detection dicts
    """
    results = []
    
    for scale_idx, pred in enumerate(detections):
        # pred shape: [B, no, H, W] where no = num_classes + 5
        # For num_classes=2: no = 7 (x, y, w, h, conf, cls0, cls1)
        B, no, H, W = pred.shape
        
        # Reshape to [B, H, W, no]
        pred = pred.permute(0, 2, 3, 1).contiguous()
        
        for b in range(B):
            for h in range(H):
                for w in range(W):
                    # Get prediction at this location
                    p = pred[b, h, w, :]
                    
                    # Extract components
                    x_center = p[0].item()
                    y_center = p[1].item()
                    width = p[2].item()
                    height = p[3].item()
                    objectness = torch.sigmoid(p[4]).item()
                    
                    if objectness < conf_thresh:
                        continue
                    
                    # Class scores
                    class_scores = torch.sigmoid(p[5:])
                    class_id = torch.argmax(class_scores).item()
                    class_conf = class_scores[class_id].item()
                    
                    # Overall confidence
                    conf = objectness * class_conf
                    
                    if conf < conf_thresh:
                        continue
                    
                    # Convert to pixel coordinates
                    # Scale based on feature map size
                    scale = 8 * (2 ** scale_idx)  # P3=8, P4=16, P5=32
                    x_center = (w + x_center) * scale
                    y_center = (h + y_center) * scale
                    width = width * 512  # Image size
                    height = height * 512
                    
                    # Convert to x1,y1,x2,y2
                    x1 = max(0, x_center - width/2)
                    y1 = max(0, y_center - height/2)
                    x2 = min(512, x_center + width/2)
                    y2 = min(512, y_center + height/2)
                    
                    # Scale to original image size
                    x1 = x1 * orig_w / 512
                    y1 = y1 * orig_h / 512
                    x2 = x2 * orig_w / 512
                    y2 = y2 * orig_h / 512
                    
                    results.append({
                        'bbox': [x1, y1, x2-x1, y2-y1],
                        'score': conf,
                        'category_id': class_id + 1  # +1 for COCO (1=person, 2=vehicle)
                    })
    
    return results

# Run detection
print("\n[3] Running YOLO detection with CB features...")
detection_results = []

for img_id in tqdm(list(images.keys()), desc="Detection"):
    img_info = images[img_id]
    city = img_info['city']
    file_name = img_info['file_name']
    base_name = file_name.replace('_leftImg8bit.png', '')
    
    # Load foggy image
    foggy_path = FOGGY_DIR / city / f"{base_name}_leftImg8bit_foggy_beta_0.02.png"
    
    if not foggy_path.exists():
        continue
    
    foggy_img = Image.open(foggy_path).convert('RGB')
    orig_w, orig_h = foggy_img.size
    
    foggy_tensor = transform(foggy_img).unsqueeze(0).to(device)
    
    # Forward pass - get detection outputs
    with torch.no_grad():
        if isinstance(model, nn.DataParallel):
            restored, detections = model.module(foggy_tensor)
        else:
            restored, detections = model(foggy_tensor)
    
    # Process YOLO outputs
    dets = process_yolo_outputs(detections, orig_h, orig_w, conf_thresh=0.3)
    
    for det in dets:
        detection_results.append({
            'image_id': img_id,
            'category_id': det['category_id'],
            'bbox': det['bbox'],
            'score': det['score']
        })

print(f"\n✓ Generated {len(detection_results)} detections")

if len(detection_results) == 0:
    print("\n⚠ No detections generated!")
    print("This could mean:")
    print("  1. Detection head not properly trained")
    print("  2. Confidence threshold too high")
    print("  3. YOLO output format incorrect")
    import sys
    sys.exit(1)

# Save results
results_file = '/home/as19197/new_project/dsnet_yolo_detection_results.json'
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
print("DSNet+YOLO Detection Results")
print("="*70)
print(f"mAP@50-95: {mAP:.3f}")
print(f"mAP@50:    {mAP50:.3f}")
print(f"mAP@75:    {mAP75:.3f}")
print("\nComparison:")
print(f"  Baseline YOLO (foggy): 64.5% mAP@50")
print(f"  WUNET + YOLO:          62.3% mAP@50")
print(f"  DSNet+DINO:            28.2% mAP@50")
print(f"  DSNet+YOLO:            {mAP50*100:.1f}% mAP@50")
print("="*70)

# Save metrics
metrics = {
    'mAP@50-95': float(mAP),
    'mAP@50': float(mAP50),
    'mAP@75': float(mAP75),
    'num_detections': len(detection_results),
    'restoration_psnr': checkpoint.get('psnr', 'N/A')
}

with open('/home/as19197/new_project/dsnet_yolo_metrics.json', 'w') as f:
    json.dump(metrics, f, indent=2)

print(f"\n✓ Metrics saved")
print(f"\nFinal Assessment:")
if mAP50 > 0.55:
    print("  ✅ SUCCESS - Joint learning effective!")
elif mAP50 > 0.40:
    print("  ⚠ MODERATE - Some joint learning benefit")
else:
    print("  ❌ POOR - Detection not learning from CB features")
