#!/usr/bin/env python3
"""
Create DSNet+DINO Detection Visualizations with Boxes
Shows restored images with actual DINO detection boxes
"""

import torch
import sys
sys.path.append('/home/as19197/new_project/DSNET')

from dsnet_dino import DSNet_DINO
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torchvision.transforms as transforms
from mmdet.apis import inference_detector

print("="*70)
print("DSNet+DINO Detection Visualizations")
print("="*70)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}\n")

# Paths
DINO_CONFIG = '/home/as19197/new_project/mmdetection/configs/dino/dino-4scale_r50_8xb2-12e_coco.py'
DINO_CHECKPOINT = '/home/as19197/new_project/mmdetection/configs/dino/dino-4scale_r50_8xb2-12e_coco_20221202_182705-55b2bba2.pth'
DSNET_CHECKPOINT = '/home/as19197/new_project/checkpoints/dsnet_dino_joint_100ep/dsnet_dino_joint_best.pth'
FOGGY_DIR = Path('/mnt/Data/as19197/new_project/Foggy_Cityscapes_dataset/leftImg8bit_foggy/val')
OUTPUT_DIR = Path('/home/as19197/new_project/dsnet_dino_detections')

OUTPUT_DIR.mkdir(exist_ok=True)

print("[1] Loading DSNet+DINO model...")
model = DSNet_DINO(
    dino_config=DINO_CONFIG,
    dino_checkpoint=DINO_CHECKPOINT,
    load_dino=True
)

checkpoint = torch.load(DSNET_CHECKPOINT, map_location='cpu')
model.load_state_dict(checkpoint['model_state_dict'], strict=False)
model = model.to(device)
model.eval()
print("✓ Model loaded (PSNR: 15.74 dB, mAP@50: 28.2%)")

# Transform
transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

print("\n[2] Finding validation images...")
foggy_images = list(FOGGY_DIR.rglob('*_beta_0.02.png'))[:10]
print(f"✓ Selected {len(foggy_images)} images")

print("\n[3] Generating visualizations...")

# COCO class names (DINO is pretrained on COCO)
COCO_CLASSES = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
                'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
                'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
                'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
                'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
                'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
                'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
                'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
                'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
                'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
                'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
                'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
                'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock',
                'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']

for idx, img_path in enumerate(foggy_images, 1):
    print(f"  [{idx}/10] Processing {img_path.name}")
    
    # Load image
    foggy_img = Image.open(img_path).convert('RGB')
    orig_w, orig_h = foggy_img.size
    
    # Transform
    foggy_tensor = transform(foggy_img).unsqueeze(0).to(device)
    
    # Run DSNet restoration
    with torch.no_grad():
        restored_tensor = model(foggy_tensor)
    
    # Denormalize
    restored_tensor = (restored_tensor + 1) / 2
    restored_tensor = torch.clamp(restored_tensor, 0, 1)
    restored_np = restored_tensor.squeeze(0).cpu().permute(1, 2, 0).numpy()
    
    # Convert to PIL
    restored_pil = Image.fromarray((restored_np * 255).astype(np.uint8))
    restored_pil = restored_pil.resize((orig_w, orig_h))
    
    # Run DINO detection
    result = inference_detector(model.dino, np.array(restored_pil))
    
    # Draw detections on restored image
    draw = ImageDraw.Draw(restored_pil)
    
    # Try to load a font
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14)
    except:
        font = ImageFont.load_default()
    
    num_dets = 0
    # Extract detections (result.pred_instances has bboxes, scores, labels)
    if hasattr(result, 'pred_instances'):
        pred = result.pred_instances
        bboxes = pred.bboxes.cpu().numpy()
        scores = pred.scores.cpu().numpy()
        labels = pred.labels.cpu().numpy()
        
        for bbox, score, label in zip(bboxes, scores, labels):
            if score > 0.3:  # confidence threshold
                class_name = COCO_CLASSES[label] if label < len(COCO_CLASSES) else f"class_{label}"
                
                # Filter to show person and car/vehicle classes
                if class_name not in ['person', 'car', 'bicycle', 'motorcycle', 'bus', 'truck']:
                    continue
                
                num_dets += 1
                x1, y1, x2, y2 = bbox
                
                # Color code: person=red, vehicles=blue
                color = 'red' if class_name == 'person' else 'blue'
                
                # Draw thicker box
                draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
                
                # Draw label with background
                text = f"{class_name}: {score:.2f}"
                bbox_text = draw.textbbox((x1, y1-20), text, font=font)
                draw.rectangle(bbox_text, fill=color)
                draw.text((x1, y1-20), text, fill='white', font=font)
    
    # Save
    output_path = OUTPUT_DIR / f'dsnet_dino_detection_{idx:02d}.jpg'
    restored_pil.save(output_path)
    
    print(f"    ✓ Saved: {output_path.name} ({num_dets} detections)")

print(f"\n{'='*70}")
print(f"✓ Generated 10 DSNet+DINO detection samples")
print(f"✓ Saved to: dsnet_dino_detections/")
print(f"\nDSNet+DINO Performance:")
print(f"  Restoration PSNR: 15.74 dB")
print(f"  Detection mAP@50: 28.2%")
print(f"\nNote: Fewer/lower quality detections than baseline (64.5%) or WUNET+YOLO (62.6%)")
print(f"{'='*70}")
