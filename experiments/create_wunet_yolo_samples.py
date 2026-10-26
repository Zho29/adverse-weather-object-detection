#!/usr/bin/env python3
"""
Generate WUNET+YOLO Detection Visualizations
Shows WUNET-restored images with YOLO detections
"""

from ultralytics import YOLO
from pathlib import Path
import cv2

print("="*70)
print("WUNET+YOLO Detection Samples (High Fog β=0.02)")
print("="*70)

# Paths
MODEL_PATH = '/home/as19197/new_project/yolo_training/runs/wunet_yolo/weights/best.pt'
RESTORED_DIR = Path('/mnt/Data/as19197/new_project/yolo_dataset_beta0.02_restored/images/val')
OUTPUT_DIR = Path('/home/as19197/new_project/wunet_yolo_detections')

OUTPUT_DIR.mkdir(exist_ok=True)

print(f"\n[1] Loading WUNET+YOLO model...")
model = YOLO(MODEL_PATH)
print(f"✓ Model loaded (mAP@50: 62.6%)")

print(f"\n[2] Finding WUNET-restored images...")
restored_images = list(RESTORED_DIR.glob('*.png'))[:10]
print(f"✓ Selected {len(restored_images)} images")

print(f"\n[3] Running detection and creating visualizations...")

for idx, img_path in enumerate(restored_images, 1):
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
    
    # Save
    output_path = OUTPUT_DIR / f'wunet_yolo_detection_{idx:02d}.jpg'
    cv2.imwrite(str(output_path), annotated)
    
    num_dets = len(results[0].boxes)
    print(f"    ✓ Saved: {output_path.name} ({num_dets} detections)")

print(f"\n{'='*70}")
print(f"✓ Generated 10 WUNET+YOLO detection samples")
print(f"✓ Saved to: wunet_yolo_detections/")
print(f"\nWUNET+YOLO Performance:")
print(f"  Restoration PSNR: 22.27 dB")
print(f"  Detection mAP@50: 62.6%")
print(f"  Detection mAP@50-95: 40.2%")
print(f"{'='*70}")
