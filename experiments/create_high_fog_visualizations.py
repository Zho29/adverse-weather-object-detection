#!/usr/bin/env python3
"""
Create side-by-side visualizations for high fog (beta=0.02)
Shows: Foggy | WUNET Restored | Clear Reference
With YOLO detection boxes overlaid
"""

import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
import random

# Paths
YOLO_MODEL = "/mnt/Data/as19197/new_project/yolo_training/runs/wunet_yolo_beta0.02/weights/best.pt"
FOGGY_DIR = Path("/home/as19197/new_project/Foggy_Cityscapes_dataset/leftImg8bit_foggy")
RESTORED_DIR = Path("/mnt/Data/as19197/new_project/yolo_dataset_beta0.02_restored/images")
CLEAR_DIR = Path("/home/as19197/new_project/Foggy_Cityscapes_dataset/leftImg8bit")
OUTPUT_DIR = Path("/home/as19197/new_project/yolo_visualizations_high_fog")

OUTPUT_DIR.mkdir(exist_ok=True)

print("="*70)
print("Creating High Fog Detection Visualizations (Beta=0.02)")
print("="*70)

# Load YOLO model
print("\nLoading YOLO model...")
model = YOLO(YOLO_MODEL)
print("✓ Model loaded")

# Get sample validation images
val_foggy_imgs = list(FOGGY_DIR.rglob("val/*/*_beta_0.02.png"))
random.seed(42)
samples = random.sample(val_foggy_imgs, min(10, len(val_foggy_imgs)))

print(f"\nProcessing {len(samples)} sample images...")

for idx, foggy_path in enumerate(samples, 1):
    print(f"\n[{idx}/{len(samples)}] {foggy_path.name}")
    
    # Get corresponding paths
    city_frame = foggy_path.stem.replace("_leftImg8bit_foggy_beta_0.02", "")
    city = foggy_path.parent.name
    
    # Restored image path
    restored_path = RESTORED_DIR / "val" / foggy_path.name
    
    # Clear image path
    clear_path = CLEAR_DIR / "val" / city / f"{city_frame}_leftImg8bit.png"
    
    # Check all images exist
    if not restored_path.exists():
        print(f"  ⚠ Restored image not found, skipping")
        continue
    if not clear_path.exists():
        print(f"  ⚠ Clear image not found, skipping")
        continue
    
    # Load images
    foggy = cv2.imread(str(foggy_path))
    restored = cv2.imread(str(restored_path))
    clear = cv2.imread(str(clear_path))
    
    # Resize for display (width=640 each panel)
    h, w = foggy.shape[:2]
    display_width = 640
    scale = display_width / w
    display_height = int(h * scale)
    
    foggy_resized = cv2.resize(foggy, (display_width, display_height))
    restored_resized = cv2.resize(restored, (display_width, display_height))
    clear_resized = cv2.resize(clear, (display_width, display_height))
    
    # Run YOLO detection on each
    foggy_results = model.predict(foggy_resized, conf=0.25, verbose=False)[0]
    restored_results = model.predict(restored_resized, conf=0.25, verbose=False)[0]
    clear_results = model.predict(clear_resized, conf=0.25, verbose=False)[0]
    
    # Draw boxes
    foggy_viz = foggy_results.plot()
    restored_viz = restored_results.plot()
    clear_viz = clear_results.plot()
    
    # Add labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.0
    thickness = 2
    
    cv2.putText(foggy_viz, f"Foggy (Beta=0.02) [{len(foggy_results.boxes)} detections]", 
                (10, 40), font, font_scale, (0, 255, 255), thickness, cv2.LINE_AA)
    cv2.putText(restored_viz, f"WUNET Restored [{len(restored_results.boxes)} detections]", 
                (10, 40), font, font_scale, (0, 255, 0), thickness, cv2.LINE_AA)
    cv2.putText(clear_viz, f"Clear Reference [{len(clear_results.boxes)} detections]", 
                (10, 40), font, font_scale, (255, 255, 0), thickness, cv2.LINE_AA)
    
    # Combine side by side
    combined = np.hstack([foggy_viz, restored_viz, clear_viz])
    
    # Save
    output_path = OUTPUT_DIR / f"sample_{idx:02d}_{foggy_path.name}"
    cv2.imwrite(str(output_path), combined)
    
    print(f"  ✓ Foggy: {len(foggy_results.boxes)}, Restored: {len(restored_results.boxes)}, Clear: {len(clear_results.boxes)}")

print("\n" + "="*70)
print(f"✓ Created {len(samples)} visualizations")
print(f"  Output directory: {OUTPUT_DIR}")
print("="*70)
