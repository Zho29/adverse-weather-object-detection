#!/usr/bin/env python3
"""
Create Side-by-Side Comparison of All Models on Clear Images
"""

import cv2
import numpy as np
from pathlib import Path

print("Creating side-by-side comparison visualizations...")

base_dir = Path('/mnt/Data/as19197/new_project')
output_dir = base_dir / 'clear_comparison_visualizations'
output_dir.mkdir(exist_ok=True)

# Paths to different model results
model_dirs = {
    'Pretrained YOLO\n(9.5 det/img)': base_dir / 'baseline_yolo_clear_detections',
    'Foggy YOLO\n(10.4 det/img)': base_dir / 'clear_image_comparison/foggy_yolo_on_clear',
    'WUNET+YOLO\n(11.9 det/img)': base_dir / 'clear_image_comparison/wunet_yolo_on_clear',
}

# Create comparisons for first 3 images
for img_idx in range(1, 4):
    print(f"\nCreating comparison {img_idx}/3...")
    
    images = []
    labels = []
    
    # Load images from each model
    for label, model_dir in model_dirs.items():
        # Find the image file
        img_files = sorted(model_dir.glob(f'*_{img_idx:02d}.jpg'))
        if img_files:
            img = cv2.imread(str(img_files[0]))
            if img is not None:
                # Resize to consistent size
                img = cv2.resize(img, (800, 450))
                
                # Add label at top
                label_img = np.ones((60, 800, 3), dtype=np.uint8) * 255
                cv2.putText(label_img, label.replace('\n', ' '), (10, 35), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
                
                # Combine label and image
                combined = np.vstack([label_img, img])
                images.append(combined)
                labels.append(label)
                print(f"  ✓ Loaded {label.split()[0]}")
    
    if len(images) == 3:
        # Stack horizontally
        comparison = np.hstack(images)
        
        # Add title
        title_height = 80
        title_img = np.ones((title_height, comparison.shape[1], 3), dtype=np.uint8) * 50
        title_text = f"Clear Image Comparison - Sample {img_idx}"
        text_size = cv2.getTextSize(title_text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 2)[0]
        text_x = (comparison.shape[1] - text_size[0]) // 2
        cv2.putText(title_img, title_text, (text_x, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
        
        # Combine title and comparison
        final = np.vstack([title_img, comparison])
        
        # Save
        output_path = output_dir / f'clear_comparison_{img_idx}.jpg'
        cv2.imwrite(str(output_path), final)
        print(f"  ✓ Saved: {output_path.name}")

print(f"\n{'='*70}")
print(f"✓ Created 3 side-by-side comparisons")
print(f"✓ Saved to: {output_dir}")
print(f"{'='*70}")
