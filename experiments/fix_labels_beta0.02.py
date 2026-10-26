#!/usr/bin/env python3
"""
Fix label filenames to match restored image filenames for YOLO training
"""

from pathlib import Path
import shutil

RESTORED_DIR = Path("/mnt/Data/as19197/new_project/yolo_dataset_beta0.02_restored")

print("=" * 70)
print("Fixing Label Filenames for Beta=0.02 Restored Dataset")
print("=" * 70)

for split in ['train', 'val', 'test']:
    print(f"\nProcessing {split}...")
    
    img_dir = RESTORED_DIR / "images" / split
    label_dir = RESTORED_DIR / "labels" / split
    
    # Get all images
    images = list(img_dir.glob("*.png"))
    
    # Create mapping of old labels to new labels
    renamed = 0
    for img_path in images:
        # Image name: frankfurt_000000_000294_leftImg8bit_foggy_beta_0.02.png
        img_stem = img_path.stem  # frankfurt_000000_000294_leftImg8bit_foggy_beta_0.02
        
        # Extract city_frame: frankfurt_000000_000294
        city_frame = img_stem.replace("_leftImg8bit_foggy_beta_0.02", "")
        
        # Find old label (beta0.01 format)
        old_labels = list(label_dir.glob(f"{city_frame}_*_beta0.01.txt"))
        
        if old_labels:
            old_label = old_labels[0]
            new_label = label_dir / f"{img_stem}.txt"
            
            if old_label != new_label:
                shutil.copy(old_label, new_label)
                old_label.unlink()  # Delete old label
                renamed += 1
    
    print(f"  ✓ Renamed {renamed} labels in {split}")

# Verification
print("\n" + "=" * 70)
print("Verification:")
print("=" * 70)
for split in ['train', 'val', 'test']:
    img_count = len(list((RESTORED_DIR / "images" / split).glob("*.png")))
    label_count = len(list((RESTORED_DIR / "labels" / split).glob("*.txt")))
    print(f"{split}: {img_count} images, {label_count} labels")

print("=" * 70)
print("✓ Label filenames fixed!")
print("=" * 70)
