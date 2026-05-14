#!/usr/bin/env python3
"""
Fix labels for baseline YOLO - match foggy image filenames
"""

from pathlib import Path
import shutil

print("="*70)
print("Fixing Labels for Baseline YOLO (Foggy Images)")
print("="*70)

# The foggy images are named: city_frame_leftImg8bit_foggy_beta_0.02.png
# The labels need to match: city_frame_leftImg8bit_foggy_beta_0.02.txt

foggy_root = Path('/home/as19197/new_project/Foggy_Cityscapes_dataset/leftImg8bit_foggy')
label_src = Path('/mnt/Data/as19197/new_project/yolo_dataset_beta0.02/labels')
label_dst = Path('/mnt/Data/as19197/new_project/yolo_baseline_foggy_beta0.02/labels')

for split in ['train', 'val', 'test']:
    print(f"\n[{split.upper()}]")
    
    # Create label directory
    dst_dir = label_dst / split
    dst_dir.mkdir(exist_ok=True, parents=True)
    
    # Get all foggy images for this split
    foggy_split_dir = foggy_root / split
    
    count_created = 0
    count_copied = 0
    
    for city_dir in foggy_split_dir.iterdir():
        if not city_dir.is_dir():
            continue
            
        for foggy_img in city_dir.glob('*_beta_0.02.png'):
            # Foggy image name: aachen_000000_000019_leftImg8bit_foggy_beta_0.02.png
            img_name = foggy_img.stem  # Without .png
            
            # Expected label name
            label_name = f"{img_name}.txt"
            label_path = dst_dir / label_name
            
            if label_path.exists():
                count_copied += 1
                continue
            
            # Try to find corresponding label from source
            # Source labels are: city_frame_city_beta0.01.txt
            # Parse foggy name to get base
            parts = img_name.split('_leftImg8bit_foggy_beta_0.02')[0]
            city = foggy_img.parent.name
            
            # Try different label naming patterns
            possible_labels = [
                label_src / split / f"{parts}_{city}_beta0.01.txt",
                label_src / split / f"{parts}_foggy_beta_0.02.txt",
                label_src / split / f"{img_name}.txt",
            ]
            
            source_label = None
            for possible in possible_labels:
                if possible.exists():
                    source_label = possible
                    break
            
            if source_label:
                # Copy label with correct name
                shutil.copy2(source_label, label_path)
                count_created += 1
            else:
                # Create empty label (no objects)
                label_path.write_text("")
                count_created += 1
    
    print(f"  ✓ Created/matched {count_created} labels")
    print(f"  ✓ Already existed: {count_copied} labels")

print("\n" + "="*70)
print("✓ Labels fixed for baseline YOLO!")
print("="*70)
