#!/usr/bin/env python3
"""
Convert Cityscapes Unified Annotations to COCO Format
Classes: person, vehicle (2 classes)
"""

import json
from pathlib import Path
from tqdm import tqdm
import argparse

def convert_cityscapes_to_coco(unified_dir, output_file, split='train'):
    """Convert unified annotations to COCO format"""
    
    unified_path = Path(unified_dir) / split
    
    # COCO format structure
    coco = {
        "images": [],
        "annotations": [],
        "categories": [
            {"id": 1, "name": "person"},
            {"id": 2, "name": "vehicle"}
        ]
    }
    
    image_id = 0
    annotation_id = 0
    
    # Process all JSON files
    json_files = list(unified_path.rglob("*.json"))
    print(f"Found {len(json_files)} annotation files for {split}")
    
    for json_file in tqdm(json_files, desc=f"Converting {split}"):
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        image_id += 1
        
        # Image info
        city = json_file.parent.name
        image_name = data['image_name'].replace('_gtBbox3d', '')
        
        coco["images"].append({
            "id": image_id,
            "file_name": f"{image_name}_leftImg8bit.png",
            "height": data['img_height'],
            "width": data['img_width'],
            "city": city
        })
        
        # Annotations
        for obj in data['objects']:
            class_name = obj['class']
            
            # Map to category ID
            if class_name == 'person':
                category_id = 1
            elif class_name == 'vehicle':
                category_id = 2
            else:
                continue  # Skip other classes
            
            bbox = obj['bbox']  # [x, y, width, height]
            
            annotation_id += 1
            coco["annotations"].append({
                "id": annotation_id,
                "image_id": image_id,
                "category_id": category_id,
                "bbox": bbox,
                "area": bbox[2] * bbox[3],
                "iscrowd": 0,
                "occlusion": obj.get('occlusion', 0.0),
                "truncation": obj.get('truncation', 0.0)
            })
    
    # Save COCO format
    with open(output_file, 'w') as f:
        json.dump(coco, f, indent=2)
    
    print(f"\n✓ Saved {split} annotations to {output_file}")
    print(f"  Images: {len(coco['images'])}")
    print(f"  Annotations: {len(coco['annotations'])}")
    print(f"  Categories: {len(coco['categories'])}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--unified-dir', type=str,
                        default='/home/as19197/new_project/Foggy_Cityscapes_dataset/unified_annotations',
                        help='Path to unified annotations')
    parser.add_argument('--output-dir', type=str,
                        default='/home/as19197/new_project/Foggy_Cityscapes_dataset/coco_annotations',
                        help='Output directory for COCO annotations')
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    print("="*70)
    print("Converting Cityscapes Unified Annotations to COCO Format")
    print("Classes: person, vehicle")
    print("="*70)
    
    # Convert each split
    for split in ['train', 'val', 'test']:
        output_file = output_dir / f"cityscapes_{split}.json"
        convert_cityscapes_to_coco(args.unified_dir, output_file, split)
    
    print("\n" + "="*70)
    print("✓ Conversion Complete!")
    print(f"COCO annotations saved to: {output_dir}")
    print("="*70)


if __name__ == '__main__':
    main()
