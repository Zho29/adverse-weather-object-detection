"""
Convert Unified Foggy Cityscapes Annotations to YOLO Format

Reads unified JSON annotations and converts bounding boxes to YOLO format:
- YOLO format: class_id x_center y_center width height (all normalized 0-1)
- Classes: 0=person, 1=vehicle

Output structure:
    yolo_dataset/
    ├── images/
    │   ├── train/
    │   ├── val/
    │   └── test/
    ├── labels/  
    │   ├── train/
    │   ├── val/
    │   └── test/
    └── data.yaml
"""

import json
import shutil
from pathlib import Path
from tqdm import tqdm


class YOLOConverter:
    """Convert unified JSON annotations to YOLO format"""
    
    def __init__(self, foggy_root, output_root):
        self.foggy_root = Path(foggy_root)
        self.output_root = Path(output_root)
        self.unified_annotations = self.foggy_root / "unified_annotations"
        
        # Class mapping
        self.class_map = {
            'person': 0,
            'vehicle': 1
        }
        
        self.stats = {
            'train': {'images': 0, 'persons': 0, 'vehicles': 0},
            'val': {'images': 0, 'persons': 0, 'vehicles': 0},
            'test': {'images': 0, 'persons': 0, 'vehicles': 0}
        }
    
    def convert_bbox_to_yolo(self, bbox, img_width, img_height):
        """
        Convert bounding box from [x, y, w, h] to YOLO format.
        
        YOLO format: [x_center, y_center, width, height] all normalized to [0, 1]
        """
        x, y, w, h = bbox
        
        # Calculate center
        x_center = x + w / 2
        y_center = y + h / 2
        
        # Normalize
        x_center_norm = x_center / img_width
        y_center_norm = y_center / img_height
        w_norm = w / img_width
        h_norm = h / img_height
        
        # Clamp to [0, 1]
        x_center_norm = max(0, min(1, x_center_norm))
        y_center_norm = max(0, min(1, y_center_norm))
        w_norm = max(0, min(1, w_norm))
        h_norm = max(0, min(1, h_norm))
        
        return [x_center_norm, y_center_norm, w_norm, h_norm]
    
    def process_split(self, split, beta=0.01):
        """
        Process one split (train/val/test) for a specific fog density.
        
        Args:
            split: 'train', 'val', or 'test'
            beta: Fog density (0.005, 0.01, or 0.02)
        """
        print(f"\nProcessing {split} split (beta={beta})...")
        
        # Create output directories
        images_dir = self.output_root / "images" / split
        labels_dir = self.output_root / "labels" / split
        images_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)
        
        # Process each city in this split
        annotations_dir = self.unified_annotations / split
        
        if not annotations_dir.exists():
            print(f"  ⚠️  No annotations found for {split}")
            return
        
        for city_dir in sorted(annotations_dir.iterdir()):
            if not city_dir.is_dir():
                continue
            
            city_name = city_dir.name
            
            # Process all annotation files in this city
            for json_file in sorted(city_dir.glob("*.json")):
                base_name = json_file.stem.replace('_unified', '')
                
                # Load annotation
                with open(json_file, 'r') as f:
                    ann = json.load(f)
                
                # Get corresponding foggy image path
                foggy_img_name = f"{base_name}_leftImg8bit_foggy_beta_{beta}.png"
                foggy_img_path = self.foggy_root / "leftImg8bit_foggy" / split / city_name / foggy_img_name
                
                if not foggy_img_path.exists():
                    continue
                
                # Copy image to YOLO dataset
                dst_img_path = images_dir / f"{base_name}_{city_name}_beta{beta}.png"
                shutil.copy(foggy_img_path, dst_img_path)
                
                # Create YOLO label file
                label_file = labels_dir / f"{base_name}_{city_name}_beta{beta}.txt"
                
                img_width = ann['img_width']
                img_height = ann['img_height']
                
                with open(label_file, 'w') as f:
                    for obj in ann['objects']:
                        class_name = obj['class']
                        class_id = self.class_map[class_name]
                        
                        # Convert bbox to YOLO format
                        bbox = obj['bbox']
                        yolo_bbox = self.convert_bbox_to_yolo(bbox, img_width, img_height)
                        
                        # Write: class_id x_center y_center width height
                        f.write(f"{class_id} {' '.join(map(str, yolo_bbox))}\n")
                        
                        # Update stats
                        self.stats[split]['images'] = self.stats[split].get('images', 0)
                        if class_name == 'person':
                            self.stats[split]['persons'] += 1
                        else:
                            self.stats[split]['vehicles'] += 1
                
                self.stats[split]['images'] += 1
        
        print(f"  ✓ Processed {self.stats[split]['images']} images")
        print(f"    Persons: {self.stats[split]['persons']}, Vehicles: {self.stats[split]['vehicles']}")
    
    def create_yaml_config(self, beta):
        """Create data.yaml configuration file for YOLO"""
        yaml_content = f"""# Foggy Cityscapes Dataset Configuration for YOLO
# Fog density: beta={beta}
# Classes: person (0), vehicle (1)

path: {self.output_root.absolute()}
train: images/train
val: images/val
test: images/test

# Number of classes
nc: 2

# Class names
names:
  0: person
  1: vehicle

# Dataset info
fog_density: {beta}
total_images:
  train: {self.stats['train']['images']}
  val: {self.stats['val']['images']}
  test: {self.stats['test']['images']}
"""
        
        yaml_path = self.output_root / "data.yaml"
        with open(yaml_path, 'w') as f:
            f.write(yaml_content)
        
        print(f"\n✓ Created YOLO config: {yaml_path}")
    
    def convert(self, beta=0.01):
        """
        Convert entire dataset for a specific fog density.
        
        Args:
            beta: Fog density (0.005, 0.01, or 0.02)
        """
        print("=" * 70)
        print(f"Converting Foggy Cityscapes to YOLO Format (beta={beta})")
        print("=" * 70)
        
        # Process all splits
        for split in ['train', 'val', 'test']:
            self.process_split(split, beta)
        
        # Create YAML config
        self.create_yaml_config(beta)
        
        # Print summary
        print("\n" + "=" * 70)
        print("Conversion Complete!")
        print("=" * 70)
        print(f"\nDataset Statistics (beta={beta}):")
        print(f"{'Split':<10} {'Images':<10} {'Persons':<12} {'Vehicles':<12}")
        print("-" * 70)
        for split in ['train', 'val', 'test']:
            stats = self.stats[split]
            print(f"{split:<10} {stats['images']:<10} {stats['persons']:<12} {stats['vehicles']:<12}")
        
        print(f"\n✓ YOLO dataset created at: {self.output_root}")
        print(f"  - Images: {self.output_root}/images/")
        print(f"  - Labels: {self.output_root}/labels/")
        print(f"  - Config: {self.output_root}/data.yaml")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Convert Foggy Cityscapes to YOLO format')
    parser.add_argument('--foggy-root', type=str,
                        default='/mnt/Data/as19197/new_project/Foggy_Cityscapes_dataset',
                        help='Root directory of Foggy Cityscapes')
    parser.add_argument('--output-root', type=str,
                        default='/mnt/Data/as19197/new_project/yolo_dataset',
                        help='Output directory for YOLO dataset')
    parser.add_argument('--beta', type=float, default=0.01,
                        choices=[0.005, 0.01, 0.02],
                        help='Fog density: 0.005 (low), 0.01 (medium), 0.02 (high)')
    
    args = parser.parse_args()
    
    # Convert
    converter = YOLOConverter(args.foggy_root, args.output_root)
    converter.convert(beta=args.beta)


if __name__ == '__main__':
    main()
