"""Paired restoration images with zero-based COCO detection targets."""

import json
from pathlib import Path
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms


class CityscapesCocoDataset(Dataset):
    """Dataset with foggy images, clear references, and COCO detection annotations"""

    def __init__(self, foggy_root, clear_root, coco_annotation_file,
                 beta=0.02, split='train', img_size=512):
        self.foggy_root = Path(foggy_root)
        self.clear_root = Path(clear_root)
        self.beta = beta
        self.split = split
        self.img_size = img_size

        # Load COCO annotations
        with open(coco_annotation_file, 'r') as f:
            self.coco = json.load(f)

        # Create image index
        self.images = self.coco['images']
        self.annotations = self.coco['annotations']
        categories = sorted(self.coco.get('categories', []), key=lambda item: item['id'])
        if not categories:
            raise ValueError("COCO annotations must declare categories")
        self.category_to_label = {item['id']: index for index, item in enumerate(categories)}
        self.class_names = [item['name'] for item in categories]
        if img_size < 1:
            raise ValueError("img_size must be positive")

        # Group annotations by image_id
        self.img_to_anns = {}
        for ann in self.annotations:
            img_id = ann['image_id']
            if img_id not in self.img_to_anns:
                self.img_to_anns[img_id] = []
            self.img_to_anns[img_id].append(ann)

        print(f"  Loaded {len(self.images)} {split} images with {len(self.annotations)} annotations")

        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_info = self.images[idx]
        img_id = img_info['id']
        city = img_info.get('city', Path(img_info['file_name']).parent.name)

        # Construct paths
        base_name = Path(img_info['file_name']).name.replace('_leftImg8bit.png', '')
        foggy_path = self.foggy_root / self.split / city / f"{base_name}_leftImg8bit_foggy_beta_{self.beta}.png"
        clear_path = self.clear_root / self.split / city / f"{base_name}_leftImg8bit.png"

        # Load images
        foggy_img = Image.open(foggy_path).convert('RGB')
        clear_img = Image.open(clear_path).convert('RGB')

        # Get original size
        orig_w, orig_h = foggy_img.size

        # Transform
        foggy_tensor = self.transform(foggy_img)
        clear_tensor = self.transform(clear_img)

        # Get annotations for this image
        anns = self.img_to_anns.get(img_id, [])

        # Convert to format for detection
        boxes = []
        labels = []
        for ann in anns:
            if ann.get('iscrowd', 0):
                continue
            bbox = ann['bbox']  # [x, y, w, h]
            # Scale bbox to resized image
            x, y, w, h = bbox
            x_scale = self.img_size / orig_w
            y_scale = self.img_size / orig_h

            scaled_bbox = [
                x * x_scale,
                y * y_scale,
                (x + w) * x_scale,
                (y + h) * y_scale
            ]  # Convert to [x1, y1, x2, y2]

            scaled_bbox[0] = max(0, min(self.img_size, scaled_bbox[0]))
            scaled_bbox[1] = max(0, min(self.img_size, scaled_bbox[1]))
            scaled_bbox[2] = max(0, min(self.img_size, scaled_bbox[2]))
            scaled_bbox[3] = max(0, min(self.img_size, scaled_bbox[3]))
            if scaled_bbox[2] <= scaled_bbox[0] or scaled_bbox[3] <= scaled_bbox[1]:
                continue
            boxes.append(scaled_bbox)
            labels.append(self.category_to_label[ann['category_id']])

        return {
            'foggy': foggy_tensor,
            'clear': clear_tensor,
            'boxes': torch.tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 4)),
            'labels': torch.tensor(labels, dtype=torch.int64) if labels else torch.zeros((0,), dtype=torch.int64),
            'image_id': img_id
        }
