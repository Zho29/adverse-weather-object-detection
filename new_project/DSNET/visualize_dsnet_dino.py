#!/usr/bin/env python3
"""
Visualize DSNet+DINO Results
Creates side-by-side comparisons of:
- Foggy input
- DSNet restored
- Clear reference
- Detection boxes overlaid
"""

import torch
import sys
sys.path.append('/home/as19197/new_project/DSNET')

from dsnet_dino import DSNet_DINO
import cv2
import numpy as np
from pathlib import Path
from PIL import Image
import torchvision.transforms as transforms

print("="*70)
print("DSNet+DINO Results Visualization")
print("="*70)

# Load trained model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
checkpoint_path = '/home/as19197/new_project/checkpoints/dsnet_dino_5ep/dsnet_dino_best.pth'

print(f"\nLoading model from: {checkpoint_path}")

DINO_CONFIG = '/home/as19197/new_project/mmdetection/configs/dino/dino-4scale_r50_8xb2-12e_coco.py'
DINO_CHECKPOINT = '/home/as19197/new_project/mmdetection/configs/dino/dino-4scale_r50_8xb2-12e_coco_20221202_182705-55b2bba2.pth'

model = DSNet_DINO(
    dino_config=DINO_CONFIG,
    dino_checkpoint=DINO_CHECKPOINT,
    load_dino=True
)

# Load weights
checkpoint = torch.load(checkpoint_path)
model.load_state_dict(checkpoint['model_state_dict'], strict=False)
model = model.to(device)
model.eval()

print("✓ Model loaded")

# Test on sample images
foggy_dir = Path('/home/as19197/new_project/Foggy_Cityscapes_dataset/leftImg8bit_foggy/val')
clear_dir = Path('/home/as19197/new_project/Foggy_Cityscapes_dataset/leftImg8bit/val')
output_dir = Path('/home/as19197/new_project/dsnet_dino_visualizations')
output_dir.mkdir(exist_ok=True)

# Get sample images
sample_imgs = list(foggy_dir.rglob('*_beta_0.02.png'))[:5]

print(f"\nProcessing {len(sample_imgs)} samples...")

transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

for idx, foggy_path in enumerate(sample_imgs, 1):
    print(f"\n[{idx}/{len(sample_imgs)}] {foggy_path.name}")
    
    # Load images
    foggy_img = Image.open(foggy_path).convert('RGB')
    
    # Get clear reference
    city_frame = foggy_path.stem.replace("_leftImg8bit_foggy_beta_0.02", "")
    city = foggy_path.parent.name
    clear_path = clear_dir / city / f"{city_frame}_leftImg8bit.png"
    clear_img = Image.open(clear_path).convert('RGB')
    
    # Process with DSNet
    foggy_tensor = transform(foggy_img).unsqueeze(0).to(device)
    
    with torch.no_grad():
        restored_tensor = model(foggy_tensor)
    
    # Denormalize
    restored_tensor = (restored_tensor + 1) / 2
    restored_tensor = torch.clamp(restored_tensor, 0, 1)
    
    # Convert to numpy
    restored_np = restored_tensor.squeeze(0).cpu().permute(1, 2, 0).numpy()
    restored_np = (restored_np * 255).astype(np.uint8)
    
    # Resize for display
    foggy_cv = cv2.resize(np.array(foggy_img), (512, 512))
    clear_cv = cv2.resize(np.array(clear_img), (512, 512))
    restored_cv = cv2.cvtColor(restored_np, cv2.COLOR_RGB2BGR)
    
    # Add labels
    cv2.putText(foggy_cv, "Foggy Input", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    cv2.putText(restored_cv, "DSNet Restored", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(clear_cv, "Clear Reference", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
    
    # Combine
    combined = np.hstack([foggy_cv, restored_cv, clear_cv])
    
    # Save
    save_path = output_dir / f"sample_{idx:02d}_{foggy_path.stem}.jpg"
    cv2.imwrite(str(save_path), combined)
    print(f"  ✓ Saved to {save_path.name}")

print(f"\n✓ Visualizations saved to: {output_dir}")
print("="*70)
