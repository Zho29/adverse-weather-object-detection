#!/usr/bin/env python3
"""
Visualize DSNet+DINO with Feature Adapter Results
Show foggy input, restored output, and clear reference
"""

from pathlib import Path

import torch
import sys
sys.path.append(str((Path(__file__).resolve().parent / '.').resolve()))

from dsnet_dino_adapter import DSNet_DINO_WithAdapter
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as transforms
import numpy as np

print("="*70)
print("DSNet+DINO Feature Adapter - Visualization")
print("="*70)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}\n")

# Paths
DINO_CONFIG = '/home/as19197/new_project/mmdetection/configs/dino/dino-4scale_r50_8xb2-12e_coco.py'
DINO_CHECKPOINT = '/home/as19197/new_project/mmdetection/configs/dino/dino-4scale_r50_8xb2-12e_coco_20221202_182705-55b2bba2.pth'
MODEL_CHECKPOINT = '/home/as19197/new_project/checkpoints/dsnet_dino_adapter_5ep/dsnet_dino_adapter_best.pth'

print("[1] Loading DSNet+DINO with Feature Adapter...")
model = DSNet_DINO_WithAdapter(
    dino_config=DINO_CONFIG,
    dino_checkpoint=DINO_CHECKPOINT,
    load_dino=False  # Don't need DINO for visualization
)

# Load trained weights
if Path(MODEL_CHECKPOINT).exists():
    checkpoint = torch.load(MODEL_CHECKPOINT, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    print(f"✓ Loaded trained model from epoch {checkpoint.get('epoch', '?')}")
else:
    print("⚠ No checkpoint found, using untrained model")

model = model.to(device)
model.eval()

# Transform
transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Get validation images
foggy_dir = Path('/home/as19197/new_project/Foggy_Cityscapes_dataset/leftImg8bit_foggy/val')
clear_dir = Path('/home/as19197/new_project/Foggy_Cityscapes_dataset/leftImg8bit/val')

# Get 10 sample images
foggy_images = list(foggy_dir.rglob('*_beta_0.02.png'))[:10]

print(f"\n[2] Creating visualizations for {len(foggy_images)} images...")

vis_dir = Path('/home/as19197/.gemini/antigravity/brain/521be7ca-380d-4b05-9a55-7a6ab4119914')
vis_dir.mkdir(exist_ok=True, parents=True)

for idx, foggy_path in enumerate(foggy_images, 1):
    # Load foggy image
    foggy_img = Image.open(foggy_path).convert('RGB')
    
    # Get corresponding clear image
    city = foggy_path.parent.name
    base_name = foggy_path.stem.replace('_leftImg8bit_foggy_beta_0.02', '')
    clear_path = clear_dir / city / f"{base_name}_leftImg8bit.png"
    
    if not clear_path.exists():
        continue
    
    clear_img = Image.open(clear_path).convert('RGB')
    
    # Transform for model
    foggy_tensor = transform(foggy_img).unsqueeze(0).to(device)
    
    # Run DSNet restoration
    with torch.no_grad():
        restored_tensor = model(foggy_tensor)
    
    # Denormalize
    restored_tensor = (restored_tensor + 1) / 2
    restored_tensor = torch.clamp(restored_tensor, 0, 1)
    restored_np = restored_tensor.squeeze(0).cpu().permute(1, 2, 0).numpy()
    
    # Create side-by-side visualization
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    axes[0].imshow(foggy_img)
    axes[0].set_title('Foggy Input (β=0.02)', fontsize=14, fontweight='bold')
    axes[0].axis('off')
    
    axes[1].imshow(restored_np)
    axes[1].set_title('DSNet+Feature Adapter Restored', fontsize=14, fontweight='bold')
    axes[1].axis('off')
    
    axes[2].imshow(clear_img)
    axes[2].set_title('Clear Reference', fontsize=14, fontweight='bold')
    axes[2].axis('off')
    
    plt.tight_layout()
    
    # Save
    save_path = vis_dir / f'dsnet_adapter_sample_{idx:02d}.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Saved visualization {idx}/10")

print(f"\n✓ Visualizations saved to: {vis_dir}")
print("="*70)
