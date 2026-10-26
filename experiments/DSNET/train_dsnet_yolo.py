#!/usr/bin/env python3
"""
Train DSNet+YOLO with Joint Restoration and Detection
YOLO learns directly from CB features!
"""

from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import sys
sys.path.append(str((Path(__file__).resolve().parent / '.').resolve()))

from dsnet_yolo import DSNet_YOLO
from coco_dataset import CityscapesCocoDataset
from detection import compute_detection_loss
import json
from tqdm import tqdm
import numpy as np

# Custom collate function
def collate_fn(batch):
    foggy = torch.stack([item['foggy'] for item in batch])
    clear = torch.stack([item['clear'] for item in batch])
    boxes = [item['boxes'] for item in batch]
    labels = [item['labels'] for item in batch]
    return {
        'foggy': foggy,
        'clear': clear,
        'boxes': boxes,
        'labels': labels
    }


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description="Train DSNet with a supervised custom grid detection head")
    parser.add_argument('--dataset-root', required=True, help='Paired image and coco_annotations root')
    parser.add_argument('--save-dir', default=str(Path(__file__).resolve().parents[2] / 'outputs/dsnet_yolo'))
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--img-size', type=int, default=512)
    parser.add_argument('--beta', type=float, default=0.02, choices=[0.005, 0.01, 0.02])
    parser.add_argument('--no-pretrained', action='store_true', help='Avoid downloading ResNet initialization')
    args = parser.parse_args(argv)
    if args.epochs < 1 or args.batch_size < 1 or args.num_workers < 0 or args.img_size < 16:
        parser.error('epochs/batch-size must be positive, num-workers nonnegative, and img-size at least 16')
    print("="*70)
    print("DSNet+YOLO Joint Training")
    print("="*70)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")

    # Configuration
    EPOCHS = args.epochs
    BATCH_SIZE = args.batch_size
    SAVE_DIR = args.save_dir
    if Path(SAVE_DIR).exists() and any(Path(SAVE_DIR).iterdir()):
        raise FileExistsError('A new DSNet run needs an empty save directory; choose another --save-dir.')
    Path(SAVE_DIR).mkdir(exist_ok=True, parents=True)

    print("[1] Creating datasets...")


    train_dataset = CityscapesCocoDataset(
        str(Path(args.dataset_root) / 'leftImg8bit_foggy'),
        str(Path(args.dataset_root) / 'leftImg8bit'),
        str(Path(args.dataset_root) / 'coco_annotations/cityscapes_train.json'),
        beta=args.beta, split='train', img_size=args.img_size
    )

    val_dataset = CityscapesCocoDataset(
        str(Path(args.dataset_root) / 'leftImg8bit_foggy'),
        str(Path(args.dataset_root) / 'leftImg8bit'),
        str(Path(args.dataset_root) / 'coco_annotations/cityscapes_val.json'),
        beta=args.beta, split='val', img_size=args.img_size
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=args.num_workers, pin_memory=device.type == 'cuda', collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=args.num_workers, pin_memory=device.type == 'cuda', collate_fn=collate_fn)

    print(f"✓ Train: {len(train_dataset)} images")
    print(f"✓ Val: {len(val_dataset)} images")
    if not len(train_loader) or not len(val_loader):
        raise ValueError("Training and validation datasets must contain images")

    print("\n[2] Creating DSNet+YOLO model...")
    if train_dataset.class_names != val_dataset.class_names:
        raise ValueError("Training and validation categories must match")
    model = DSNet_YOLO(num_classes=len(train_dataset.class_names), pretrained_cb=not args.no_pretrained)

    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)

    model = model.to(device)

    print("\n[3] Setting up training...")
    # Loss weights
    alpha = 1.0  # Restoration
    beta = 0.5   # Detection

    # Losses
    l1_loss = nn.L1Loss()

    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    print(f"Loss weights: α={alpha} (restoration), β={beta} (detection)")

    # Training loop
    best_val_loss = float('inf')
    history = []

    for epoch in range(1, EPOCHS + 1):
        print(f"\n{'='*70}")
        print(f"Epoch {epoch}/{EPOCHS}")
        print(f"{'='*70}")

        # Train
        model.train()
        total_loss = 0
        total_rest = 0
        total_det = 0

        for batch in tqdm(train_loader, desc="Training"):
            foggy = batch['foggy'].to(device)
            clear = batch['clear'].to(device)
            boxes = batch['boxes']
            labels = batch['labels']

            optimizer.zero_grad()

            # Forward
            restored, detections = model(foggy)

            # Loss 1: Restoration
            rest_loss = l1_loss(restored, clear)

            # Loss 2: Supervised custom grid detection
            det_loss = compute_detection_loss(detections, boxes, labels,
                                               image_size=foggy.shape[-2:], num_classes=len(train_dataset.class_names))

            # Total loss
            loss = alpha * rest_loss + beta * det_loss

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_rest += rest_loss.item()
            total_det += det_loss.item()

        avg_loss = total_loss / len(train_loader)
        avg_rest = total_rest / len(train_loader)
        avg_det = total_det / len(train_loader)

        print(f"Train Loss: {avg_loss:.6f} (Rest: {avg_rest:.6f}, Det: {avg_det:.6f})")

        # Validation
        model.eval()
        val_loss = 0
        psnr_values = []

        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                foggy = batch['foggy'].to(device)
                clear = batch['clear'].to(device)

                restored, detections = model(foggy)

                rest_loss = l1_loss(restored, clear)
                det_loss = compute_detection_loss(
                    detections, batch['boxes'], batch['labels'],
                    image_size=foggy.shape[-2:], num_classes=len(train_dataset.class_names))
                val_loss += (alpha * rest_loss + beta * det_loss).item()

                # PSNR
                mse = ((restored - clear) ** 2).mean()
                psnr = 10 * torch.log10(4.0 / (mse + 1e-8))
                psnr_values.append(psnr.item())

        avg_val_loss = val_loss / len(val_loader)
        avg_psnr = float(np.mean(psnr_values))

        print(f"Val Loss: {avg_val_loss:.6f}, PSNR: {avg_psnr:.2f} dB")

        # Save best
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            state_dict = model.state_dict() if not isinstance(model, nn.DataParallel) else model.module.state_dict()
            torch.save({
                'epoch': epoch,
                'model_state_dict': state_dict,
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_loss': best_val_loss,
                'psnr': avg_psnr,
                'class_names': train_dataset.class_names,
                'num_classes': len(train_dataset.class_names),
                'detection_box_format': 'sigmoid'
            }, Path(SAVE_DIR) / 'dsnet_yolo_best.pth')
            print(f"  ✓ Saved best model (PSNR: {avg_psnr:.2f} dB)!")

        scheduler.step()

        history.append({
            'epoch': epoch,
            'train_loss': avg_loss,
            'train_rest': avg_rest,
            'train_det': avg_det,
            'val_loss': avg_val_loss,
            'val_psnr': avg_psnr,
            'lr': optimizer.param_groups[0]['lr']
        })

        with open(Path(SAVE_DIR) / 'history.json', 'w') as f:
            json.dump(history, f, indent=2)

    print("\n" + "="*70)
    print("Training Complete!")
    print(f"Best Val Loss: {best_val_loss:.6f}")
    print(f"Models saved to: {SAVE_DIR}")
    print("="*70)

    print("\nNext steps:")
    print("1. Check if both losses decreased")
    print("2. Evaluate detection performance")
    print("3. If successful → 100-epoch training")


if __name__ == '__main__':
    main()
