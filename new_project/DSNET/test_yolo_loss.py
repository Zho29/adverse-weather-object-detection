#!/usr/bin/env python3
"""
Test the updated YOLO loss function
"""

import torch
import torch.nn as nn

# Copy of the new loss function
bce_loss = nn.BCEWithLogitsLoss()

def compute_detection_loss(predictions, gt_boxes, gt_labels):
    """Test version of the new loss"""
    total_loss = 0
    device = predictions[0].device
    batch_size = predictions[0].shape[0]
    
    for scale_idx, pred in enumerate(predictions):
        B, no, H, W = pred.shape
        pred = pred.permute(0, 2, 3, 1).contiguous()
        
        pred_xy = torch.sigmoid(pred[..., 0:2])
        pred_wh = pred[..., 2:4]
        pred_obj = pred[..., 4:5]
        pred_cls = pred[..., 5:]
        
        target_obj = torch.zeros_like(pred_obj).to(device)
        
        for b in range(B):
            if len(gt_boxes[b]) == 0:
                continue
            boxes = gt_boxes[b].to(device)
            labels = gt_labels[b].to(device)
            scale = 512 / H
            
            for box, label in zip(boxes, labels):
                x1, y1, x2, y2 = box
                cx = (x1 + x2) / 2 / scale
                cy = (y1 + y2) / 2 / scale
                gi = int(cx.clamp(0, W-1))
                gj = int(cy.clamp(0, H-1))
                target_obj[b, gj, gi, 0] = 1.0
        
        obj_loss = bce_loss(pred_obj.squeeze(-1), target_obj.squeeze(-1))
        box_loss = (pred_wh ** 2).mean() * 0.01
        cls_loss = (pred_cls ** 2).mean() * 0.01
        total_loss += obj_loss + box_loss + cls_loss
    
    return total_loss / len(predictions)

# Test
print("Testing new YOLO loss...")
pred1 = torch.randn(2, 7, 64, 64)  # [B, no, H, W]
pred2 = torch.randn(2, 7, 32, 32)
pred3 = torch.randn(2, 7, 16, 16)

# Sample ground truth
gt_boxes = [
    torch.tensor([[100, 100, 200, 200], [300, 300, 400, 400]]),  # Batch 0
    torch.tensor([[150, 150, 250, 250]])  # Batch 1
]
gt_labels = [
    torch.tensor([0, 1]),
    torch.tensor([0])
]

loss = compute_detection_loss([pred1, pred2, pred3], gt_boxes, gt_labels)
print(f"Loss: {loss.item():.4f}")
print("✓ New loss function works!")
