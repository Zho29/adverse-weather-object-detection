"""Decoding and supervised loss for the custom DSNet grid detection head.

This is a small research head, not an Ultralytics YOLOv8 output format. Each
scale contains three slots of (x, y, width, height, objectness, class logits).
Historical checkpoints used unsupervised width/height values; ``legacy``
decoding retains their absolute-value interpretation. New training uses
``sigmoid`` normalized widths/heights and records that codec in checkpoints.
"""

import torch
import torch.nn.functional as F
from torchvision.ops import batched_nms


def _reshape_head(prediction, num_classes):
    if prediction.ndim != 4:
        raise ValueError("Detection heads must have shape [B, channels, H, W]")
    batch, channels, height, width = prediction.shape
    outputs = 5 + num_classes
    if num_classes < 1 or channels != 3 * outputs:
        raise ValueError(f"Expected {3 * outputs} channels for {num_classes} classes, got {channels}")
    return prediction.reshape(batch, 3, outputs, height, width).permute(0, 1, 3, 4, 2)


def decode_detections(predictions, num_classes, image_size=(512, 512),
                      confidence=0.25, iou_threshold=0.45,
                      max_detections=300, box_format="legacy"):
    """Return one Nx6 (xyxy, score, class) tensor per image after class NMS.

    Coordinates use ``image_size=(height, width)``. Strides are computed from
    actual feature-map shapes; the first two DSNet heads share a resolution.
    """
    if not predictions or not 0 <= confidence <= 1 or not 0 <= iou_threshold <= 1:
        raise ValueError("Supply heads and confidence/IoU thresholds in [0, 1]")
    if box_format not in {"legacy", "sigmoid"}:
        raise ValueError("box_format must be 'legacy' or 'sigmoid'")
    image_h, image_w = image_size
    if image_h <= 0 or image_w <= 0 or max_detections < 1:
        raise ValueError("Image dimensions and max_detections must be positive")
    batch_size = predictions[0].shape[0]
    candidates = [[] for _ in range(batch_size)]
    for raw in predictions:
        pred = _reshape_head(raw, num_classes)
        if pred.shape[0] != batch_size:
            raise ValueError("All detection heads must have the same batch size")
        grid_h, grid_w = pred.shape[2:4]
        ys, xs = torch.meshgrid(torch.arange(grid_h, device=pred.device),
                                torch.arange(grid_w, device=pred.device), indexing="ij")
        grid = torch.stack((xs, ys), dim=-1).to(pred.dtype)
        xy = (pred[..., :2].sigmoid() + grid) * pred.new_tensor([image_w / grid_w, image_h / grid_h])
        wh = pred[..., 2:4].abs() if box_format == "legacy" else pred[..., 2:4].sigmoid()
        wh = wh * pred.new_tensor([image_w, image_h])
        boxes = torch.cat((xy - wh / 2, xy + wh / 2), dim=-1)
        boxes[..., [0, 2]] = boxes[..., [0, 2]].clamp(0, image_w)
        boxes[..., [1, 3]] = boxes[..., [1, 3]].clamp(0, image_h)
        class_score, class_id = pred[..., 5:].sigmoid().max(dim=-1)
        scores = pred[..., 4].sigmoid() * class_score
        for index in range(batch_size):
            image_boxes = boxes[index].reshape(-1, 4)
            image_scores = scores[index].reshape(-1)
            image_classes = class_id[index].reshape(-1)
            valid = ((image_scores >= confidence) & torch.isfinite(image_boxes).all(dim=1)
                     & torch.isfinite(image_scores) & (image_boxes[:, 2] > image_boxes[:, 0])
                     & (image_boxes[:, 3] > image_boxes[:, 1]))
            candidates[index].append(torch.cat((image_boxes[valid], image_scores[valid, None],
                                                 image_classes[valid, None].to(pred.dtype)), dim=1))
    results = []
    for image_candidates in candidates:
        detections = torch.cat(image_candidates, dim=0)
        if len(detections):
            # Bound NMS work on untrained/noisy heads before the final limit.
            detections = detections[detections[:, 4].argsort(descending=True)[:30000]]
            keep = batched_nms(detections[:, :4], detections[:, 4],
                               detections[:, 5].long(), iou_threshold)[:max_detections]
            detections = detections[keep]
        results.append(detections)
    return results


def compute_detection_loss(predictions, gt_boxes, gt_labels, image_size=(512, 512),
                           num_classes=2):
    """Supervise every head's objectness plus positive-cell boxes and classes.

    Labels must be zero based; boxes are xyxy pixels at ``image_size``. Each
    ground truth occupies an available slot in its center cell on each scale.
    This simple assignment is a research baseline, not the YOLOv8 criterion.
    """
    if not predictions:
        raise ValueError("At least one detection head is required")
    image_h, image_w = image_size
    if image_h <= 0 or image_w <= 0:
        raise ValueError("Image dimensions must be positive")
    total_loss = predictions[0].sum() * 0
    for raw in predictions:
        pred = _reshape_head(raw, num_classes)
        batch, slots, grid_h, grid_w, _ = pred.shape
        if len(gt_boxes) != batch or len(gt_labels) != batch:
            raise ValueError("Ground truth batch size must match predictions")
        target_obj = torch.zeros_like(pred[..., 4])
        target_box = torch.zeros_like(pred[..., :4])
        target_cls = torch.zeros_like(pred[..., 5:])
        for index, (boxes, labels) in enumerate(zip(gt_boxes, gt_labels)):
            boxes = torch.as_tensor(boxes, device=pred.device, dtype=pred.dtype)
            labels = torch.as_tensor(labels, device=pred.device, dtype=torch.long)
            if boxes.ndim != 2 or boxes.shape[-1] != 4 or len(boxes) != len(labels):
                raise ValueError("Boxes must be Nx4 with one class label per box")
            if not torch.isfinite(boxes).all() or ((labels < 0) | (labels >= num_classes)).any():
                raise ValueError("Boxes must be finite and labels must be zero-based class IDs")
            for box, label in zip(boxes, labels):
                clipped = box.clone()
                clipped[[0, 2]] = clipped[[0, 2]].clamp(0, image_w)
                clipped[[1, 3]] = clipped[[1, 3]].clamp(0, image_h)
                size = clipped[2:] - clipped[:2]
                if (size <= 0).any():
                    continue
                center = (clipped[:2] + clipped[2:]) / 2
                scaled = center / pred.new_tensor([image_w / grid_w, image_h / grid_h])
                column = min(int(scaled[0]), grid_w - 1)
                row = min(int(scaled[1]), grid_h - 1)
                available = (target_obj[index, :, row, column] == 0).nonzero().flatten()
                if not len(available):
                    continue
                slot = int(available[0])
                target_obj[index, slot, row, column] = 1
                target_box[index, slot, row, column] = torch.cat((
                    scaled - pred.new_tensor([column, row]),
                    size / pred.new_tensor([image_w, image_h])))
                target_cls[index, slot, row, column, label] = 1
        positives = target_obj.bool()
        objectness_loss = F.binary_cross_entropy_with_logits(pred[..., 4], target_obj)
        if positives.any():
            # Explicit positive-cell loss prevents sparse positives disappearing
            # inside the much larger background average.
            objectness_loss = objectness_loss + F.binary_cross_entropy_with_logits(
                pred[..., 4][positives], target_obj[positives])
            box_loss = F.smooth_l1_loss(pred[..., :4][positives].sigmoid(), target_box[positives])
            class_loss = F.binary_cross_entropy_with_logits(pred[..., 5:][positives], target_cls[positives])
        else:
            box_loss = pred[..., :4].sum() * 0
            class_loss = pred[..., 5:].sum() * 0
        total_loss = total_loss + objectness_loss + 5 * box_loss + class_loss
    return total_loss / len(predictions)
