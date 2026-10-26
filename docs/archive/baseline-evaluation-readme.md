# Baseline Evaluation - Object Detection in Foggy Conditions 🌫️

> Historical experiment report. Claims and paths reflect the original run; see [results and limitations](../results.md) before comparing metrics.

**Date:** December 29, 2025  
**Dataset:** Foggy Cityscapes (β=0.02 - High Fog)  
**Evaluation Metric:** mAP@0.5 (mean Average Precision at IoU 0.5)

---

## Overview

This experiment evaluates different object detection approaches for autonomous driving in foggy conditions, comparing:

1. **Direct foggy training** - YOLO trained on raw foggy images
2. **Restoration + detection** - YOLO trained on WUNET-restored images
3. **Pretrained baseline** - YOLOv8n pretrained on COCO (no fog adaptation)

**Goal:** Determine the best approach for object detection in heavy fog conditions.

---

## Results Summary

### Performance on Foggy Images (β=0.02)

| Approach                   | mAP@50    | mAP@50-95 | Precision | Recall    |
| -------------------------- | --------- | --------- | --------- | --------- |
| **Baseline YOLO (Foggy)**  | **65.1%** | **41.7%** | **79.8%** | **56.5%** |
| **WUNET + YOLO**           | **53.2%** | **34.0%** | **77.7%** | **43.6%** |
| **Pretrained YOLO (COCO)** | **19.3%** | **8.8%**  | **44.6%** | **15.6%** |

### Performance Gains

- Foggy training: **+238% vs pretrained** (65.1% vs 19.3%)
- WUNET+YOLO: **+176% vs pretrained** (53.2% vs 19.3%)
- Foggy training: **+22% vs WUNET** (65.1% vs 53.2%)

---

## Key Findings

### **Direct Foggy Training is BEST**

- **65.1% mAP@50** - highest performance
- Learns fog-invariant features naturally
- No preprocessing overhead at inference
- Simpler pipeline, more robust
- **RECOMMENDED FOR PRODUCTION**

### **WUNET Restoration Helps, But Not Optimal**

- **53.2% mAP@50** - mid-tier performance
- 18% lower than direct foggy training
- **Trade-offs:**
  - Better than no adaptation (2.7x pretrained)
  - Information loss during restoration
  - Lower recall (43.6% vs 56.5%)
  - Additional computational cost
- **Use case:** When you need both restored images AND detection

### **Pretrained Models Fail in Fog**

- **19.3% mAP@50** - severe performance drop
- 70% decrease from typical COCO performance (~60-70% mAP@50)
- Very low recall (15.6%) - misses 84% of objects
- **Critical need for domain-specific training**

---

## 📊 Detailed Metrics

### Model Comparison Chart

```
mAP@50 Performance (Higher is Better):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Baseline YOLO (Foggy)  ████████████████████  65.1%
WUNET + YOLO          ████████████████       53.2%
Pretrained YOLO       ████                   19.3%
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Recall - Detection Rate (Higher is Better):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Baseline YOLO (Foggy)  ████████████████████  56.5%
WUNET + YOLO          ██████████████         43.6%
Pretrained YOLO       ████                   15.6%
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Precision - Accuracy (Higher is Better):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Baseline YOLO (Foggy)  ████████████████████  79.8%
WUNET + YOLO          ███████████████████    77.7%
Pretrained YOLO       ██████████             44.6%
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

### Why Baseline YOLO Wins

| Metric        | Value | Analysis                                                        |
| ------------- | ----- | --------------------------------------------------------------- |
| **mAP@50**    | 65.1% | Best overall detection performance                              |
| **Precision** | 79.8% | High accuracy - 80% of detections are correct                   |
| **Recall**    | 56.5% | Detects 56.5% of all objects (vs 43.6% WUNET, 15.6% pretrained) |
| **Balance**   | ★★★★★ | Excellent precision-recall trade-off                            |

---

## Experimental Setup

### Dataset Details

- **Source:** Foggy Cityscapes Validation Set
- **Fog Level:** β=0.02 (heavy fog - worst case)
- **Images:** 500 validation images
- **Ground Truth:** 8,367 object instances
- **Classes:** person, rider (2 classes)
- **Resolution:** 640×640 (YOLO input)

### Training Configuration

| Model         | Training Data         | Epochs | Batch Size | Device              |
| ------------- | --------------------- | ------ | ---------- | ------------------- |
| Baseline YOLO | Raw foggy images      | 100    | 16         | 3× A6000 GPUs       |
| WUNET+YOLO    | WUNET-restored images | 100    | 16         | 3× A6000 GPUs       |
| DSNET+YOLO    | COCO dataset          | N/A    | N/A        | Pre-trained weights |

### Evaluation Parameters

- **Metric:** mAP@0.5 (IoU threshold = 0.5)
- **Confidence:** 0.001 (low threshold to capture all predictions)
- **IoU threshold:** 0.6 for NMS
- **Batch size:** 16
- **Device:** NVIDIA RTX A6000 (48GB VRAM)

---

## Recommendations

### For Production Deployment

**USE: Baseline YOLO trained on foggy images**

**Reasons:**

1. **Best Performance:** 65.1% mAP@50
2. **Simplest Pipeline:** No restoration preprocessing
3. **Lowest Latency:** Direct inference on foggy images
4. **Most Robust:** Learned fog-invariant features
5. **Easiest to Deploy:** Standard YOLO pipeline

### When to Consider WUNET+YOLO

Use restoration approach if:

- You need restored images for human review/visualization
- Multi-task scenario (restoration + detection together)
- Very limited foggy training data available
- Accept 18% performance trade-off (53.2% vs 65.1%)
- Accept additional computational cost

**Do NOT use pretrained models without fog-specific training:**

- 70% performance degradation in fog
- Only 15.6% recall - misses 84% of objects
- Safety-critical failure for autonomous driving

### Datasets

- **Foggy Cityscapes:** Synthetic foggy images based on Cityscapes
- **COCO:** Common Objects in Context (pretrained baseline)

### Models

- **YOLOv8n:** Ultralytics YOLOv8 nano (base architecture)
- **WUNET:** Wavelet-based U-Net for image defogging

### Metrics

- **mAP@0.5:** Mean Average Precision at IoU threshold 0.5
- **Precision:** TP / (TP + FP) - accuracy of positive predictions
- **Recall:** TP / (TP + FN) - coverage of ground truth objects

---

---

_Last Updated: 2025-12-29_  
_Evaluation on 500 foggy validation images with 8,367 ground truth objects_
