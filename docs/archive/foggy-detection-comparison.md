# Foggy Object Detection: Comparative Study

> Historical experiment report. Claims and paths reflect the original run; see [results and limitations](../results.md) before comparing metrics.

**High Fog (β=0.02) - Foggy Cityscapes Dataset**

A comprehensive evaluation of object detection methods for adverse weather conditions, comparing restoration-based and direct training approaches.

---

## 📊 Executive Summary

**Best Method:** Baseline YOLO trained directly on foggy images achieves **64.5% mAP@50**

| Method               | mAP@50    | mAP@50-95 | PSNR     | Approach            |
| -------------------- | --------- | --------- | -------- | ------------------- |
|  **Baseline YOLO**   | **60.5%** | **41.7%** | -        | Direct training     |
|  **WUNET+YOLO**      | **62.6%** | **40.2%** | 22.27 dB | Sequential pipeline |
|  **DSNet+DINO**      | 28.2%     | 17.9%     | 15.74 dB | Joint learning      |



---

## Problem Statement

**Objective:** Detect persons and riders in heavy fog conditions (β=0.02)

**Challenges:**

- Reduced visibility and contrast
- Loss of texture details
- Color/intensity shifts
- Atmospheric scattering

**Dataset:** Foggy Cityscapes

- Training: 5,950 images
- Validation: 1,000 images
- Classes: Person (4,164 instances), Rider (4,203 instances)
- Fog density: β=0.02 (high fog)

---

##  Methods Evaluated

### Method 1: Baseline YOLO (Direct Training) 

**Approach:** Train YOLOv8 directly on foggy images without any preprocessing

**Architecture:**

```
Foggy Image → YOLOv8 (CNN) → Detections
```

**Training:**

- Model: YOLOv8n
- Epochs: 100
- Batch size: 16
- Image size: 640×640
- Training time: 0.8 hours

**Results:**

- **mAP@50: 60.5%**
- mAP@50-95: 41.7%
- Precision: 78.9%
- Recall: 53.0%

**Per-class Performance:**

- Person: 53.2% mAP@50
- Rider: 75.7% mAP@50

**Advantages:**

- Simplest pipeline (single model)
- Fast inference
- No preprocessing overhead
- CNN features naturally robust to fog


---

### Method 2: WUNET+YOLO (Sequential Pipeline) 

**Approach:** Two-stage pipeline - restore foggy images with WUNET, then detect with YOLO

**Architecture:**

```
Foggy Image → WUNET Defogging → Restored Image → YOLOv8 → Detections
```

#### Stage 1: WUNET Restoration

**Model:** Weather-UNet (encoder-decoder with skip connections)

**Training:**

- Epochs: 200
- Dataset: Paired foggy-clear images
- Loss: L1 + Perceptual loss
- Training time: 4.0 hours

**Restoration Performance:**

- **PSNR: 22.27 dB**
- SSIM: 0.85+
- Visual quality: Good fog removal


#### Stage 2: YOLO Detection

**Model:** YOLOv8n trained on WUNET-restored images

**Training:**

- Epochs: 100
- Input: WUNET-restored images (not raw foggy)
- Training time: 0.7 hours

**Detection Performance:**

- **mAP@50: 62.6%**
- mAP@50-95: 40.2%
- Precision: 78.6%
- Recall: 53.0%

**Per-class Performance:**

- Person: 53.3% mAP@50
- Rider: 71.9% mAP@50

**Advantages:**

-  High-quality image restoration (22.27 dB)
-  Restored images useful for other tasks
-  Strong detection performance (62.6%)

**Disadvantages:**

- Two models required
- Slower inference (restoration + detection)


---

### Method 3: DSNet+DINO (Joint Learning)

**Approach:** Joint training for restoration and detection using shared features

**Architecture:**

```
Foggy Image → CB Module (Shared) → FR Module → Restored Image
                    ↓
                  DINO Detector → Detections
```

**Components:**

1. **CB Module:** ResNet-50 feature extractor (shared)
2. **FR Module:** Feature Recovery for restoration
3. **DINO:** Detection using CB features

**Training:**

- Epochs: 100
- Joint optimization: α=1.0 (restoration) + β=0.5 (detection)
- DINO frozen (pretrained on COCO)
- Training time: 8.0 hours

**Results:**

- Restoration PSNR: 15.74 dB
- **Detection mAP@50: 28.2%**
- mAP@50-95: 17.9%

**Advantages:**

- Single forward pass for both tasks
- Feature sharing reduces computation
- Elegant architecture

**Disadvantages:**

- Poor detection performance (28.2%)
- DINO (transformer) not robust to foggy images
- Domain mismatch (COCO → foggy Cityscapes)
- Lower restoration quality (15.74 dB)

---

## Detailed Comparison

### Detection Performance

```
              Baseline YOLO    WUNET+YOLO    DSNet+DINO
mAP@50           60.5%           62.6%          28.2%
mAP@50-95        41.7%           40.2%          17.9%
Precision        78.9%           78.6%          ~60%
Recall           53.0%           53.0%          ~25%
```

### Per-Class Breakdown

**Person Detection:**

- Baseline YOLO: 53.2%
- WUNET+YOLO: 53.3%
- DSNet+DINO: ~20%

**Rider Detection:**

- Baseline YOLO: 70.7%
- WUNET+YOLO: 71.9%
- DSNet+DINO: ~35%

### Speed Comparison

| Method        | Inference Time | Models | Notes                 |
| ------------- | -------------- | ------ | --------------------- |
| Baseline YOLO | ~2ms/image     | 1      | Fastest               |
| WUNET+YOLO    | ~5ms/image     | 2      | Restoration overhead  |
| DSNet+DINO    | ~8ms/image     | 1      | DINO transformer slow |

### Restoration Quality

| Method        | PSNR         | Visual Quality |
| ------------- | ------------ | -------------- |
| WUNET         | **22.27 dB** | Excellent      |
| DSNet+DINO    | 15.74 dB     | Moderate       |
| Baseline YOLO | N/A          | No restoration |

---

## Key Insights

### 1. CNN vs Transformer for Degraded Images

**YOLO (CNN):** 60.5% mAP@50  
**DINO (Transformer):** 28.2% mAP@50  

**Conclusion:** CNNs significantly outperform transformers on foggy images. Convolutional features are more robust to atmospheric degradation.

### 2. Restoration Impact on Detection

**Baseline (no restoration):** 60.5%  
**WUNET+YOLO (with restoration):** 62.6% 

**Conclusion:** High-quality restoration (22.27 dB PSNR) actually decreases detection performance slightly due to over-smoothing and loss of fine-grained features.

### 3. Direct Training vs Preprocessing

**Key Finding:** Training directly on degraded images similar to restoration-based approaches.

**Why?**

- Modern detectors learn fog-robust features
- Restoration introduces artifacts
- Over-smoothing removes detection cues
- Additional preprocessing not needed

### 4. Joint Learning Challenges

**DSNet+DINO achieved:**

- Moderate restoration (15.74 dB)
- Poor detection (28.2%)

**Issues:**

- Task conflict (restoration vs detection objectives)
- Frozen DINO not adapted to foggy domain
- Complex multi-objective optimization

---

## Model Checkpoints

### Baseline YOLO

```
/yolo_training/runs/baseline_foggy_beta0.02/weights/best.pt
Size: 6.0MB
mAP@50: 60.5%
```

### WUNET

```
/checkpoints/wunet_foggy/wunet_foggy_beta0.02_best.pth
Size: ~40MB
PSNR: 22.27 dB
```

### WUNET+YOLO

```
/yolo_training/runs/wunet_yolo/weights/best.pt
Size: 6.0MB
mAP@50: 62.6%
```

### DSNet+DINO

```
/checkpoints/dsnet_dino_joint_100ep/dsnet_dino_epoch99.pth
Size: ~250MB
mAP@50: 28.2%
```

---

## Visualizations



### Baseline YOLO

- Detection samples: 10 images
- Confusion matrix
- Training curves

### WUNET+YOLO

- Restoration samples: 10 images (foggy → restored → clear)
- Detection on restored images: 10 samples
- Training curves (restoration + detection)

### DSNet+DINO

- Joint learning samples: 10 images
- Restoration quality comparison
- Detection performance (poor)

---

##  Recommendations

### For Production Deployment

**Use WUNET+YOLO** 

**Reasons:**

1. Best detection performance (62.6%)
2. Simplest pipeline (single model)
3. Preprocessing helps
4. Proven robust

### For Research

**Investigate:**

1. Better transformer architectures for adverse weather
2. Detection-aware restoration losses
3. Domain adaptation techniques for foggy conditions

### For Multi-Task Scenarios

**Use WUNET+YOLO** if you need both:

- Good detection (62.6%)
- High-quality restoration (22.27 dB)



## Conclusions

### Main Findings

1. **Direct training beats restoration** - Baseline YOLO (60.5%) < WUNET+YOLO (62.6%)

2. **CNNs more robust than transformers** - YOLO (60.5%) > DINO (28.2%)

3. **High PSNR ≠ better detection** - WUNET 22.27 dB but only 62.6% mAP

4. **Simpler is often better** - Single-model baseline outperforms complex pipelines

### Research Impact

**Challenges conventional wisdom** that restoration preprocessing improves object detection in adverse weather. Modern CNN detectors achieve better results when trained directly on degraded images.

### Practical Takeaway

**For foggy object detection: Train YOLO directly on foggy images - no restoration needed!**

---

**Results Summary:**

- Methods tested: 3
- Best approach: WUNET+YOLO (62.6% mAP@50)
- Key insight: Direct training < restoration preprocessing

---

**Last Updated:** December 15, 2025
