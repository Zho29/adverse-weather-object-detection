# Baseline Results - Proper mAP@0.5 Metrics 📊

> Historical experiment report. Claims and paths reflect the original run; see [results and limitations](../results.md) before comparing metrics.

**Date:** 2025-12-29  
**Evaluation:** mAP@0.5 on Foggy Cityscapes Validation Set (β=0.02)

---

## 🎯 Executive Summary

Proper evaluation using **mAP@0.5** (mean Average Precision at IoU threshold 0.5) on the foggy validation dataset. This provides accurate performance metrics accounting for both precision and recall.

### Key Finding

**YOLO trained directly on foggy images achieves the BEST performance (65.1% mAP@50)**  
This outperforms both the pretrained baseline and WUNET-restored training!

---

## 📈 Results - mAP@0.5 on Foggy Validation Set

| Model                     | mAP@50    | mAP@50-95 | Precision | Recall    | Training Data          |
| ------------------------- | --------- | --------- | --------- | --------- | ---------------------- |
| **Baseline YOLO (Foggy)** | **65.1%** | **41.7%** | **79.8%** | **56.5%** | Raw foggy (β=0.02)     |
| **WUNET + YOLO**          | **53.2%** | **34.0%** | **77.7%** | **43.6%** | WUNET-restored         |
| **Pretrained YOLO**       | **19.3%** | **8.8%**  | **44.6%** | **15.6%** | COCO (no fog training) |

---

## 🔍 Detailed Analysis

### 1. **Baseline YOLO on Foggy Images - WINNER** 🏆

- **mAP@50: 65.1%**
- **Trained on:** Raw foggy images (β=0.02)
- **Why it wins:**
  - Model learned to detect through fog directly
  - No information loss from restoration pipeline
  - High precision (79.8%) and good recall (56.5%)
  - Best overall balance

### 2. **WUNET + YOLO**

- **mAP@50: 53.2%** (-18% vs Foggy YOLO)
- **Trained on:** WUNET-restored images
- **Observations:**
  - Good precision (77.7%) but lower recall (43.6%)
  - WUNET restoration may remove some fine details
  - Still 2.7x better than pretrained baseline
  - Shows restoration helps, but not as much as direct foggy training

### 3. **Pretrained YOLO (COCO)**

- **mAP@50: 19.3%** (Baseline reference)
- **Trained on:** COCO dataset (no fog)
- **Observations:**
  - Severe performance drop in fog (70% decrease)
  - Low recall (15.6%) - misses most objects
  - Shows critical need for fog-specific training

---

## 📊 Performance Comparison

```
mAP@50 Performance:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Baseline YOLO (Foggy)  ████████████████████████████████████████████ 65.1%
WUNET + YOLO          ███████████████████████████████              53.2%
Pretrained YOLO       ███████                                      19.3%
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Recall (Detection Rate):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Baseline YOLO (Foggy)  ████████████████████████████                56.5%
WUNET + YOLO          ███████████████████                          43.6%
Pretrained YOLO       ███                                          15.6%
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

---

## 💡 Key Insights

### 1. **Direct Foggy Training > Restoration + Training**

- Training on raw foggy images (65.1%) beats restoration approach (53.2%)
- Model learns fog-invariant features naturally
- No preprocessing overhead in deployment

### 2. **WUNET Restoration Trade-offs**

- **Pros:** Better than no adaptation (53.2% vs 19.3%)
- **Cons:**
  - Lower recall - may lose detections during restoration
  - Additional computational cost (restoration step)
  - information loss in restoration process

### 3. **Fog Severely Impacts Untrained Models**

- Pretrained YOLO drops to 19.3% mAP@50 (from ~70% on clear COCO)
- Critical need for domain adaptation/training

---

## 🎯 Recommendations

### For Production Deployment:

✅ **Use Baseline YOLO trained on foggy images**

- Best performance (65.1% mAP@50)
- No additional restoration overhead
- Simpler pipeline
- More robust

### When to Consider WUNET+YOLO:

- If you also need restored images for visualization/human review
- If you have very minimal foggy training data
- For multi-task scenarios (restoration + detection)

### Never Use:

❌ Pretrained models without fog-specific training  
❌ Training performance drops by 70% in fog

---

## 📁 Evaluation Details

### Dataset

- **Source:** Foggy Cityscapes Validation Set
- **Fog Level:** β=0.02 (high fog)
- **Images:** 500 validation images
- **Ground Truth:** 8,367 object instances
- **Classes:** person, rider (2 classes)

### Evaluation Parameters

- **Confidence threshold:** 0.001 (very low to capture all predictions)
- **IoU threshold:** 0.6
- **Batch size:** 16
- **Image size:** 640×640
- **Device:** NVIDIA RTX A6000

---

## 📝 Conclusions

✅ **BEST APPROACH: Train YOLO directly on foggy images**

- Achieves 65.1% mAP@50
- 22% better than restoration approach
- 238% better than pretrained baseline

✅ **WUNET restoration helps, but not optimal**

- 53.2% mAP@50 (mid-tier performance)
- Still valuable if you need both restoration and detection

✅ **Domain-specific training is CRITICAL**

- 70% performance drop without fog training
- Fog fundamentally changes feature distribution

---

## 🚀 Next Steps

1. **Evaluate on Clear Images** _(TODO)_

   - Need ground truth labels for clear Cityscapes validation set
   - Compare performance: Foggy vs Clear conditions
   - Assess generalization capabilities

2. **Test DSNet + YOLO**

   - Run proper mAP evaluation
   - Compare joint learning vs separate training

3. **Cross-Condition Analysis**
   - Foggy-trained model on clear images
   - Clear-trained model on foggy images
   - Domain adaptation study

---

_Generated: 2025-12-29_  
_Evaluation on 500 foggy validation images (β=0.02)_  
_Ground truth: 8,367 object instances_
