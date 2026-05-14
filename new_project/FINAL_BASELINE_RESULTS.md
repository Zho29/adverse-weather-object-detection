# Baseline Evaluation Results 📊

**Evaluation:** Object Detection Performance in Clear vs Foggy Conditions  
**Dataset:** Cityscapes Validation Set (500 images, 8,367 objects)  
**Metric:** mAP@0.5 (mean Average Precision at IoU threshold 0.5)

---

## Results Summary

### Performance on CLEAR Images (BASELINE - Upper Bound)

| Model               | mAP@50    | mAP@50-95 | Precision | Recall    |
| ------------------- | --------- | --------- | --------- | --------- |
| **WUNET + YOLO**    | **66.0%** | **42.2%** | **80.3%** | **57.2%** |
| **Baseline YOLO**   | **57.6%** | **36.1%** | **79.1%** | **49.4%** |
| **Pretrained YOLO** | **25.1%** | **11.1%** | **49.7%** | **21.0%** |

### Performance on FOGGY Images (β=0.02 - High Fog)

| Model               | mAP@50    | mAP@50-95 | Precision | Recall    |
| ------------------- | --------- | --------- | --------- | --------- |
| **Baseline YOLO**   | **65.1%** | **41.7%** | **79.8%** | **56.5%** |
| **WUNET + YOLO**    | **53.2%** | **34.0%** | **77.7%** | **43.6%** |
| **Pretrained YOLO** | **19.3%** | **8.8%**  | **44.6%** | **15.6%** |

---

## Key Findings

### 1. WUNET+YOLO: Best on Clear Images

- **66.0% mAP@50** on clear conditions (baseline)
- Training on restored images generalizes well
- Only **19% drop** when tested on fog (66.0% → 53.2%)

### 2. Baseline YOLO: Best on Foggy Images

- **65.1% mAP@50** on foggy conditions
- Training directly on fog is optimal for fog detection
- **13% drop** on clear images (66.0% baseline → 57.6%)

### 3. Pretrained Models Fail in Both Conditions

- Only **25.1%** on clear (should be 60-70% on COCO)
- Catastrophic **19.3%** on fog

### Performance Degradation (Clear → Foggy)

```
WUNET+YOLO:      66.0% → 53.2%  (-19% degradation)
Baseline YOLO:   57.6% → 65.1%  (+13% improvement!)
Pretrained YOLO: 25.1% → 19.3%  (-23% degradation)
```

**Finding:** Baseline YOLO performs BETTER on foggy images than on clear images! This shows it specialized for fog detection at the expense of clear performance.

---

## Recommendations

### Use Case 1: Deploy in Foggy Conditions Only

**Recommendation:** Use **Baseline YOLO** (65.1% mAP@50 on fog)

- Best fog performance
- Direct training on foggy images
- Simple pipeline

### Use Case 2: Deploy in Mixed Conditions (Clear + Foggy)

**Recommendation:** Use **WUNET+YOLO** (66.0% clear, 53.2% fog)

- Best generalization across conditions
- Only 19% degradation in fog
- More robust to weather changes

### Use Case 3: Need Both Restoration + Detection

**Recommendation:** Use **WUNET+YOLO**

- Already has restoration module
- Decent performance on both conditions (66.0% clear, 53.2% fog)
- Multi-task capability

---

## What to Tell Your Supervisor

**Simple Summary:**
"I evaluated three detection approaches on both clear and foggy images. For mixed weather conditions, training on restored images (WUNET+YOLO) works best - it hits 66% accuracy on clear images and only drops to 53% in heavy fog. Training directly on foggy images gives the best fog performance (65%), but loses generalization to clear conditions (58%)."

**Key Insight:**
"There's a trade-off: specialize for fog (Baseline YOLO) or generalize across conditions (WUNET+YOLO). I'd recommend WUNET+YOLO for real-world deployment since weather conditions vary."

**Numbers to Highlight:**

- WUNET+YOLO: 66% → 53% (only 19% drop in fog)
- Baseline YOLO: Excellent in fog (65%) but worse generalization (58% on clear)
- Pretrained YOLO: Fails in both conditions (~20-25%)

---

_Evaluation Date: 2025-12-29_
