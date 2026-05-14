# Adverse Weather Object Detection

End-to-end research project: object detection in foggy/adverse-weather conditions using image restoration + YOLOv8.

## Repository Structure

```
├── new_project/
│   ├── WUNET/                        # WeatherUNet model (dehazing)
│   ├── DSNET/                        # DSNet + YOLO / DINO joint models
│   ├── checkpoints/                  # Trained model weights (Git LFS)
│   ├── evaluation/                   # Evaluation scripts & results
│   ├── runs/                         # YOLO training runs & metrics
│   └── *.py / *.md / *.sh           # Training, evaluation, visualization scripts
│
├── video_detection_analysis/         # Video inference pipeline
│   ├── process_video_baseline_yolo.py
│   ├── process_video_wunet_yolo.py
│   ├── process_video_dsnet_yolo.py
│   ├── compare_models.py
│   ├── calculate_metrics.py
│   ├── run_all_models.sh
│   └── comparison_samples/          # Sample output frames
│
└── yolo_training/
    └── runs/                        # WUNET+YOLO training results & curves
```

> **Note:** Large datasets (~100 GB total), virtual environments, and framework clones (mmdetection, mmcv) are excluded via `.gitignore`. Model checkpoints are stored via **Git LFS**.

---

## Models

| Model | Description | Code |
|---|---|---|
| **Baseline YOLO** | YOLOv8n directly on foggy frames | `video_detection_analysis/process_video_baseline_yolo.py` |
| **WUNET + YOLO** | WeatherUNet dehazing → YOLOv8 detection | `new_project/WUNET/` + video pipeline |
| **DSNet + YOLO** | Joint dehazing + detection (single forward pass) | `new_project/DSNET/dsnet_yolo.py` |
| **DSNet + DINO** | Joint dehazing + DINO feature detection | `new_project/DSNET/dsnet_dino.py` |

---

## Quick Start

### Environment Setup
```bash
python -m venv venv
source venv/bin/activate
pip install torch torchvision opencv-python ultralytics numpy scikit-image
```

### Train WUNET
```bash
cd new_project/WUNET
python train_wunet_foggy.py
```

### Train DSNet + YOLO
```bash
cd new_project/DSNET
python train_dsnet_yolo.py
```

### Run Video Inference
```bash
cd video_detection_analysis
cp /path/to/your/video.mp4 input_video.mp4

# Baseline
python process_video_baseline_yolo.py --input input_video.mp4

# WUNET + YOLO
python process_video_wunet_yolo.py \
  --wunet ../new_project/checkpoints/wunet_foggy/wunet_foggy_beta0.02_epoch200.pth \
  --wunet-src ../new_project/WUNET

# DSNet + YOLO
python process_video_dsnet_yolo.py \
  --checkpoint ../new_project/checkpoints/dsnet_yolo_proper_loss_5ep/dsnet_yolo_best.pth \
  --dsnet-src ../new_project/DSNET
```

---

## Dataset

**Foggy Cityscapes** (Sakaridis et al.) — high-fog setting β=0.02.

Detection classes: **person** and **vehicle** (mapped from Cityscapes fine categories).

---

## Results

See `new_project/README_Foggy_Detection_Comparison.md` and `new_project/BASELINE_RESULTS_mAP.md` for detailed mAP results.

Sample comparison frames are in `video_detection_analysis/comparison_samples/`.
