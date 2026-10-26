# Adverse Weather Object Detection

Research experiments comparing direct object detection, image restoration, and joint restoration–detection models on Foggy Cityscapes.

The current experiments focus on **fog**, including the high-density setting **β = 0.02**. The repository contains WeatherUNet, DSNet with YOLO/DINO, saved checkpoints, and evaluation artifacts.

## Approaches

| Approach | Pipeline | Implementation |
| --- | --- | --- |
| Baseline YOLO | Foggy image → YOLOv8 | [Baseline training](experiments/yolo_training/train_baseline_yolo_foggy.py) |
| WeatherUNet + YOLO | Foggy image → restoration → YOLOv8 | [WeatherUNet](experiments/WUNET/) · [Detector training](experiments/yolo_training/train_yolo_with_wunet.py) |
| DSNet + custom grid head | Shared restoration features and a YOLO-style research head | [Model](experiments/DSNET/dsnet_yolo.py) |
| DSNet + DINO | Joint restoration and DINO-based detection experiments | [Model](experiments/DSNET/dsnet_dino.py) |

## Getting started

Use a Python 3.10+ environment. Install a PyTorch build appropriate for your hardware before the remaining dependencies if you need CUDA.

```bash
git clone https://github.com/Zho29/adverse-weather-object-detection.git
cd adverse-weather-object-detection
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Weights use Git LFS. If you need the saved checkpoints, install Git LFS and retrieve them:

```bash
git lfs install
git lfs pull
```

Datasets are acquired separately. Place paired foggy and clear Cityscapes images in this layout:

```text
/path/to/Foggy_Cityscapes_dataset/
├── leftImg8bit/
│   ├── train/<city>/*_leftImg8bit.png
│   └── val/<city>/*_leftImg8bit.png
└── leftImg8bit_foggy/
    ├── train/<city>/*_leftImg8bit_foggy_beta_0.02.png
    └── val/<city>/*_leftImg8bit_foggy_beta_0.02.png
```

Train WeatherUNet from the repository root:

```bash
python experiments/WUNET/train_wunet_foggy.py \
  --root-dir /path/to/Foggy_Cityscapes_dataset \
  --beta 0.02 \
  --epochs 200 \
  --save-dir outputs/wunet_foggy
```

The WeatherUNet script selects CUDA when available and otherwise uses CPU. See the [setup guide](docs/setup.md) for detector training, checkpoint handling, and the additional DINO requirements.

## Video detection

With the model weights downloaded, run the baseline on a local video:

```bash
python video_detection_analysis/process_video_baseline_yolo.py \
  --input /path/to/video.mp4 --output outputs/video/baseline.mp4 --device cpu
```

The [video guide](video_detection_analysis/README.md) covers WeatherUNet + YOLO, DSNet, class IDs, and output files. Each mode writes an annotated video, per-frame detection JSONL, and summary JSON. New video runs require local checkpoints; they do not download weights automatically.

## Recorded results

The following mAP@50 values come from the saved evaluation JSON files. They are historical measurements, not newly reproduced benchmarks.

| Model | Clear images | Foggy images, β = 0.02 |
| --- | ---: | ---: |
| COCO-pretrained YOLOv8n | 25.1% | 19.3% |
| YOLO trained on foggy images | 57.6% | 65.1% |
| YOLO trained on WUNET-restored images | 66.0% | 53.2% |

Sources: [clear-image metrics](experiments/CLEAR_IMAGE_mAP_BASELINE.json) and [foggy-image metrics](experiments/foggy_image_map_results.json). Class mappings and other reports differ across experiments; read [results and limitations](docs/results.md) before interpreting these comparisons.

## Repository layout

```text
├── experiments/
│   ├── WUNET/                 # Restoration model and paired-image training
│   ├── DSNET/                 # Joint models, training, and data conversion
│   ├── yolo_training/         # Detector training and saved runs
│   ├── evaluation/            # Detection visualization
│   ├── checkpoints/           # Git LFS weights and training histories
│   ├── clear_image_comparison/ # Saved qualitative comparisons
│   └── *.py / *.json          # Evaluation scripts and recorded metrics
├── docs/
│   ├── setup.md               # Environment and workflow notes
│   ├── results.md             # Metric provenance and known limitations
│   └── archive/               # Original experiment reports
├── yolo_training/runs/        # Additional historical training artifacts
├── video_detection_analysis/ # Baseline, WeatherUNet, and DSNet video CLIs
├── tests/                    # Synthetic regression tests
├── .github/                  # CI and contribution templates
└── requirements.txt          # Core Python dependencies
```

## Workflow support

| Workflow | Entry point | Status |
| --- | --- | --- |
| Restoration training | `experiments/WUNET/train_wunet_foggy.py` | CLI with checkpoint resume |
| Baseline detector training | `experiments/yolo_training/train_baseline_yolo_foggy.py` | Validated standard YOLO datasets |
| Restoration + detector training | `experiments/yolo_training/train_yolo_with_wunet.py` | Standard YOLO datasets; explicit restoration stage |
| DSNet training and evaluation | `experiments/DSNET/train_dsnet_yolo.py` · `evaluate_dsnet_yolo_detection.py` | Custom detection head; use newly trained checkpoints |
| Video inference | `video_detection_analysis/process_video_*.py` | Baseline, WeatherUNet, and DSNet |
| DINO and older comparison scripts | Other experiment scripts | Historical workflows; environment/path review required |

Use `--help` on the listed CLIs. New training outputs belong in `outputs/`; saved research artifacts remain under `experiments/`. Fresh WeatherUNet and DSNet runs require empty save directories to protect previous results.

## Research status

WeatherUNet, YOLO training, and DSNet + YOLO now have dataset CLIs and regression tests. Some older top-level evaluation scripts and DINO experiments retain server-specific settings. DINO requires an additional MMDetection environment. Historical DSNet checkpoints were trained without meaningful box/class supervision; the corrected training loss requires retraining before detection quality can be assessed.

Original experiment reports are preserved in [docs/archive](docs/archive/). Research papers are stored alongside the [WeatherUNet](experiments/WUNET/2407.02581v1.pdf) and [DSNet](experiments/DSNET/DSNet_Joint_Semantic_Learning_for_Object_Detection_in_Inclement_Weather_Conditions.pdf) code for reference. Synthetic tests cover code behavior; full benchmark reproduction and paper equivalence have not been established.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for checks and experiment-reporting expectations. There is currently no repository-wide license; datasets, papers, model weights, and dependencies may have separate terms.
