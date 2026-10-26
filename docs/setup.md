# Setup and workflows

Run commands from the repository root unless a command explicitly changes directories. Core dependencies are listed in [requirements.txt](../requirements.txt); the original environment was not locked, so installing them does not guarantee historical checkpoint compatibility.

## WeatherUNet

The [Foggy Cityscapes loader](../experiments/WUNET/foggy_dataset.py) pairs images by city and basename. Both clear and foggy train/validation images are required. The dataset is not bundled.

```bash
python experiments/WUNET/train_wunet_foggy.py --help
python experiments/WUNET/train_wunet_foggy.py \
  --root-dir /path/to/Foggy_Cityscapes_dataset \
  --beta 0.02 --epochs 200 --batch-size 24 \
  --save-dir outputs/wunet_foggy
```

Use a smaller batch size if memory is limited. The script supports `--resume`, `--num-workers`, and `--multi-gpu`. Its default dataset path resolves relative to the source directory; checkpoints default to the ignored root `outputs/wunet_foggy/` directory. Fresh runs require an empty save directory. To continue an existing WeatherUNet run, pass `--resume /path/to/checkpoint.pth` and set `--epochs` to the total desired epoch count.

## YOLO workflows

A YOLO dataset needs images, normalized bounding-box labels, and a `data.yaml` specifying splits and class names. Verify that class IDs agree with the annotations and checkpoint. The archived code uses both `person/vehicle` and `person/rider`; these mappings are not interchangeable.

The [baseline training script](../experiments/yolo_training/train_baseline_yolo_foggy.py) accepts an existing validated dataset:

```bash
python experiments/yolo_training/train_baseline_yolo_foggy.py \
  --data /path/to/foggy_yolo_dataset/data.yaml --device cpu
```

To prepare a raw-fog dataset from paired labels without starting training:

```bash
python experiments/yolo_training/train_baseline_yolo_foggy.py \
  --foggy-root /path/to/leftImg8bit_foggy \
  --labels-dataset /path/to/source_yolo_dataset \
  --output-dataset outputs/baseline_dataset --prepare-only
```

Preparation preserves class names and renames annotations to match image stems. Every image must have an explicit label file; an empty file represents a known background image. Missing annotations fail rather than becoming background labels.

For the restoration-then-detection pipeline, supply explicit dataset locations and select an available device:

```bash
python experiments/yolo_training/train_yolo_with_wunet.py \
  --wunet-checkpoint experiments/checkpoints/wunet_foggy/wunet_foggy_beta0.02_best.pth \
  --yolo-dataset /path/to/foggy_yolo_dataset \
  --restored-dataset /path/to/restored_yolo_dataset \
  --gpus 0
```

The input dataset must already contain images, labels, and a YAML configuration in the standard `images/{train,val,test}` / `labels/{train,val,test}` directory layout. Mixed image sizes and nested city directories are supported. The restored output directory must be empty. Use `--skip-restore` to train on an existing restored dataset, including when resuming with `--resume /path/to/last.pt`. For CPU training, use `--gpus cpu`.

## DSNet and DINO

[DSNet + YOLO](../experiments/DSNET/train_dsnet_yolo.py) uses paired images and COCO-format `coco_annotations/cityscapes_{train,val}.json` under the supplied dataset root:

```bash
python experiments/DSNET/train_dsnet_yolo.py \
  --dataset-root /path/to/Foggy_Cityscapes_dataset \
  --save-dir outputs/dsnet --epochs 100 --batch-size 2
```

Fresh DSNet runs require an empty save directory and default to `outputs/dsnet_yolo/`. Use `--no-pretrained` to initialize the content branch without downloading ResNet weights. The corrected custom detection loss supervises boxes, objectness, and classes. New checkpoints record class names and box decoding metadata. Original checkpoints remain loadable, but their untrained box/class channels cannot be repaired by decoding alone; retraining is needed.

The DSNet [evaluation CLI](../experiments/DSNET/evaluate_dsnet_yolo_detection.py) and [visualization CLI](../experiments/DSNET/visualize_dsnet_yolo.py) accept explicit paths; use `--help` for options. Other DSNet/DINO research variants still need configuration.

DINO variants import MMDetection and require compatible PyTorch, CUDA, MMCV, MMEngine, and MMDetection versions, framework configs, and pretrained weights. These are not installed by the core requirements. The historical `install_mmdetection.sh` is preserved as a record of the original setup; it contains an absolute virtual-environment path and should be reviewed before use. The standalone `dino_config.py` also references base configs that are not included here. `dsnet_dino_features.py` is an empty placeholder.

## Checkpoints and artifacts

`.pt`, `.pth`, `.onnx`, and `.weights` files use Git LFS. A roughly 130-byte text file beginning with `version https://git-lfs.github.com/spec/v1` is a pointer, not a downloaded model. Run `git lfs pull` to retrieve available weights before loading them.

Keep new temporary runs in the ignored root `outputs/` directory. Existing checkpoints, curves, JSON metrics, and training logs are retained for provenance. The additional root `yolo_training/runs/` directory is also historical; its relationship to individual experiments has not been independently verified.

## Video pipeline

See the [video guide](../video_detection_analysis/README.md) for baseline, WeatherUNet, and DSNet inference. The empty historical Git link has been replaced with repository-owned source. Synthetic videos and randomly initialized local model checkpoints exercise each pipeline; trained-weight detection quality still needs evaluation on actual data.

## Local checks

```bash
python -m compileall -q experiments video_detection_analysis tests
python -m unittest discover -s tests -v
bash -n experiments/create_yolo_dataset_beta0.02.sh
bash -n experiments/fix_labels_beta0.02.sh
bash -n experiments/install_mmdetection.sh
```

The test suite uses synthetic data and exercises training state, labels, decoding, normalization, and video I/O. Full training and benchmark evaluation require real datasets, downloaded checkpoints, and the appropriate framework environment.

## Choosing the right script

The commands shown here are maintained workflows. Files such as `experiments/evaluate_clear_map.py`, `evaluate_on_clear_images.py`, and the top-level `test_*_on_clear.py` scripts are historical experiment records with server-specific paths. They are not the regression tests and should not be launched as a general evaluation suite.

Run regression tests only from `tests/` using the command above. A zero-error syntax check does not validate class mappings, checkpoint quality, or historical benchmark numbers. In particular, DSNet's custom grid head is not the Ultralytics YOLOv8 head; its checkpoints and losses are not interchangeable with standard YOLO weights.
