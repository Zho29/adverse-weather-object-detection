# Video detection

Repository-owned video inference for baseline YOLO, WeatherUNet + YOLO, and DSNet. This replaces the empty Git link previously recorded at commit `f6458f926dd899002e5f752ef8ff446cb9169a2c`; its original source was not present in the checkout.

Install the root [requirements](../requirements.txt), download weights with Git LFS, and run commands from the repository root. Input videos are acquired separately. Output paths must be new `.mp4` or `.avi` files.

## Baseline YOLO

```bash
python video_detection_analysis/process_video_baseline_yolo.py \
  --input /path/to/video.mp4 \
  --weights experiments/yolo_training/runs/baseline_foggy_beta0.02/weights/best.pt \
  --output outputs/video/baseline.mp4 --device cpu
```

## WeatherUNet + YOLO

```bash
python video_detection_analysis/process_video_wunet_yolo.py \
  --input /path/to/video.mp4 \
  --wunet experiments/checkpoints/wunet_foggy/wunet_foggy_beta0.02_best.pth \
  --weights experiments/yolo_training/runs/wunet_yolo_beta0.02/weights/best.pt \
  --output outputs/video/wunet.mp4 --device cpu
```

WeatherUNet restores RGB frames in the same `[-1, 1]` range as training. Restored images are converted back to BGR pixels before detector inference. The output video shows detections on the restored frames.

## DSNet

```bash
python video_detection_analysis/process_video_dsnet_yolo.py \
  --input /path/to/video.mp4 \
  --checkpoint /path/to/retrained/dsnet_yolo_best.pth \
  --output outputs/video/dsnet.mp4 --device cpu
```

The custom head is decoded with its recorded box format and class-aware NMS. New checkpoints include class names. If older checkpoints lack names, labels default to `class_0`, `class_1`, and so on; provide `--class-names person vehicle` only after checking the actual training mapping. Original checkpoint heads did not receive meaningful box/class supervision, so detection quality requires retraining with the corrected loss.

## Options and outputs

Use `--help` on any script. Class filters are applied before NMS and the maximum detection limit, so excluded classes cannot displace requested detections. Shared options include `--device cpu`, `--device cuda:0`, `--device mps`, `--conf 0.25`, `--iou 0.45`, `--classes 0 1`, `--max-frames 100`, and `--max-detections 300`.

Restoration uses a square input size of 256 by default (`--restoration-size`, multiple of 16). Coordinates are scaled back to the source dimensions. Source FPS is preserved; use `--fps` if the video has missing or incorrect metadata. Odd frame dimensions are padded to even output dimensions for codec compatibility.

For `outputs/video/baseline.mp4`, successful inference creates:

- `baseline.mp4`: annotated video.
- `baseline.jsonl`: detections for every processed frame, including frames with no detections. Boxes use `xyxy` source-image pixels; timestamps are zero-based `frame_index / fps`.
- `baseline.json`: frame count, video dimensions, frame rate, processing speed, class names, and detection counts.

Override the JSON paths with `--detections` and `--summary`. Counts represent detections across frames, not unique tracked objects, accuracy, or mAP. Existing files are never overwritten. Failed inference closes the video handles and removes partial files created by that run. All weights must already exist locally; Git LFS pointers receive an actionable error.

The pipeline is checked with synthetic video and local randomly initialized checkpoints for all three modes. Real trained-model performance and long-video throughput have not been benchmarked.
