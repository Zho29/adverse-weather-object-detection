"""Video inference with local YOLO, WeatherUNet, or custom DSNet checkpoints."""

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import sys
import time
import warnings

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def checkpoint_path(value):
    path = Path(value).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f'Checkpoint not found: {path}')
    with path.open('rb') as stream:
        if stream.read(100).startswith(b'version https://git-lfs.github.com/spec/v1'):
            raise ValueError(f'{path} is a Git LFS pointer; run git lfs pull to download the weights.')
    return path


def load_state(path):
    import torch
    checkpoint = torch.load(checkpoint_path(path), map_location='cpu', weights_only=True)
    if not isinstance(checkpoint, dict):
        raise ValueError('Expected a checkpoint dictionary or model state dictionary.')
    state = checkpoint.get('model_state_dict', checkpoint)
    if not isinstance(state, dict) or not state:
        raise ValueError('Checkpoint has no model state dictionary.')
    return checkpoint, {key.removeprefix('module.'): value for key, value in state.items()}


def normalized_frame(frame, size, device):
    import torch
    rgb = cv2.cvtColor(cv2.resize(frame, (size, size)), cv2.COLOR_BGR2RGB)
    array = np.ascontiguousarray(rgb.transpose(2, 0, 1))
    return torch.from_numpy(array).to(device=device, dtype=torch.float32).unsqueeze(0) / 127.5 - 1


def restored_frame(tensor, shape):
    array = tensor.detach().cpu()[0].permute(1, 2, 0).numpy()
    rgb = np.rint(np.clip((array + 1) / 2, 0, 1) * 255).astype(np.uint8)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    return cv2.resize(bgr, (shape[1], shape[0]))


def select_device(value):
    import torch
    value = f'cuda:{value}' if value.isdigit() else value
    device = torch.device(value)
    if device.type == 'cuda' and (not torch.cuda.is_available() or
                                  (device.index or 0) >= torch.cuda.device_count()):
        raise ValueError(f'CUDA device {value} is unavailable; use --device cpu.')
    if device.type == 'mps' and not torch.backends.mps.is_available():
        raise ValueError('MPS is unavailable; use --device cpu.')
    if device.type not in {'cpu', 'cuda', 'mps'}:
        raise ValueError('Use a cpu, cuda, or mps device.')
    return device


def names_mapping(names, count=None):
    if isinstance(names, (list, tuple)):
        names = dict(enumerate(names))
    if not isinstance(names, dict) or not names:
        raise ValueError('Model class names must be a nonempty list or mapping.')
    names = {int(key): str(value) for key, value in names.items()}
    if set(names) != set(range(count or len(names))):
        raise ValueError('Class names must match all consecutive model class IDs.')
    return names


class Predictor:
    def __init__(self, args):
        import torch
        self.args = args
        self.device = select_device(args.device)
        self.restorer = None
        if args.mode == 'dsnet':
            sys.path.insert(0, str(ROOT / 'experiments/DSNET'))
            from dsnet_yolo import DSNet_YOLO
            from detection import decode_detections
            checkpoint, state = load_state(args.checkpoint)
            head = state.get('detect_p3.weight')
            if head is None or head.ndim != 4 or head.shape[0] % 3 or head.shape[0] // 3 <= 5:
                raise ValueError('Checkpoint does not contain a valid DSNet detection head.')
            count = head.shape[0] // 3 - 5
            self.names = names_mapping(args.class_names or checkpoint.get('class_names') or
                                       [f'class_{index}' for index in range(count)], count)
            self.box_format = checkpoint.get('detection_box_format', 'legacy')
            if 'detection_box_format' not in checkpoint:
                warnings.warn('Historical DSNet checkpoint: its original loss did not supervise boxes '
                              'and classes correctly. Reliable detection requires retraining.', stacklevel=2)
            self.detector = DSNet_YOLO(num_classes=count, pretrained_cb=False)
            self.detector.load_state_dict(state, strict=True)
            self.detector.to(self.device).eval()
            self.decode = decode_detections
        else:
            from ultralytics import YOLO
            # Local files only: a typo must not silently download unrelated weights.
            self.detector = YOLO(str(checkpoint_path(args.weights)))
            original_names = names_mapping(self.detector.names)
            self.names = names_mapping(args.class_names, len(original_names)) if args.class_names else original_names
            if args.mode == 'wunet':
                sys.path.insert(0, str(ROOT / 'experiments/WUNET'))
                from weather_unet import WeatherUNet
                _, state = load_state(args.wunet)
                self.restorer = WeatherUNet(n_channels=3, n_classes=3, bilinear=True)
                self.restorer.load_state_dict(state, strict=True)
                self.restorer.to(self.device).eval()
        if args.classes is not None and not set(args.classes) <= set(self.names):
            raise ValueError('--classes contains IDs not present in this checkpoint.')

    def predict(self, frame):
        import torch
        args = self.args
        if args.mode == 'dsnet':
            with torch.inference_mode():
                image = normalized_frame(frame, args.restoration_size, self.device)
                restored, heads = self.detector(image)
                rows = self.decode(heads, self.detector.num_classes, image.shape[-2:],
                                   args.conf, args.iou, args.max_detections,
                                   self.box_format, classes=args.classes)[0].cpu().numpy()
            rows[:, [0, 2]] *= frame.shape[1] / image.shape[-1]
            rows[:, [1, 3]] *= frame.shape[0] / image.shape[-2]
            return restored_frame(restored, frame.shape), rows
        source = frame
        if self.restorer is not None:
            with torch.inference_mode():
                restored = self.restorer(normalized_frame(frame, args.restoration_size, self.device))
            source = restored_frame(restored, frame.shape)
        results = self.detector.predict(source=source, device=str(self.device), imgsz=args.img_size,
                                        conf=args.conf, iou=args.iou, classes=args.classes,
                                        max_det=args.max_detections, verbose=False, save=False)
        return source, results[0].boxes.data.detach().cpu().numpy()


def detection_records(rows, names, shape):
    rows = np.asarray(rows, dtype=float)
    if rows.size == 0:
        return []
    if rows.ndim != 2 or rows.shape[1] != 6 or not np.isfinite(rows).all():
        raise ValueError('Predictor must return finite Nx6 [xyxy, score, class] detections.')
    records = []
    for x1, y1, x2, y2, score, label in rows:
        if not label.is_integer() or int(label) not in names or not 0 <= score <= 1:
            raise ValueError('Predictor returned an invalid confidence or class ID.')
        x1, x2 = np.clip([x1, x2], 0, shape[1])
        y1, y2 = np.clip([y1, y2], 0, shape[0])
        if x2 <= x1 or y2 <= y1:
            continue
        records.append({'xyxy': [float(x1), float(y1), float(x2), float(y2)],
                        'confidence': float(score), 'class_id': int(label),
                        'class_name': names[int(label)]})
    return records


def annotate(frame, records):
    result = frame.copy()
    for record in records:
        x1, y1, x2, y2 = (int(round(value)) for value in record['xyxy'])
        cv2.rectangle(result, (x1, y1), (x2, y2), (70, 220, 100), 2)
        text = f"{record['class_name']} {record['confidence']:.2f}"
        cv2.putText(result, text, (x1, max(14, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (70, 220, 100), 1, cv2.LINE_AA)
    return result


def run_video(args, predictor=None):
    input_path = Path(args.input).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    summary_path = Path(args.summary).expanduser().resolve() if args.summary else output_path.with_suffix('.json')
    detections_path = (Path(args.detections).expanduser().resolve() if args.detections
                       else output_path.with_suffix('.jsonl'))
    outputs = [output_path, summary_path, detections_path]
    protected = {input_path}
    for attribute in ('weights', 'wunet', 'checkpoint'):
        value = getattr(args, attribute, None)
        if value:
            protected.add(Path(value).expanduser().resolve())
    if len(set(outputs)) != 3 or any(path in protected for path in outputs):
        raise ValueError('Input, checkpoints, video, summary, and detection paths must be distinct.')
    if any(left in right.parents or right in left.parents
           for index, left in enumerate(outputs) for right in outputs[index + 1:]):
        raise ValueError('Output paths cannot contain one another as parent directories.')
    if output_path.suffix.lower() not in {'.mp4', '.avi'}:
        raise ValueError('Output video must use .mp4 or .avi.')
    if not input_path.is_file():
        raise FileNotFoundError(f'Input video not found: {input_path}')
    for path in outputs:
        if path.exists():
            raise FileExistsError(f'Output already exists; choose another path: {path}')
    cap = cv2.VideoCapture(str(input_path))
    writer = None
    created = []
    try:
        if not cap.isOpened():
            raise ValueError(f'Cannot open video: {input_path}')
        fps = args.fps if args.fps is not None else cap.get(cv2.CAP_PROP_FPS)
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError('Video has no valid frame rate; supply --fps.')
        ok, frame = cap.read()
        if not ok or frame is None:
            raise ValueError('Input video contains no readable frames.')
        predictor = predictor or Predictor(args)
        names = names_mapping(predictor.names)
        height, width = frame.shape[:2]
        # Common video codecs require even dimensions; pad rather than distort boxes.
        output_width, output_height = width + width % 2, height + height % 2
        for path in outputs:
            path.parent.mkdir(parents=True, exist_ok=True)
        created.append(output_path)
        codec = 'mp4v' if output_path.suffix.lower() == '.mp4' else 'MJPG'
        writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*codec), fps,
                                 (output_width, output_height))
        if not writer.isOpened():
            raise ValueError(f'Cannot create output video using {codec}: {output_path}')
        frames = 0
        counts = Counter()
        start = time.perf_counter()
        with detections_path.open('x') as stream:
            created.append(detections_path)
            while ok and (args.max_frames is None or frames < args.max_frames):
                if frame.shape[:2] != (height, width):
                    raise ValueError('Video frame dimensions changed during decoding.')
                display, rows = predictor.predict(frame)
                if display.shape != frame.shape or display.dtype != np.uint8:
                    raise ValueError('Predictor must return a uint8 BGR image matching the input size.')
                records = detection_records(rows, names, frame.shape)
                counts.update(record['class_name'] for record in records)
                rendered = annotate(display, records)
                rendered = cv2.copyMakeBorder(rendered, 0, output_height - height,
                                               0, output_width - width, cv2.BORDER_CONSTANT)
                writer.write(rendered)
                stream.write(json.dumps({'frame_index': frames, 'timestamp_seconds': frames / fps,
                                         'detections': records}, allow_nan=False) + '\n')
                frames += 1
                if args.max_frames is not None and frames >= args.max_frames:
                    break
                ok, frame = cap.read()
        elapsed = time.perf_counter() - start
        writer.release()
        writer = None
        summary = {'mode': args.mode, 'input': str(input_path), 'output': str(output_path),
                   'detections_file': str(detections_path), 'frames_processed': frames,
                   'source_fps': fps, 'processing_fps': frames / elapsed if elapsed else None,
                   'source_size': [width, height], 'output_size': [output_width, output_height],
                   'detections': sum(counts.values()), 'detections_by_class': dict(counts),
                   'class_names': names, 'note': 'Counts are frame detections, not unique objects or mAP.'}
        with summary_path.open('x') as stream:
            created.append(summary_path)
            json.dump(summary, stream, indent=2, allow_nan=False)
            stream.write('\n')
        return summary
    except BaseException:
        if writer is not None:
            writer.release()
            writer = None
        for path in created:
            path.unlink(missing_ok=True)
        raise
    finally:
        cap.release()
        if writer is not None:
            writer.release()


def parse_args(argv=None, default_mode=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['baseline', 'wunet', 'dsnet'], default=default_mode or 'baseline')
    parser.add_argument('--input', required=True, help='Local input video')
    parser.add_argument('--output', required=True, help='New .mp4 or .avi file')
    parser.add_argument('--weights', help='Local Ultralytics detector checkpoint')
    parser.add_argument('--wunet', default=str(ROOT / 'experiments/checkpoints/wunet_foggy/wunet_foggy_beta0.02_best.pth'))
    parser.add_argument('--checkpoint', default=str(ROOT / 'experiments/checkpoints/dsnet_yolo_proper_loss_100ep/dsnet_yolo_best.pth'))
    parser.add_argument('--device', default='cpu', help='cpu, cuda:0, mps, or a CUDA device number')
    parser.add_argument('--img-size', type=int, default=640)
    parser.add_argument('--restoration-size', type=int, default=256, help='Square restoration input size, multiple of 16')
    parser.add_argument('--conf', type=float, default=0.25)
    parser.add_argument('--iou', type=float, default=0.45)
    parser.add_argument('--max-detections', type=int, default=300)
    parser.add_argument('--classes', nargs='+', type=int, help='Keep these zero-based checkpoint class IDs')
    parser.add_argument('--class-names', nargs='+', help='Names in checkpoint class-ID order')
    parser.add_argument('--max-frames', type=int, help='Stop after this many frames')
    parser.add_argument('--fps', type=float, help='Override missing or incorrect source FPS')
    parser.add_argument('--summary', help='Summary JSON (default: output stem + .json)')
    parser.add_argument('--detections', help='Per-frame JSONL (default: output stem + .jsonl)')
    args = parser.parse_args(argv)
    if args.weights is None:
        run = 'wunet_yolo_beta0.02' if args.mode == 'wunet' else 'baseline_foggy_beta0.02'
        args.weights = str(ROOT / 'experiments/yolo_training/runs' / run / 'weights/best.pt')
    if not 0 <= args.conf <= 1 or not 0 <= args.iou <= 1:
        parser.error('--conf and --iou must be in [0, 1].')
    if args.img_size < 32 or args.restoration_size < 16 or args.restoration_size % 16:
        parser.error('--img-size must be >=32; --restoration-size must be a multiple of 16 and >=16.')
    if args.max_detections < 1 or (args.max_frames is not None and args.max_frames < 1):
        parser.error('--max-detections and --max-frames must be positive.')
    if args.fps is not None and (not math.isfinite(args.fps) or args.fps <= 0):
        parser.error('--fps must be finite and positive.')
    return args


def main(argv=None, default_mode=None):
    args = parse_args(argv, default_mode)
    try:
        summary = run_video(args)
    except (OSError, ValueError, RuntimeError, KeyError) as error:
        raise SystemExit(f'Video inference failed: {error}') from error
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
