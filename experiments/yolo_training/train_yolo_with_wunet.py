"""Restore a labeled YOLO dataset with WeatherUNet, then train a detector.

Run with --help for dataset, device, checkpoint, and output options. Importing
this module does not install packages, copy data, or start training.
"""

import argparse
from collections import defaultdict
from pathlib import Path
import sys

from PIL import Image
import torch
from tqdm import tqdm
import torchvision.transforms as transforms
import yaml

EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = EXPERIMENT_ROOT.parent
IMAGE_SUFFIXES = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.webp'}


def validate_checkpoint(path):
    """Reject missing weights and Git LFS pointer files before torch.load."""
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f'Checkpoint does not exist: {path}')
    with path.open('rb') as stream:
        if stream.read(100).startswith(b'version https://git-lfs.github.com/spec/v1'):
            raise ValueError(f'{path} is a Git LFS pointer. Run git lfs pull first.')
    return path


def load_dataset_config(dataset_root):
    """Validate the directory-based YOLO layout used by this pipeline.

    Every image requires an explicit label file; an intentionally empty label
    file represents a background image. Missing files must not silently become
    background annotations.
    """
    dataset_root = Path(dataset_root).expanduser().resolve()
    yaml_path = dataset_root / 'data.yaml'
    with yaml_path.open() as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict) or not isinstance(config.get('names'), (list, dict)) or not config['names']:
        raise ValueError(f'{yaml_path} must define a nonempty class names list or mapping.')
    names = config['names']
    if isinstance(names, dict) and set(names) != set(range(len(names))):
        raise ValueError('Class names must use consecutive integer IDs starting at zero.')
    if config.get('nc', len(names)) != len(names):
        raise ValueError('The nc field does not match the number of class names.')
    declared_root = Path(config.get('path') or dataset_root)
    if not declared_root.is_absolute():
        declared_root = yaml_path.parent / declared_root
    declared_root = declared_root.resolve()
    splits = {}
    for split in ('train', 'val', 'test'):
        value = config.get(split)
        if not value:
            if split != 'test':
                raise ValueError(f'{yaml_path} must define the {split} split.')
            continue
        if not isinstance(value, str):
            raise ValueError(f'{split}: only image directory paths are supported by this restoration workflow.')
        images_dir = Path(value)
        if not images_dir.is_absolute():
            images_dir = declared_root / images_dir
        images_dir = images_dir.resolve()
        expected = dataset_root / 'images' / split
        if images_dir != expected:
            raise ValueError(f'{split} must point to {expected}; received {images_dir}.')
        if not images_dir.is_dir():
            raise FileNotFoundError(f'Missing {split} images directory: {images_dir}')
        image_paths = sorted(p for p in images_dir.rglob('*') if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)
        if not image_paths:
            raise ValueError(f'No images found in {images_dir}')
        for image_path in image_paths:
            label_path = dataset_root / 'labels' / split / image_path.relative_to(images_dir).with_suffix('.txt')
            if not label_path.is_file():
                raise FileNotFoundError(f'Missing annotation for {image_path}: {label_path}')
            validate_label_file(label_path, len(names))
        splits[split] = image_paths
    return config, splits


def validate_label_file(label_path, class_count):
    for number, line in enumerate(Path(label_path).read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            values = [float(value) for value in line.split()]
            valid = (len(values) == 5 and values[0].is_integer()
                     and 0 <= values[0] < class_count
                     and all(0 <= value <= 1 for value in values[1:])
                     and all(value > 0 for value in values[3:]))
        except ValueError:
            valid = False
        if not valid:
            raise ValueError(f'Invalid YOLO box at {label_path}:{number}')


class WUNETImageRestorer:
    """Apply WeatherUNet using the same [-1, 1] RGB range as training."""

    def __init__(self, checkpoint_path, device=None, use_multi_gpu=True):
        self.device = torch.device(device or ('cuda:0' if torch.cuda.is_available() else 'cpu'))
        if self.device.type == 'cuda' and not torch.cuda.is_available():
            raise ValueError('CUDA was requested but is unavailable; select --gpus cpu.')
        checkpoint_path = validate_checkpoint(checkpoint_path)
        sys.path.insert(0, str(EXPERIMENT_ROOT / 'WUNET'))
        from weather_unet import WeatherUNet
        model = WeatherUNet(n_channels=3, n_classes=3, bilinear=True)
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
        state = checkpoint.get('model_state_dict', checkpoint)
        if not isinstance(state, dict) or not state:
            raise ValueError('Checkpoint does not contain a model state dictionary.')
        state = {key.removeprefix('module.'): value for key, value in state.items()}
        model.load_state_dict(state)
        # DataParallel requires its parameters on device_ids[0]. Use the selected
        # GPU rather than silently choosing GPU 0 or all GPUs during restoration.
        self.model = model.to(self.device).eval()
        if use_multi_gpu and self.device.type == 'cuda' and self.device.index in (None, 0) and torch.cuda.device_count() > 1:
            self.model = torch.nn.DataParallel(self.model)
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3),
        ])
        print(f'Loaded WUNET: {checkpoint_path}')
        for key in ('beta', 'val_loss', 'val_psnr'):
            if key in checkpoint:
                print(f'  {key}: {checkpoint[key]}')

    @staticmethod
    def inverse_transform(tensor):
        return (tensor + 1) / 2

    def restore_image(self, img_path, save_path=None):
        with Image.open(img_path) as source:
            image = source.convert('RGB')
        if min(image.size) < 16:
            raise ValueError(f'WeatherUNet images must be at least 16x16 pixels: {img_path}')
        with torch.inference_mode():
            restored = self.model(self.transform(image).unsqueeze(0).to(self.device))
            restored = self.inverse_transform(restored[0].cpu()).clamp(0, 1)
        result = transforms.ToPILImage()(restored)
        if save_path is not None:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            result.save(save_path)
        return result

    def restore_dataset(self, yolo_dataset_root, output_root, batch_size=4):
        if batch_size < 1:
            raise ValueError('Restoration batch size must be positive.')
        source_root = Path(yolo_dataset_root).expanduser().resolve()
        output_root = Path(output_root).expanduser().resolve()
        if source_root == output_root or source_root in output_root.parents or output_root in source_root.parents:
            raise ValueError('Source and restored datasets must not overlap.')
        config, splits = load_dataset_config(source_root)
        if output_root.exists() and any(output_root.iterdir()):
            raise FileExistsError(f'Restored dataset output must be empty: {output_root}')
        output_root.mkdir(parents=True, exist_ok=True)
        for split, paths in splits.items():
            # Group by dimensions: stacking differently sized tensors used to
            # crash on legitimate YOLO datasets containing mixed resolutions.
            groups = defaultdict(list)
            for path in paths:
                with Image.open(path) as image:
                    if min(image.size) < 16:
                        raise ValueError(f'WeatherUNet images must be at least 16x16 pixels: {path}')
                    groups[image.size].append(path)
            for group in groups.values():
                for offset in tqdm(range(0, len(group), batch_size), desc=f'Restore {split}'):
                    batch_paths = group[offset:offset + batch_size]
                    tensors = []
                    for path in batch_paths:
                        with Image.open(path) as image:
                            tensors.append(self.transform(image.convert('RGB')))
                    with torch.inference_mode():
                        batch = self.model(torch.stack(tensors).to(self.device))
                        batch = self.inverse_transform(batch.cpu()).clamp(0, 1)
                    for path, tensor in zip(batch_paths, batch):
                        relative = path.relative_to(source_root / 'images' / split)
                        destination = output_root / 'images' / split / relative
                        destination.parent.mkdir(parents=True, exist_ok=True)
                        transforms.ToPILImage()(tensor).save(destination)
                        label_relative = relative.with_suffix('.txt')
                        label_destination = output_root / 'labels' / split / label_relative
                        label_destination.parent.mkdir(parents=True, exist_ok=True)
                        label_destination.write_bytes((source_root / 'labels' / split / label_relative).read_bytes())
        config['path'] = str(output_root)
        config['nc'] = len(config['names'])
        for split in splits:
            config[split] = f'images/{split}'
        with (output_root / 'data.yaml').open('w') as stream:
            yaml.safe_dump(config, stream, sort_keys=False)
        print(f'Restored dataset saved to {output_root}')


def train_yolo(data_yaml, epochs=100, batch_size=16, img_size=640, model_name='yolov8n.pt',
               resume=None, gpus=None, project=None, name='wunet_yolo'):
    if min(epochs, batch_size, img_size) < 1:
        raise ValueError('Epochs, batch size, and image size must be positive.')
    data_yaml = Path(data_yaml).expanduser().resolve()
    load_dataset_config(data_yaml.parent)
    from ultralytics import YOLO
    if resume:
        validate_checkpoint(resume)
    model = YOLO(str(resume or model_name))
    results = model.train(
        data=str(data_yaml), epochs=epochs, batch=batch_size, imgsz=img_size,
        device=gpus or ('0' if torch.cuda.is_available() else 'cpu'),
        project=str(project or REPOSITORY_ROOT / 'outputs' / 'training'),
        name=name, exist_ok=False, resume=bool(resume), verbose=True,
    )
    return model, results


def main(argv=None, high_fog=False):
    parser = argparse.ArgumentParser(description='Restore a YOLO dataset with WUNET and train YOLO')
    default_dataset = 'yolo_dataset_beta0.02' if high_fog else 'yolo_dataset'
    parser.add_argument('--wunet-checkpoint', default=str(EXPERIMENT_ROOT / 'checkpoints/wunet_foggy/wunet_foggy_beta0.02_best.pth') if high_fog else None)
    parser.add_argument('--yolo-dataset', default=str(EXPERIMENT_ROOT / default_dataset))
    parser.add_argument('--restored-dataset', default=str(REPOSITORY_ROOT / 'outputs' / f'{default_dataset}_restored'))
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--restore-batch-size', type=int, default=4)
    parser.add_argument('--img-size', type=int, default=640)
    parser.add_argument('--model', default='yolov8n.pt', help='Detector checkpoint or model name')
    parser.add_argument('--resume', help='Checkpoint for resuming an interrupted detector run')
    parser.add_argument('--skip-restore', action='store_true')
    parser.add_argument('--gpus', default='0' if torch.cuda.is_available() else 'cpu', help='cpu, mps, or CUDA IDs such as 0 or 0,1')
    parser.add_argument('--project', default=str(REPOSITORY_ROOT / 'outputs' / 'training'))
    parser.add_argument('--name', default='wunet_yolo_beta0.02' if high_fog else 'wunet_yolo')
    args = parser.parse_args(argv)
    if not args.skip_restore:
        if not args.wunet_checkpoint:
            parser.error('--wunet-checkpoint is required unless --skip-restore is set')
        restoration_device = args.gpus if args.gpus in ('cpu', 'mps') else f'cuda:{args.gpus.split(",")[0]}'
        restorer = WUNETImageRestorer(args.wunet_checkpoint, device=restoration_device, use_multi_gpu=False)
        restorer.restore_dataset(args.yolo_dataset, args.restored_dataset, args.restore_batch_size)
    model, results = train_yolo(
        Path(args.restored_dataset) / 'data.yaml', epochs=args.epochs,
        batch_size=args.batch_size, img_size=args.img_size, model_name=args.model,
        resume=args.resume, gpus=args.gpus, project=args.project, name=args.name,
    )
    print(f'Training finished. Results: {getattr(results, "save_dir", args.project)}')
    return model, results


if __name__ == '__main__':
    main()
