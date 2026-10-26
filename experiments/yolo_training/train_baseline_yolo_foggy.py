#!/usr/bin/env python3
"""Prepare or train a baseline YOLO detector on raw foggy images.

Dataset preparation preserves source class names and renames annotations to
match the foggy images. It never creates empty labels for missing annotations.
"""

import argparse
from pathlib import Path
import shutil

import torch
import yaml

from fix_baseline_labels import label_plan
from train_yolo_with_wunet import load_dataset_config, validate_checkpoint, validate_label_file

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def prepare_baseline_dataset(foggy_root, labels_dataset, output_root, beta=0.02,
                             source_beta=None, splits=('train', 'val')):
    labels_dataset = Path(labels_dataset).expanduser().resolve()
    output_root = Path(output_root).expanduser().resolve()
    foggy_root = Path(foggy_root).expanduser().resolve()
    if output_root == labels_dataset or output_root == foggy_root or output_root in labels_dataset.parents or output_root in foggy_root.parents:
        raise ValueError('Baseline output must be separate from source datasets.')
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f'Baseline output directory must be empty: {output_root}')
    with (labels_dataset / 'data.yaml').open() as stream:
        source_config = yaml.safe_load(stream)
    if not isinstance(source_config, dict) or not isinstance(source_config.get('names'), (dict, list)) or not source_config['names']:
        raise ValueError('The source data.yaml must declare class names.')
    names = source_config['names']
    if isinstance(names, dict) and set(names) != set(range(len(names))):
        raise ValueError('Source class IDs must be consecutive integers starting at zero.')
    if source_config.get('nc', len(names)) != len(names):
        raise ValueError('Source nc does not match the number of class names.')
    plan = label_plan(foggy_root, labels_dataset / 'labels', beta, source_beta, splits)
    for _, _, annotation in plan:
        validate_label_file(annotation, len(names))
    output_root.mkdir(parents=True, exist_ok=True)
    for split, image, annotation in plan:
        image_destination = output_root / 'images' / split / image.name
        label_destination = output_root / 'labels' / split / image.with_suffix('.txt').name
        image_destination.parent.mkdir(parents=True, exist_ok=True)
        label_destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(image, image_destination)
        shutil.copy2(annotation, label_destination)
    config = {'path': str(output_root), 'nc': len(names), 'names': names}
    config.update({split: f'images/{split}' for split in splits})
    config['fog_density'] = beta
    yaml_path = output_root / 'data.yaml'
    with yaml_path.open('w') as stream:
        yaml.safe_dump(config, stream, sort_keys=False)
    load_dataset_config(output_root)
    return yaml_path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', help='Existing directory-based YOLO data.yaml; required unless preparing data')
    parser.add_argument('--foggy-root', help='leftImg8bit_foggy directory for optional dataset preparation')
    parser.add_argument('--labels-dataset', help='YOLO dataset containing source labels and data.yaml')
    parser.add_argument('--output-dataset', default=str(REPOSITORY_ROOT / 'outputs' / 'baseline_dataset'))
    parser.add_argument('--beta', type=float, choices=[0.005, 0.01, 0.02], default=0.02)
    parser.add_argument('--source-beta', type=float, help='Different fog density used in source annotation filenames')
    parser.add_argument('--splits', nargs='+', choices=['train', 'val', 'test'], default=['train', 'val'])
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--img-size', type=int, default=640)
    parser.add_argument('--device', default='0' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--model', default='yolov8n.pt')
    parser.add_argument('--resume')
    parser.add_argument('--project', default=str(REPOSITORY_ROOT / 'outputs' / 'training'))
    parser.add_argument('--name', default='baseline_foggy_beta0.02')
    args = parser.parse_args(argv)
    if args.data:
        if args.foggy_root or args.labels_dataset or args.prepare_only:
            parser.error('--data cannot be combined with dataset preparation options')
        data_yaml = Path(args.data).expanduser().resolve()
        if data_yaml.name != 'data.yaml':
            parser.error('--data must point to the dataset data.yaml')
    else:
        if not args.foggy_root or not args.labels_dataset:
            parser.error('provide --data, or both --foggy-root and --labels-dataset')
        if 'train' not in args.splits or 'val' not in args.splits:
            parser.error('baseline preparation requires train and val splits')
        data_yaml = prepare_baseline_dataset(args.foggy_root, args.labels_dataset,
                                            args.output_dataset, args.beta, args.source_beta, args.splits)
    load_dataset_config(data_yaml.parent)
    if args.prepare_only:
        print(f'Baseline dataset ready: {data_yaml}')
        return
    if min(args.epochs, args.batch_size, args.img_size) < 1:
        parser.error('epochs, batch size, and image size must be positive')
    from ultralytics import YOLO
    if args.resume:
        validate_checkpoint(args.resume)
    model = YOLO(args.resume or args.model)
    results = model.train(data=str(data_yaml), epochs=args.epochs, batch=args.batch_size,
                          imgsz=args.img_size, device=args.device, project=args.project,
                          name=args.name, exist_ok=False, patience=20, save=True,
                          plots=True, verbose=True, resume=bool(args.resume))
    print(f'Best mAP@50: {results.results_dict.get("metrics/mAP50(B)", "N/A")}')
    return results


if __name__ == '__main__':
    main()
