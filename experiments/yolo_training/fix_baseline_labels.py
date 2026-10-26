#!/usr/bin/env python3
"""Match existing annotations to foggy image stems without inventing labels."""

import argparse
from pathlib import Path
import shutil


def find_source_label(image_path, labels_root, split, beta=0.02, source_beta=None):
    image_path = Path(image_path)
    labels_root = Path(labels_root)
    base = image_path.stem.split('_leftImg8bit_foggy_beta_')[0]
    city = image_path.parent.name
    source_beta = beta if source_beta is None else source_beta
    names = [
        image_path.stem + '.txt',
        f'{base}_{city}_beta{source_beta:g}.txt',
        f'{base}_foggy_beta_{source_beta:g}.txt',
        f'{base}_leftImg8bit.txt',
        f'{base}.txt',
    ]
    matches = [directory / name for directory in (labels_root / split, labels_root / split / city)
               for name in names if (directory / name).is_file()]
    if not matches:
        raise FileNotFoundError(f'No source annotation for {image_path}; refusing to create an empty background label.')
    if any(path.read_bytes() != matches[0].read_bytes() for path in matches[1:]):
        raise ValueError(f'Ambiguous annotations for {image_path}: {matches}')
    return matches[0]


def label_plan(foggy_root, labels_root, beta=0.02, source_beta=None, splits=('train', 'val')):
    """Validate all matches before writing any output labels."""
    plan = []
    for split in splits:
        split_root = Path(foggy_root) / split
        if not split_root.is_dir():
            raise FileNotFoundError(f'Missing image split: {split_root}')
        images = sorted(split_root.rglob(f'*_beta_{beta:g}.png'))
        if not images:
            raise ValueError(f'No images with beta={beta:g} in {split_root}')
        seen = set()
        for image in images:
            if image.name in seen:
                raise ValueError(f'Duplicate flattened image name in {split}: {image.name}')
            seen.add(image.name)
            label = find_source_label(image, labels_root, split, beta, source_beta)
            plan.append((split, image, label))
    return plan


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--foggy-root', required=True, help='leftImg8bit_foggy directory')
    parser.add_argument('--labels-root', required=True, help='Source labels directory containing split directories')
    parser.add_argument('--output-root', required=True, help='Destination labels directory')
    parser.add_argument('--beta', type=float, choices=[0.005, 0.01, 0.02], default=0.02)
    parser.add_argument('--source-beta', type=float, help='Explicitly select a different density in source filenames')
    parser.add_argument('--splits', nargs='+', default=['train', 'val'], choices=['train', 'val', 'test'])
    args = parser.parse_args(argv)
    plan = label_plan(args.foggy_root, args.labels_root, args.beta, args.source_beta, args.splits)
    for split, image, source in plan:
        destination = Path(args.output_root) / split / image.with_suffix('.txt').name
        if destination.exists() and destination.read_bytes() != source.read_bytes():
            raise FileExistsError(f'Conflicting destination annotation: {destination}')
    for split, image, source in plan:
        destination = Path(args.output_root) / split / image.with_suffix('.txt').name
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.resolve() != source.resolve():
            shutil.copy2(source, destination)
    print(f'Matched {len(plan)} existing annotations; no synthetic background labels were created.')


if __name__ == '__main__':
    main()
