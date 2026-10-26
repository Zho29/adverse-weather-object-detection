#!/usr/bin/env python3
"""Render foggy inputs, DSNet restoration, and decoded detection boxes."""

import argparse
from pathlib import Path

import torch
from PIL import Image, ImageDraw
from torchvision import transforms

from dsnet_yolo import DSNet_YOLO
from detection import decode_detections


def process_detections(detections, conf_thresh=0.3, num_classes=2,
                       image_size=(512, 512), box_format='legacy'):
    """Compatibility helper returning first-image boxes after shared NMS."""
    if detections[0].shape[0] != 1:
        raise ValueError('Visualization expects a single image')
    decoded = decode_detections(detections, num_classes, image_size,
                                confidence=conf_thresh, box_format=box_format)[0]
    return [{'bbox': row[:4], 'conf': row[4], 'class': int(row[5])}
            for row in decoded.detach().cpu().tolist()]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--foggy-dir', required=True)
    parser.add_argument('--output-dir', default='outputs/dsnet_visualizations')
    parser.add_argument('--beta', type=float, default=0.02, choices=[0.005, 0.01, 0.02])
    parser.add_argument('--num-samples', type=int, default=10)
    parser.add_argument('--img-size', type=int, default=512)
    parser.add_argument('--confidence', type=float, default=0.3)
    args = parser.parse_args(argv)
    if args.num_samples < 1 or args.img_size < 1 or not 0 <= args.confidence <= 1:
        parser.error('Sample count/image size must be positive; confidence must be in [0, 1]')
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.is_file():
        parser.error(f'Checkpoint does not exist: {checkpoint_path}')
    with checkpoint_path.open('rb') as stream:
        if stream.read(128).startswith(b'version https://git-lfs.github.com/spec/v1'):
            parser.error('Checkpoint is a Git LFS pointer; run git lfs pull first')
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    state = checkpoint.get('model_state_dict', checkpoint)
    state = {key.removeprefix('module.'): value for key, value in state.items()}
    num_classes = state['detect_p3.weight'].shape[0] // 3 - 5
    class_names = checkpoint.get('class_names', [f'class_{i}' for i in range(num_classes)])
    if len(class_names) != num_classes:
        parser.error('Checkpoint class names do not match its detection head')
    model = DSNet_YOLO(num_classes=num_classes, pretrained_cb=False)
    model.load_state_dict(state)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device).eval()
    box_format = checkpoint.get('detection_box_format', 'legacy')
    if box_format == 'legacy':
        print('Legacy checkpoint: original box/class training was incomplete; these are qualitative outputs, not validated detections.')
    transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)), transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3)])
    paths = sorted(Path(args.foggy_dir).rglob(f'*_beta_{args.beta}.png'))[:args.num_samples]
    if not paths:
        parser.error('No matching foggy images found')
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for index, path in enumerate(paths, 1):
        with Image.open(path) as source:
            image = source.convert('RGB')
        with torch.inference_mode():
            restored, heads = model(transform(image).unsqueeze(0).to(device))
        restored_image = transforms.ToPILImage()(((restored[0].cpu() + 1) / 2).clamp(0, 1))
        annotated = restored_image.copy()
        draw = ImageDraw.Draw(annotated)
        for item in process_detections(heads, args.confidence, num_classes,
                                       (args.img_size, args.img_size), box_format):
            draw.rectangle(item['bbox'], outline='orange', width=2)
            draw.text(tuple(item['bbox'][:2]),
                      f"{class_names[item['class']]} {item['conf']:.2f}", fill='orange')
        canvas = Image.new('RGB', (args.img_size * 3, args.img_size))
        for panel_index, panel in enumerate([image.resize(restored_image.size), restored_image, annotated]):
            canvas.paste(panel, (panel_index * args.img_size, 0))
        save_path = output_dir / f'dsnet_yolo_{index:02d}.png'
        canvas.save(save_path)
        print(save_path)


if __name__ == '__main__':
    main()
