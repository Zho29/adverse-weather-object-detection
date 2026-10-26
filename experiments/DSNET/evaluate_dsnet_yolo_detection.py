#!/usr/bin/env python3
"""Evaluate the custom DSNet grid head against COCO annotations."""

import argparse
import json
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

from dsnet_yolo import DSNet_YOLO
from detection import decode_detections


def process_yolo_outputs(detections, orig_h, orig_w, conf_thresh=0.3,
                         num_classes=2, category_ids=None, box_format='legacy',
                         image_size=(512, 512)):
    """Decode a single image to COCO xywh boxes using the real head layout."""
    if detections[0].shape[0] != 1:
        raise ValueError('This COCO conversion helper expects a single image')
    category_ids = category_ids or list(range(1, num_classes + 1))
    if len(category_ids) != num_classes:
        raise ValueError('One category ID is required per detector class')
    decoded = decode_detections(detections, num_classes, image_size,
                                confidence=conf_thresh, box_format=box_format)[0]
    results = []
    input_h, input_w = image_size
    for x1, y1, x2, y2, score, label in decoded.detach().cpu().tolist():
        x1, x2 = x1 * orig_w / input_w, x2 * orig_w / input_w
        y1, y2 = y1 * orig_h / input_h, y2 * orig_h / input_h
        results.append({'bbox': [x1, y1, x2 - x1, y2 - y1],
                        'score': score, 'category_id': category_ids[int(label)]})
    return results


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--annotations', required=True, help='COCO validation JSON')
    parser.add_argument('--foggy-dir', required=True, help='Validation split containing city folders')
    parser.add_argument('--output-dir', default='outputs/dsnet_evaluation')
    parser.add_argument('--img-size', type=int, default=512)
    parser.add_argument('--beta', type=float, default=0.02, choices=[0.005, 0.01, 0.02])
    parser.add_argument('--confidence', type=float, default=0.3)
    args = parser.parse_args(argv)
    if args.img_size < 1 or not 0 <= args.confidence <= 1:
        parser.error('img-size must be positive and confidence must be in [0, 1]')
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.is_file():
        parser.error(f'Checkpoint does not exist: {checkpoint_path}')
    with checkpoint_path.open('rb') as stream:
        if stream.read(128).startswith(b'version https://git-lfs.github.com/spec/v1'):
            parser.error('Checkpoint is a Git LFS pointer; run git lfs pull first')
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    coco_gt = COCO(args.annotations)
    categories = sorted(coco_gt.dataset['categories'], key=lambda item: item['id'])
    class_names = [item['name'] for item in categories]
    category_ids = [item['id'] for item in categories]
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    state = checkpoint.get('model_state_dict', checkpoint)
    state = {key.removeprefix('module.'): value for key, value in state.items()}
    if checkpoint.get('class_names', class_names) != class_names:
        parser.error('Checkpoint class names do not match the ordered COCO categories')
    model = DSNet_YOLO(num_classes=len(categories), pretrained_cb=False)
    model.load_state_dict(state)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device).eval()
    box_format = checkpoint.get('detection_box_format', 'legacy')
    if box_format == 'legacy':
        print('Using legacy box decoding. Historical training did not supervise box/class predictions; retraining is required for a meaningful detector.')
    transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)), transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3)])
    results = []
    for image_id, info in tqdm(coco_gt.imgs.items(), desc='Detection'):
        name = Path(info['file_name']).name.removesuffix('_leftImg8bit.png')
        city = info.get('city', Path(info['file_name']).parent.name)
        foggy_path = Path(args.foggy_dir) / city / f'{name}_leftImg8bit_foggy_beta_{args.beta}.png'
        if not foggy_path.is_file():
            raise FileNotFoundError(f'Missing evaluation image: {foggy_path}')
        with Image.open(foggy_path) as source:
            image = source.convert('RGB')
        original_w, original_h = image.size
        with torch.inference_mode():
            _, heads = model(transform(image).unsqueeze(0).to(device))
        for item in process_yolo_outputs(
                heads, original_h, original_w, args.confidence, len(categories),
                category_ids, box_format, (args.img_size, args.img_size)):
            results.append({'image_id': image_id, **item})
    if not coco_gt.imgs:
        raise ValueError('Evaluation annotations contain no images')
    if results:
        coco_dt = coco_gt.loadRes(results)
    else:
        # loadRes([]) indexes the first result in some pycocotools versions.
        coco_dt = COCO()
        coco_dt.dataset = {'images': list(coco_gt.imgs.values()),
                           'categories': categories, 'annotations': []}
        coco_dt.createIndex()
    evaluator = COCOeval(coco_gt, coco_dt, 'bbox')
    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / 'detections.json').write_text(json.dumps(results))
    metrics = {'mAP@50-95': float(evaluator.stats[0]),
               'mAP@50': float(evaluator.stats[1]), 'mAP@75': float(evaluator.stats[2]),
               'num_detections': len(results), 'class_names': class_names,
               'detection_box_format': box_format, 'checkpoint': str(checkpoint_path)}
    (output_dir / 'metrics.json').write_text(json.dumps(metrics, indent=2) + '\n')


if __name__ == '__main__':
    main()
