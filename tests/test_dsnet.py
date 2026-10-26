"""Regression checks for DSNet detection layout and supervised training targets."""

import importlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'experiments/DSNET'))
from detection import compute_detection_loss, decode_detections
from dsnet_yolo import DSNet_YOLO
from coco_dataset import CityscapesCocoDataset


class DetectionTests(unittest.TestCase):
    def test_class_filter_runs_before_detection_limit(self):
        head = torch.full((1, 21, 1, 2), -20.0)
        for column, label, objectness in [(0, 0, 10), (1, 1, 5)]:
            head[0, :2, 0, column] = 0
            head[0, 2:4, 0, column] = 0.2
            head[0, 4, 0, column] = objectness
            head[0, 5 + label, 0, column] = 10
        decoded = decode_detections([head], 2, (40, 80), max_detections=1,
                                    confidence=0.5, classes=[1])[0]
        self.assertEqual(decoded.shape, (1, 6))
        self.assertEqual(decoded[0, 5].item(), 1)
        self.assertEqual(len(decode_detections([head], 2, classes=[])[0]), 0)
        with self.assertRaisesRegex(ValueError, 'zero-based'):
            decode_detections([head], 2, classes=[2])

    def test_new_training_run_preserves_existing_output(self):
        module = importlib.import_module('train_dsnet_yolo')
        with tempfile.TemporaryDirectory() as directory:
            marker = Path(directory) / 'history.json'
            marker.write_text('existing run')
            with self.assertRaisesRegex(FileExistsError, 'empty save directory'):
                module.main(['--dataset-root', directory, '--save-dir', directory])
            self.assertEqual(marker.read_text(), 'existing run')

    def test_three_slots_stride_class_ids_and_class_aware_nms(self):
        head = torch.full((1, 21, 2, 4), -20.0)
        # Two identical boxes of distinct classes in slots 0 and 1.
        for slot, label in [(0, 0), (1, 1)]:
            offset = slot * 7
            head[0, offset:offset + 2, 1, 2] = 0
            head[0, offset + 2:offset + 4, 1, 2] = 0.25
            head[0, offset + 4, 1, 2] = 10
            head[0, offset + 5 + label, 1, 2] = 10
        decoded = decode_detections([head, head], 2, (40, 80), confidence=0.5)[0]
        self.assertEqual(decoded.shape, (2, 6))
        self.assertEqual(set(decoded[:, 5].tolist()), {0.0, 1.0})
        # x=2.5*20=50, y=1.5*20=30, wh=(20,10).
        torch.testing.assert_close(decoded[0, :4], torch.tensor([40., 25., 60., 35.]))

    def test_coco_conversion_rescales_boxes_and_maps_category_ids(self):
        from evaluate_dsnet_yolo_detection import process_yolo_outputs
        head = torch.full((1, 21, 2, 4), -20.0)
        head[0, :2, 1, 2] = 0
        head[0, 2:4, 1, 2] = 0.25
        head[0, 4, 1, 2] = 10
        head[0, 6, 1, 2] = 10
        result = process_yolo_outputs([head], 80, 160, 0.5, 2, [3, 9],
                                      image_size=(40, 80))
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]['category_id'], 9)
        self.assertEqual(result[0]['bbox'], [80., 50., 40., 20.])

    def test_loss_supervises_xy_wh_class_and_objectness(self):
        heads = [torch.zeros((1, 21, 2, 4), requires_grad=True)]
        loss = compute_detection_loss(heads, [torch.tensor([[10., 10., 30., 20.]])],
                                      [torch.tensor([1])], image_size=(40, 80))
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        grads = heads[0].grad[0]
        for channel in [0, 1, 2, 3, 4, 5, 6]:
            self.assertGreater(grads[channel].abs().sum().item(), 0, channel)
        changed_class = compute_detection_loss(heads, [torch.tensor([[10., 10., 30., 20.]])],
                                               [torch.tensor([0])], image_size=(40, 80))
        # At zero logits losses coincide, but class gradients must reverse.
        gradient = torch.autograd.grad(changed_class, heads)[0]
        self.assertFalse(torch.equal(gradient[:, 5:7], heads[0].grad[:, 5:7]))

    def test_empty_ground_truth_and_invalid_class(self):
        head = torch.zeros((1, 21, 2, 4), requires_grad=True)
        loss = compute_detection_loss([head], [torch.empty((0, 4))], [torch.empty((0,), dtype=torch.long)])
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        with self.assertRaisesRegex(ValueError, 'zero-based'):
            compute_detection_loss([head], [torch.tensor([[1., 1., 2., 2.]])], [torch.tensor([2])])

    def test_training_module_import_has_no_training_side_effects(self):
        module = importlib.import_module('train_dsnet_yolo')
        self.assertTrue(callable(module.main))
        self.assertFalse(hasattr(module, 'model'))

    def test_dataset_maps_arbitrary_categories_to_zero_based_labels(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for directory, name in [('clear', 'sample_leftImg8bit.png'),
                                    ('foggy', 'sample_leftImg8bit_foggy_beta_0.02.png')]:
                path = root / directory / 'train/city' / name
                path.parent.mkdir(parents=True)
                Image.new('RGB', (20, 10)).save(path)
            annotations = root / 'annotations.json'
            annotations.write_text(json.dumps({
                'images': [{'id': 1, 'city': 'city', 'file_name': 'sample_leftImg8bit.png'}],
                'categories': [{'id': 9, 'name': 'vehicle'}, {'id': 3, 'name': 'person'}],
                'annotations': [{'image_id': 1, 'category_id': 9, 'bbox': [2, 1, 4, 3]}]}))
            dataset = CityscapesCocoDataset(root / 'foggy', root / 'clear', annotations, img_size=40)
            self.assertEqual(dataset.class_names, ['person', 'vehicle'])
            self.assertEqual(dataset[0]['labels'].tolist(), [1])
            torch.testing.assert_close(dataset[0]['boxes'], torch.tensor([[4., 4., 12., 16.]]))

    def test_predict_preserves_model_mode_and_odd_input_extent(self):
        torch.set_num_threads(1)
        model = DSNet_YOLO(num_classes=2, pretrained_cb=False).train()
        before = model.cb_module.bn1.running_mean.clone()
        restored, raw = model.predict(torch.randn(1, 3, 33, 49))
        self.assertTrue(model.training)
        torch.testing.assert_close(model.cb_module.bn1.running_mean, before)
        self.assertEqual(restored.shape, (1, 3, 33, 49))
        self.assertTrue(all(head.shape[1] == 21 for head in raw))


if __name__ == '__main__':
    unittest.main()
