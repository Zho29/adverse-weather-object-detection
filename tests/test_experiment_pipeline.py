"""Regression checks for dataset preparation and restoration correctness."""

import importlib
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

from PIL import Image
import torch
import torchvision.transforms as transforms
import yaml

TRAINING_ROOT = Path(__file__).resolve().parents[1] / 'experiments/yolo_training'
sys.path.insert(0, str(TRAINING_ROOT))
workflow = importlib.import_module('train_yolo_with_wunet')
baseline = importlib.import_module('train_baseline_yolo_foggy')
labels = importlib.import_module('fix_baseline_labels')


class DatasetPipelineTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.source = self.root / 'source'
        self.source.mkdir()
        config = {'names': ['person', 'vehicle'], 'nc': 2, 'path': '.',
                  'train': 'images/train', 'val': 'images/val'}
        (self.source / 'data.yaml').write_text(yaml.safe_dump(config))
        for split in ('train', 'val'):
            for name, size in [('nested/first.png', (20, 24)), ('second.jpg', (32, 16))]:
                image = self.source / 'images' / split / name
                image.parent.mkdir(parents=True, exist_ok=True)
                Image.new('RGB', size, (64, 128, 192)).save(image)
                annotation = self.source / 'labels' / split / Path(name).with_suffix('.txt')
                annotation.parent.mkdir(parents=True, exist_ok=True)
                annotation.write_text('1 0.5 0.5 0.2 0.4\n')

    def tearDown(self):
        self.temporary.cleanup()

    @staticmethod
    def restorer():
        instance = workflow.WUNETImageRestorer.__new__(workflow.WUNETImageRestorer)
        instance.device = torch.device('cpu')
        instance.model = torch.nn.Identity()
        instance.transform = transforms.Compose([
            transforms.ToTensor(), transforms.Normalize([0.5] * 3, [0.5] * 3),
        ])
        return instance

    def test_mixed_size_images_preserve_paths_labels_and_normalization(self):
        destination = self.root / 'restored'
        self.restorer().restore_dataset(self.source, destination)
        config, splits = workflow.load_dataset_config(destination)
        self.assertEqual(Path(config['path']).resolve(), destination.resolve())
        self.assertEqual(config['names'], ['person', 'vehicle'])
        self.assertEqual(len(splits['train']), 2)
        self.assertEqual((destination / 'labels/train/nested/first.txt').read_text(), '1 0.5 0.5 0.2 0.4\n')
        with Image.open(destination / 'images/train/nested/first.png') as image:
            self.assertEqual(image.size, (20, 24))
            # Identity in normalized space must remain the same RGB color.
            for actual, expected in zip(image.getpixel((0, 0)), (64, 128, 192)):
                self.assertLessEqual(abs(actual - expected), 1)

    def test_missing_label_does_not_create_background_or_partial_output(self):
        (self.source / 'labels/train/second.txt').unlink()
        destination = self.root / 'restored'
        with self.assertRaisesRegex(FileNotFoundError, 'Missing annotation'):
            self.restorer().restore_dataset(self.source, destination)
        self.assertFalse(destination.exists())

    def test_restore_cannot_overwrite_source_or_stale_output(self):
        original = (self.source / 'labels/train/second.txt').read_bytes()
        with self.assertRaisesRegex(ValueError, 'overlap'):
            self.restorer().restore_dataset(self.source, self.source)
        self.assertEqual((self.source / 'labels/train/second.txt').read_bytes(), original)
        with self.assertRaises(FileExistsError):
            self.restorer().restore_dataset(self.source, TRAINING_ROOT)

    def test_class_id_and_yaml_root_validation(self):
        (self.source / 'labels/train/second.txt').write_text('2 0.5 0.5 0.2 0.4\n')
        with self.assertRaisesRegex(ValueError, 'Invalid YOLO box'):
            workflow.load_dataset_config(self.source)
        config = yaml.safe_load((self.source / 'data.yaml').read_text())
        config['train'] = '/wrong/source/images/train'
        (self.source / 'data.yaml').write_text(yaml.safe_dump(config))
        with self.assertRaisesRegex(ValueError, 'must point to'):
            workflow.load_dataset_config(self.source)

    def test_lfs_pointer_fails_before_deserialization(self):
        pointer = self.root / 'model.pth'
        pointer.write_text('version https://git-lfs.github.com/spec/v1\noid sha256:abc\nsize 123\n')
        with self.assertRaisesRegex(ValueError, 'git lfs pull'):
            workflow.validate_checkpoint(pointer)

    def test_checkpoint_without_metric_metadata_loads(self):
        sys.path.insert(0, str(TRAINING_ROOT.parent / 'WUNET'))
        model = torch.nn.Conv2d(3, 3, 1)
        checkpoint = self.root / 'model.pth'
        torch.save({'model_state_dict': model.state_dict()}, checkpoint)
        with patch('weather_unet.WeatherUNet', return_value=model):
            restored = workflow.WUNETImageRestorer(checkpoint, device='cpu')
        self.assertIsInstance(restored.model, torch.nn.Conv2d)

    def foggy_data(self):
        foggy = self.root / 'foggy'
        label_dataset = self.root / 'label_dataset'
        label_dataset.mkdir()
        (label_dataset / 'data.yaml').write_text(yaml.safe_dump({'names': ['person', 'vehicle'], 'nc': 2}))
        for split in ('train', 'val'):
            image = foggy / split / 'city' / 'city_000000_000000_leftImg8bit_foggy_beta_0.02.png'
            image.parent.mkdir(parents=True)
            Image.new('RGB', (32, 32)).save(image)
            annotation = label_dataset / 'labels' / split / 'city_000000_000000_city_beta0.02.txt'
            annotation.parent.mkdir(parents=True)
            annotation.write_text('1 0.5 0.5 0.2 0.4\n')
        return foggy, label_dataset

    def test_baseline_preserves_class_mapping_and_renames_annotations(self):
        foggy, source_labels = self.foggy_data()
        destination = self.root / 'baseline'
        path = baseline.prepare_baseline_dataset(foggy, source_labels, destination)
        config, splits = workflow.load_dataset_config(path.parent)
        self.assertEqual(config['names'], ['person', 'vehicle'])
        self.assertEqual(len(splits['train']), 1)
        expected = destination / 'labels/train/city_000000_000000_leftImg8bit_foggy_beta_0.02.txt'
        self.assertEqual(expected.read_text(), '1 0.5 0.5 0.2 0.4\n')

    def test_baseline_missing_annotations_fail_without_writes(self):
        foggy, source_labels = self.foggy_data()
        next((source_labels / 'labels/val').glob('*.txt')).unlink()
        destination = self.root / 'baseline'
        with self.assertRaisesRegex(FileNotFoundError, 'refusing to create an empty'):
            baseline.prepare_baseline_dataset(foggy, source_labels, destination)
        self.assertFalse(destination.exists())

    def test_baseline_output_cannot_overlap_source_datasets(self):
        foggy, source_labels = self.foggy_data()
        for output in (foggy / 'prepared', source_labels / 'prepared', self.root):
            with self.assertRaisesRegex(ValueError, 'separate'):
                baseline.prepare_baseline_dataset(foggy, source_labels, output)
        self.assertFalse((foggy / 'prepared').exists())
        self.assertFalse((source_labels / 'prepared').exists())

    def test_training_validates_the_selected_yaml_instead_of_another_file(self):
        other_config = self.source / 'different.yaml'
        other_config.write_text('names: [wrong]\n')
        with self.assertRaisesRegex(ValueError, 'named data.yaml'):
            workflow.train_yolo(other_config)

    def test_training_resume_is_forwarded_and_cpu_default_is_used(self):
        captured = {}
        checkpoint = self.root / 'resume.pt'
        checkpoint.write_bytes(b'placeholder local checkpoint for mocked detector')
        class Detector:
            def __init__(self, model):
                captured['model'] = model
            def train(self, **kwargs):
                captured.update(kwargs)
                return object()
        fake_module = types.ModuleType('ultralytics')
        fake_module.YOLO = Detector
        with patch.dict(sys.modules, {'ultralytics': fake_module}), patch('torch.cuda.is_available', return_value=False):
            workflow.train_yolo(self.source / 'data.yaml', resume=checkpoint)
        self.assertIs(captured['resume'], True)
        self.assertEqual(captured['device'], 'cpu')
        self.assertIs(captured['exist_ok'], False)


if __name__ == '__main__':
    unittest.main()
