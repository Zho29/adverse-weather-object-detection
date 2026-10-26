"""Regression checks for restoration metrics, checkpoint resume, and pairing."""

import contextlib
import io
from pathlib import Path
import random
import sys
import tempfile
import unittest
from unittest.mock import patch

import matplotlib
matplotlib.use('Agg')
import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'experiments' / 'WUNET'))
from foggy_dataset import FoggyCityscapesDataset
from train_wunet_foggy import WUNETFoggyTrainer, calculate_psnr


class WUNETTrainingTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(11)
        self.loader = DataLoader([
            {'foggy': torch.zeros(3, 8, 8), 'clear': torch.full((3, 8, 8), 0.25)}
        ], batch_size=1)

    def make_trainer(self):
        return WUNETFoggyTrainer(nn.Conv2d(3, 3, 1), device='cpu', learning_rate=0.001)

    def test_psnr_uses_normalized_rgb_range_and_perfect_match_is_tensor(self):
        actual = calculate_psnr(torch.zeros(1), torch.ones(1))
        self.assertAlmostEqual(actual.item(), 6.0206, places=4)
        exact = calculate_psnr(torch.ones(1), torch.ones(1))
        self.assertIsInstance(exact, torch.Tensor)
        self.assertTrue(torch.isinf(exact))

    def test_exact_validation_and_small_batch_visualization(self):
        trainer = self.make_trainer()
        trainer.model = nn.Identity()
        exact_loader = DataLoader([
            {'foggy': torch.zeros(3, 8, 8), 'clear': torch.zeros(3, 8, 8)}
        ], batch_size=1)
        loss, psnr = trainer.validate(exact_loader, 1)
        self.assertEqual(loss, 0)
        self.assertEqual(psnr, float('inf'))
        with tempfile.TemporaryDirectory() as directory:
            trainer.visualize_restoration(exact_loader, 1, directory, num_samples=4)
            self.assertTrue((Path(directory) / 'visualizations/epoch_001_beta0.01.png').is_file())

    def test_resume_restores_optimizer_scheduler_and_continues_epoch_numbers(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            original = self.make_trainer()
            with patch.object(original, 'visualize_restoration'):
                original.train(self.loader, self.loader, num_epochs=1,
                               save_dir=directory, save_every=1)
            checkpoint = Path(directory) / 'wunet_foggy_beta0.01_epoch1.pth'
            resumed = self.make_trainer()
            next_epoch = resumed.load_checkpoint(checkpoint)
            self.assertEqual(next_epoch, 2)
            self.assertEqual(resumed.history, original.history)
            self.assertEqual(resumed.best_val_loss, original.best_val_loss)
            self.assertEqual(resumed.scheduler.state_dict(), original.scheduler.state_dict())
            self.assertEqual(len(resumed.optimizer.state), len(original.optimizer.state))
            # The same next batch must produce the same optimizer update.
            original.train_epoch(self.loader, 2)
            resumed.train_epoch(self.loader, 2)
            for left, right in zip(original.model.parameters(), resumed.model.parameters()):
                torch.testing.assert_close(left, right)
            with patch.object(resumed, 'train_epoch', return_value=0.1) as epoch, \
                 patch.object(resumed, 'validate', return_value=(0.1, 10.0)), \
                 patch.object(resumed, 'visualize_restoration'):
                resumed.train(self.loader, self.loader, num_epochs=2,
                              start_epoch=next_epoch, save_dir=directory)
            epoch.assert_called_once_with(self.loader, 2)
            self.assertEqual(len(resumed.history['train_loss']), 2)
            self.assertEqual(resumed.history['psnr_data_ranges'], [2.0, 2.0])

    def test_resume_accepts_dataparallel_keys_and_rejects_lfs_pointer(self):
        trainer = self.make_trainer()
        checkpoint = trainer.checkpoint_state(1, 0.1, 10.0)
        checkpoint['model_state_dict'] = {'module.' + k: v for k, v in checkpoint['model_state_dict'].items()}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'checkpoint.pth'
            torch.save(checkpoint, path)
            self.assertEqual(trainer.load_checkpoint(path), 2)
            path.write_text('version https://git-lfs.github.com/spec/v1\noid sha256:abc\nsize 123\n')
            with self.assertRaisesRegex(ValueError, 'Git LFS pointer'):
                trainer.load_checkpoint(path)

    def test_empty_dataset_reports_actionable_error(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(FileNotFoundError, 'Missing dataset directory'):
                FoggyCityscapesDataset(directory)
            for subdir in ['leftImg8bit/train', 'leftImg8bit_foggy/train']:
                (Path(directory) / subdir).mkdir(parents=True)
            with self.assertRaisesRegex(ValueError, 'No foggy/clear image pairs'):
                FoggyCityscapesDataset(directory)

    def test_random_pair_transforms_stay_aligned(self):
        class RandomOffset:
            def __call__(self, image):
                return transforms.ToTensor()(image) + torch.rand(1).item() + np.random.rand() + random.random()

        with tempfile.TemporaryDirectory() as directory:
            clear = Path(directory) / 'leftImg8bit/train/aachen/aachen_000000_000001_leftImg8bit.png'
            foggy = Path(directory) / 'leftImg8bit_foggy/train/aachen/aachen_000000_000001_leftImg8bit_foggy_beta_0.01.png'
            for path in (clear, foggy):
                path.parent.mkdir(parents=True)
                Image.new('RGB', (8, 8), color=(100, 100, 100)).save(path)
            sample = FoggyCityscapesDataset(directory, transform=RandomOffset())[0]
            torch.testing.assert_close(sample['foggy'], sample['clear'])


if __name__ == '__main__':
    unittest.main()
