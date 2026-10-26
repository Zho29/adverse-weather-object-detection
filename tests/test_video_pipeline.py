"""Video I/O and coordinate regression tests without external weights or data."""

import json
from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np
import torch

from video_detection_analysis.pipeline import (
    checkpoint_path, detection_records, normalized_frame, parse_args,
    restored_frame, run_video,
)


class FakePredictor:
    names = {0: 'person', 1: 'vehicle'}

    def predict(self, frame):
        return frame, np.array([[2, 3, 20, 22, 0.9, 1]])


class VideoTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / 'input.avi'
        writer = cv2.VideoWriter(str(self.source), cv2.VideoWriter_fourcc(*'MJPG'), 10, (32, 24))
        self.assertTrue(writer.isOpened())
        for value in (30, 60, 90):
            writer.write(np.full((24, 32, 3), value, dtype=np.uint8))
        writer.release()

    def args(self, *extra):
        return parse_args(['--input', str(self.source), '--output', str(self.root / 'output.avi'), *extra])

    def test_video_output_frame_count_fps_json_and_class_names(self):
        args = self.args()
        summary = run_video(args, FakePredictor())
        self.assertEqual(summary['frames_processed'], 3)
        self.assertEqual(summary['detections_by_class'], {'vehicle': 3})
        self.assertEqual(summary['source_fps'], 10)
        rows = [json.loads(line) for line in (self.root / 'output.jsonl').read_text().splitlines()]
        self.assertEqual([row['frame_index'] for row in rows], [0, 1, 2])
        self.assertEqual(rows[2]['timestamp_seconds'], 0.2)
        self.assertEqual(rows[0]['detections'][0]['xyxy'], [2, 3, 20, 22])
        capture = cv2.VideoCapture(args.output)
        try:
            self.assertEqual(int(capture.get(cv2.CAP_PROP_FRAME_COUNT)), 3)
            self.assertEqual(capture.get(cv2.CAP_PROP_FPS), 10)
            self.assertTrue(capture.read()[0])
        finally:
            capture.release()

    def test_max_frames_and_empty_detections(self):
        predictor = FakePredictor()
        predictor.predict = lambda frame: (frame, np.empty((0, 6)))
        summary = run_video(self.args('--max-frames', '1'), predictor)
        self.assertEqual(summary['frames_processed'], 1)
        self.assertEqual(summary['detections'], 0)

    def test_preserves_existing_outputs_and_input(self):
        target = self.root / 'output.avi'
        target.write_bytes(b'keep existing output')
        with self.assertRaises(FileExistsError):
            run_video(self.args(), FakePredictor())
        self.assertEqual(target.read_bytes(), b'keep existing output')
        args = self.args()
        args.output = str(self.source)
        original = self.source.read_bytes()
        with self.assertRaises(ValueError):
            run_video(args, FakePredictor())
        self.assertEqual(self.source.read_bytes(), original)

    def test_invalid_video_fails_without_outputs(self):
        self.source.write_bytes(b'not a video')
        with self.assertRaises(ValueError):
            run_video(self.args(), FakePredictor())
        self.assertFalse((self.root / 'output.avi').exists())

    def test_inference_failure_cleans_outputs_and_closes_video(self):
        predictor = FakePredictor()
        def fail(frame):
            raise RuntimeError('fake inference error')
        predictor.predict = fail
        with self.assertRaisesRegex(RuntimeError, 'fake inference'):
            run_video(self.args(), predictor)
        for suffix in ('.avi', '.json', '.jsonl'):
            self.assertFalse((self.root / ('output' + suffix)).exists())
        # A retry can create all outputs again after the writer has been released.
        self.assertEqual(run_video(self.args(), FakePredictor())['frames_processed'], 3)

    def test_lfs_pointer_explains_missing_weights(self):
        path = self.root / 'weights.pt'
        path.write_text('version https://git-lfs.github.com/spec/v1\noid sha256:abc\nsize 100\n')
        with self.assertRaisesRegex(ValueError, 'git lfs pull'):
            checkpoint_path(path)

    def test_normalization_rgb_order_and_tanh_inverse(self):
        bgr = np.zeros((16, 16, 3), dtype=np.uint8)
        bgr[:, :, 2] = 255
        tensor = normalized_frame(bgr, 16, torch.device('cpu'))
        torch.testing.assert_close(tensor[0, :, 0, 0], torch.tensor([1., -1., -1.]))
        np.testing.assert_array_equal(restored_frame(tensor, bgr.shape), bgr)

    def test_invalid_class_ids_and_boxes_are_not_reported_as_detections(self):
        self.assertEqual(detection_records([[3, 4, 2, 9, .8, 0]], {0: 'person'}, (24, 32)), [])
        for rows in ([[0, 0, 2, 2, .8, 5]], [[0, 0, 2, 2, .8, .5]], [[0, 0, float('nan'), 2, .8, 0]]):
            with self.assertRaises(ValueError):
                detection_records(rows, {0: 'person'}, (24, 32))

    def test_cli_rejects_invalid_numeric_options(self):
        import contextlib
        import io
        for extra in (['--conf', 'nan'], ['--fps', '0'], ['--max-frames', '0'], ['--restoration-size', '17']):
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                self.args(*extra)


if __name__ == '__main__':
    unittest.main()
