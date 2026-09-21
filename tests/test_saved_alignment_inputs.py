"""Verify production matching consumes persisted JPEGs and preserves coordinate frames."""
import ast
import hashlib
import json
import os
from pathlib import Path
import runpy
import tempfile
import unittest
from unittest.mock import Mock, patch

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'multi_event_irfilter.py'


class SavedAlignmentTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.ns = dict(runpy.run_path(str(ROOT / 'tests/test_roi_auto_correct_irfilter.py'))['NS'],
                       os=os, json=json, hashlib=hashlib, ROI_ALIGN_IMAGE_DIR=self.temp.name,
                       ROI_ALIGN_IMAGE_WIDTH=640, ROI_ALIGN_IMAGE_JPEG_QUALITY=70)
        tree = ast.parse(SOURCE.read_text(encoding='utf-8-sig'))
        nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in
                 ('_resize_for_align_log', 'estimate_saved_alignment_homography')]
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(SOURCE), 'exec'), self.ns)

    def run_saved(self, base, current, lines):
        ref = dict(gray=base, shape=base.shape, poly=[], lines=lines)
        return self.ns['estimate_saved_alignment_homography'](ref, current, 'CAM4_20260911_120000')

    def test_disk_pixels_and_replay_result_are_identical(self):
        base = np.random.default_rng(19).integers(0, 256, (480, 640), dtype=np.uint8)
        current = cv2.warpAffine(cv2.cvtColor(base, cv2.COLOR_GRAY2BGR),
                                 np.float32([[1, 0, 30], [0, 1, 12]]), (640, 480))
        lines = [[200, 200], [200, 350]]
        estimator = self.ns['estimate_alignment_homography']
        spy = Mock(wraps=estimator)
        self.ns['estimate_alignment_homography'] = spy
        H, status = self.run_saved(base, current, lines)
        self.assertIsNotNone(H, status)
        meta_path = next(Path(self.temp.name).rglob('*_matching.json'))
        meta = json.loads(meta_path.read_text(encoding='utf-8'))
        images = []
        for name, expected_hash in zip((meta['base_file'], meta['current_file']), meta['sha256']):
            p = meta_path.parent / name
            self.assertEqual(hashlib.sha256(p.read_bytes()).hexdigest(), expected_hash)
            images.append(cv2.imread(str(p), cv2.IMREAD_GRAYSCALE))
        for actual, expected in zip(spy.call_args.args[:2], images):
            np.testing.assert_array_equal(actual, expected)
        replay, _ = estimator(*images, meta['saved_poly'], meta['saved_lines'])
        np.testing.assert_array_equal(H, replay)
        self.assertEqual(meta['projected_lines'], [[230, 212], [230, 362]])
        replay_result = runpy.run_path(str(ROOT / 'replay_roi_alignment.py'))['replay'](meta_path)
        self.assertTrue(replay_result['matches_recorded_result'])
        (meta_path.parent / meta['current_file']).write_bytes(b'changed input')
        with self.assertRaisesRegex(ValueError, 'hash mismatch'):
            runpy.run_path(str(ROOT / 'replay_roi_alignment.py'))['replay'](meta_path)

    def test_resize_converts_roi_and_homography_back_to_original_coordinates(self):
        base = np.zeros((720, 1280), np.uint8)
        H_saved = np.float64([[1, 0, 25], [0, 1, 10], [0, 0, 1]])
        estimator = Mock(return_value=(H_saved, 'ok'))
        self.ns['estimate_alignment_homography'] = estimator
        H, status = self.run_saved(base, cv2.cvtColor(base, cv2.COLOR_GRAY2BGR),
                                   [[200, 200], [200, 600]])
        self.assertIsNotNone(H, status)
        self.assertEqual(estimator.call_args.args[0].shape, (360, 640))
        self.assertEqual(estimator.call_args.args[3], [[100, 100], [100, 300]])
        np.testing.assert_array_equal(H, [[1, 0, 50], [0, 1, 20], [0, 0, 1]])

    def test_save_failure_never_matches_memory_frame(self):
        estimator = Mock()
        self.ns['estimate_alignment_homography'] = estimator
        with patch.object(cv2, 'imwrite', return_value=False):
            H, status = self.run_saved(np.zeros((480, 640), np.uint8),
                                       np.zeros((480, 640, 3), np.uint8), [[2, 275], [9, 475]])
        self.assertIsNone(H)
        self.assertIn('image_write_failed', status)
        estimator.assert_not_called()

    def test_read_failure_never_matches_memory_frame(self):
        estimator = Mock()
        self.ns['estimate_alignment_homography'] = estimator
        with patch.object(cv2, 'imdecode', return_value=None):
            H, status = self.run_saved(np.zeros((480, 640), np.uint8),
                                       np.zeros((480, 640, 3), np.uint8), [[2, 275], [9, 475]])
        self.assertIsNone(H)
        self.assertIn('image_read_failed', status)
        estimator.assert_not_called()


if __name__ == '__main__':
    unittest.main()
