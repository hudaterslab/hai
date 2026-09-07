"""화각 보정 회귀 테스트. 장치/API 초기화 없이 실제 소스의 관련 메서드를 실행한다."""
import ast
import itertools
import math
from pathlib import Path
import threading
import types
import unittest
from unittest.mock import Mock


SOURCE = Path(__file__).resolve().parents[1] / globals().get(
    "SOURCE_FILENAME", "multi_event_irfilter_danmal.py")
TREE = ast.parse(SOURCE.read_text(encoding="utf-8-sig"))
CAMERA = next(n for n in TREE.body if isinstance(n, ast.ClassDef) and n.name == "Camera")
METHODS = {n.name: n for n in CAMERA.body if isinstance(n, ast.FunctionDef)}
NS = dict(math=math, threading=threading, SYS_CFG={}, ROI_ALIGN_LEARNING_DEFAULTS={},
          ROI_DRIFT_CONFIRM_COUNT=3, GRID_DISTURBED_CONFIRM_COUNT=3,
          GRID_ABNORMAL_CONFIRM_COUNT=3)
STORE = next(n for n in TREE.body if isinstance(n, ast.ClassDef) and n.name == "ROIAlignLearningStore")
exec(compile(ast.Module(body=[STORE, METHODS['_record_roi_auto_correct_observation'],
                             METHODS['_reset_alignment_state']],
                        type_ignores=[]), str(SOURCE), 'exec'), NS)

# _update_alignment의 실제 보정 블록을 분리한다. 앞단 영상 측정 및 외부 전송만 제외한다.
body = METHODS['_update_alignment'].body
def assigns(node, name):
    return isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in node.targets)
start = next(i for i, n in enumerate(body) if assigns(n, 'roi_corrected'))
end = next(i for i, n in enumerate(body) if assigns(n, 'healthcheck_requested'))
CORRECTION = compile(ast.Module(body=body[start:end], type_ignores=[]), str(SOURCE), 'exec')
REPORT = compile(ast.Module(body=body[end:end + 3], type_ignores=[]), str(SOURCE), 'exec')


class RoiAutoCorrectTests(unittest.TestCase):
    def setUp(self):
        self.store = NS['ROIAlignLearningStore']()
        self.store._now_iso = lambda: 'test'
        self.cam = types.SimpleNamespace(roi_auto_correct_observations=[])

    def observe(self, kind, shift=(10., 20.)):
        decision = self.store.record_check('cam', {}, kind == 'S', disturbed=kind == 'D')
        candidate = NS['_record_roi_auto_correct_observation'](
            self.cam, decision, dict(median_dx=shift[0], median_dy=shift[1]))
        return decision, candidate

    def test_all_three_observation_combinations(self):
        for sequence in itertools.product('SD', repeat=3):
            with self.subTest(sequence=sequence):
                self.setUp()
                for index, kind in enumerate(sequence):
                    decision, candidate = self.observe(kind, (10., 20.) if kind == 'S' else (120., -90.))
                    if index < 2:
                        self.assertIsNone(candidate)
                self.assertEqual(candidate, (10., 20.) if sequence.count('S') >= 2 else None)
                self.assertTrue(decision['pending'])
                self.assertTrue(decision['healthcheck'])
                self.assertEqual(decision['decision'], 'confirm')
                # 대기 중 합성 카운터가 증가해도 새 보정 후보는 생성하지 않는다.
                for _ in range(4):
                    self.assertIsNone(self.observe('S')[1])

    def test_reported_csv_case(self):
        self.observe('S', (-7.5, -25.3))
        self.observe('S', (-4.1, -20.4))
        decision, candidate = self.observe('D', (27.6, -9.1))
        self.assertAlmostEqual(candidate[0], -5.8)
        self.assertAlmostEqual(candidate[1], -22.85)
        self.assertFalse(decision['confirmed'])  # 기존 confirmed 조건에 의존하지 않아야 함

    def test_normal_clears_old_shifts(self):
        self.observe('S', (100., 100.))
        self.observe('N')
        self.assertEqual(self.cam.roi_auto_correct_observations, [])
        self.observe('D')
        self.observe('S', (2., 4.))
        self.assertEqual(self.observe('S', (4., 6.))[1], (3., 5.))

    def test_nonfinite_suspect_is_not_corrected(self):
        self.observe('S', (float('nan'), 0.))
        self.observe('S')
        self.assertIsNone(self.observe('D')[1])

    def test_configuration_reset_clears_observations_and_latch(self):
        self.observe('S')
        self.cam.roi_auto_corrected = True
        NS['AnchorTrackingROIAligner'] = Mock
        NS['_reset_alignment_state'](self.cam)
        self.assertEqual(self.cam.roi_auto_correct_observations, [])
        self.assertFalse(self.cam.roi_auto_corrected)

    def correction_env(self, candidate=(-5.8, -22.85), homography=True):
        cam = self.cam
        cam.events = ['roi_change_apply']
        cam.roi_auto_corrected = False
        cam.base_roi_poly = [[100, 100]]
        cam.base_roi_lines = []
        cam.aligned_roi_poly = [[100, 100]]
        cam.aligned_roi_lines = []
        cam.roi_shift = [0., 0.]
        cam.cam_id, cam.ip, cam.camera_key = 4, 'camera', 'cam'
        cam.roi_setup_pending = True
        cam.roi_align_untrusted = True
        cam.aligner = types.SimpleNamespace(anchor_slots={'updated': {'gray': 'anchor'}},
                                            _gray_plain=lambda f: 'current', refresh_grid_anchor=Mock())
        cam._inject_roi_to_handlers = Mock()
        cam._shift_roi_points = lambda points, shift: [[p[0] + shift[0], p[1] + shift[1]] for p in points]
        estimate = Mock(return_value=('H' if homography else None, 'test'))
        return dict(math=math, self=cam, auto_correct_shift=candidate,
                    grid={'median_dx': 27.6, 'median_dy': -9.1}, frame='frame',
                    ROI_CHANGE_APPLY_EVENT='roi_change_apply', GRID_APPLY_MAX_SHIFT_PX=150.,
                    GRID_APPLY_SHIFT_SIGN=1., ANCHOR_UPDATED='updated', ANCHOR_BASE='base',
                    estimate_alignment_homography=estimate,
                    transform_roi_points_h=lambda points, h: [[p[0] + 3, p[1] + 4] for p in points],
                    refine_roi_local_residual=lambda *args: (None, 'skip'),
                    logger=Mock(), append_roi_change_log=Mock(), check_id='test',
                    consistent=3, consistent_quorum=4)

    def test_homography_uses_suspect_mean_but_applies_its_own_transform(self):
        env = self.correction_env()
        exec(CORRECTION, env)
        self.assertEqual(env['estimate_alignment_homography'].call_args.kwargs['expected_shift'], (-5.8, -22.85))
        self.assertEqual(self.cam.aligned_roi_poly, [[103, 104]])
        self.assertEqual(env['roi_correct_method'], 'homography')
        self.assertTrue(self.cam.roi_auto_corrected)
        self.assertTrue(self.cam.roi_setup_pending)
        self.assertTrue(self.cam.roi_align_untrusted)
        exec(CORRECTION, env)
        self.assertEqual(env['estimate_alignment_homography'].call_count, 1)

    def test_failed_homography_falls_back_to_suspect_mean(self):
        env = self.correction_env(homography=False)
        exec(CORRECTION, env)
        self.assertEqual(self.cam.roi_shift, [-5.8, -22.85])
        self.assertEqual(env['roi_correct_method'], 'translation')
        self.assertEqual(self.cam.aligned_roi_poly, [[94.2, 77.15]])

    def test_no_candidate_or_invalid_magnitude_does_not_correct(self):
        for candidate in (None, (0., 0.), (151., 0.)):
            with self.subTest(candidate=candidate):
                env = self.correction_env(candidate=candidate)
                exec(CORRECTION, env)
                env['estimate_alignment_homography'].assert_not_called()
                self.assertFalse(self.cam.roi_auto_corrected)

    def test_notify_only_camera_does_not_correct(self):
        env = self.correction_env()
        self.cam.events = ['roi_change']
        exec(CORRECTION, env)
        env['estimate_alignment_homography'].assert_not_called()

    def test_mixed_sequences_report_success_and_keep_setup_request(self):
        for sequence in ('SSD', 'SDS', 'DSS', 'SSS'):
            with self.subTest(sequence=sequence):
                self.setUp()
                for kind in sequence:
                    decision, candidate = self.observe(kind)
                env = self.correction_env(candidate=candidate)
                exec(CORRECTION, env)
                env.update(decision=decision, observed_decision=decision['observed_decision'],
                           suspect_count=decision['suspect_count'], disturbed_count=decision['disturbed_count'],
                           abnormal_count=3, abnormal_required=3, disturbed_required=3, confirm_required=3,
                           n_mov=9, n_meas=9, request_terminal_roi_setup_required=Mock())
                exec(REPORT, env)
                self.assertIn('auto_corrected=True', env['healthcheck_reason'])
                self.assertIn('ROI AUTO-CORRECT[homography]', self.cam.align_status_text)
                env['request_terminal_roi_setup_required'].assert_called_once()
                self.assertTrue(self.cam.roi_setup_pending)
                self.assertTrue(self.cam.roi_align_untrusted)


if __name__ == '__main__':
    unittest.main()
