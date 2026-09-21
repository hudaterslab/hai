"""Actual image registration and Camera commit/deferral regression tests."""
import ast
import math
import itertools
from pathlib import Path
import types
import sys
import unittest
from unittest.mock import Mock
import cv2
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

SOURCE = Path(__file__).resolve().parents[1] / "multi_event_irfilter.py"
TREE = ast.parse(SOURCE.read_text(encoding="utf-8-sig"))
NS = dict(cv2=cv2, np=np, math=math)
for node in TREE.body:
    if isinstance(node, ast.Assign):
        for t in node.targets:
            if isinstance(t, ast.Name) and t.id.startswith("GRID_HOMOGRAPHY_"):
                NS[t.id] = ast.literal_eval(node.value)
functions = {n.name: n for n in TREE.body if isinstance(n, ast.FunctionDef)}
exec(compile(ast.Module(body=[functions[n] for n in (
    "roi_validation_samples", "validate_roi_homography", "estimate_alignment_homography", "transform_roi_points_h",
    "unique_roi_matches")],
    type_ignores=[]), str(SOURCE), "exec"), NS)
CAM = next(n for n in TREE.body if isinstance(n, ast.ClassDef) and n.name == "Camera")
methods = {n.name: n for n in CAM.body if isinstance(n, ast.FunctionDef)}
body = methods["_update_alignment"].body
start = next(i for i,n in enumerate(body) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id == "roi_corrected" for t in n.targets))
end = next(i for i,n in enumerate(body) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id == "healthcheck_requested" for t in n.targets))
BLOCK = compile(ast.Module(body=body[start:end], type_ignores=[]), str(SOURCE), "exec")

class RegistrationTests(unittest.TestCase):
    def setUp(self):
        cv2.setRNGSeed(42)
        self.src = np.float32([(x,y) for x in range(40,301,30) for y in range(50,421,30)])
        self.H = np.float64([[1,0,250],[0,1,0],[0,0,1]])
        self.dst = self.src + [250,0]
        self.roi = [[100,130],[100,340]]
        self.mask = np.ones(len(self.src), dtype=np.uint8)
    def valid(self, **kw):
        args = dict(H=self.H, src=self.src, dst=self.dst, mask=self.mask,
                    roi_poly=[], roi_lines=self.roi, shape=(480,640))
        args.update(kw)
        return NS['validate_roi_homography'](**args)
    def test_large_supported_translation_allowed(self):
        self.assertTrue(self.valid()[0])
    def test_independent_lines_do_not_create_phantom_midpoints(self):
        pts = np.float32([(x+d,y) for x in (80,560) for d in (-10,0,10)
                          for y in range(80,321,20)])
        valid, reason = NS['validate_roi_homography'](
            np.eye(3), pts, pts, np.ones(len(pts)), [],
            [[80,100],[80,300],[560,100],[560,300]], (480,640))
        self.assertTrue(valid, reason)
    def test_polygon_closure_and_line_pairs_are_preserved(self):
        samples = NS['roi_validation_samples'](
            [[10,10],[30,10],[30,30],[10,30]], [[100,100],[100,200],[500,100],[500,200]])
        points = {tuple(p) for p in samples}
        self.assertIn((10.,20.),points)
        self.assertIn((100.,150.),points)
        self.assertIn((500.,150.),points)
        self.assertNotIn((300.,150.),points)
        self.assertNotIn((55.,65.),points)
    def test_odd_line_count_rejected(self):
        self.assertEqual(self.valid(roi_lines=[[100,100]])[1], 'invalid_roi_topology')
    def test_roi_extrapolation_allowed_when_geometry_is_valid(self):
        valid, reason = self.valid(roi_lines=[[20,470],[100,340]])
        self.assertTrue(valid, reason)
    def test_bad_residual_rejected(self):
        self.assertFalse(self.valid(dst=self.dst+[0,10])[0])
    def test_out_of_frame_rejected(self):
        self.assertFalse(self.valid(roi_lines=[[500,200],[500,300]])[0])
    def test_low_inliers_rejected(self):
        mask = self.mask.copy(); mask[10:] = 0
        self.assertFalse(self.valid(mask=mask)[0])
    def test_clustered_support_rejected(self):
        src = np.float32([(100+x,100+y) for x in range(6) for y in range(6)])
        self.assertFalse(self.valid(src=src,dst=src+[250,0],mask=np.ones(len(src)))[0])
    def test_count_and_ratio_thresholds_both_required(self):
        src=np.float32([(x,y) for y in (60,220,400) for x in (40,160,280)])
        src=np.concatenate([src,[[100,300]]]).astype(np.float32)
        for count,total,expected in ((9,10,False),(10,22,True),(10,23,False)):
            points=np.concatenate([src,np.repeat(src[:1],total-10,axis=0)])
            mask=np.zeros(total,np.uint8);mask[:count]=1
            valid,reason=self.valid(src=points,dst=points+[250,0],mask=mask)
            self.assertEqual(valid,expected,reason)
    def test_reflection_and_excessive_scale_still_rejected(self):
        src=np.float32([(x,y) for x in (40,80,120,160) for y in (40,80,120,160)])
        for H in (np.float64([[-1,0,500],[0,1,0],[0,0,1]]),
                  np.float64([[3,0,0],[0,3,0],[0,0,1]])):
            dst=cv2.perspectiveTransform(src.reshape(-1,1,2),H).reshape(-1,2)
            valid,reason=self.valid(H=H,src=src,dst=dst,mask=np.ones(len(src)),
                                    roi_lines=[[60,60],[100,100]])
            self.assertFalse(valid)
            self.assertIn('roi_bad_local_scale',reason)
    def test_real_feature_matching_translation(self):
        image = np.random.default_rng(4).integers(0,256,(480,640),dtype=np.uint8)
        moved = cv2.warpPerspective(image,self.H,(640,480))
        H, reason = NS['estimate_alignment_homography'](image,moved,[],self.roi)
        self.assertIsNotNone(H,reason)
        pts = cv2.perspectiveTransform(np.float32(self.roi).reshape(-1,1,2),H).reshape(-1,2)
        self.assertLess(np.max(np.linalg.norm(pts-(np.array(self.roi)+[250,0]),axis=1)),2)
    def test_no_texture_rejected(self):
        image = np.zeros((480,640),np.uint8)
        self.assertIsNone(NS['estimate_alignment_homography'](image,image,[],self.roi)[0])
    def weak_roi_scene(self):
        rng = np.random.default_rng(19)
        image = np.full((480,640),128,np.uint8)
        image[:160] = rng.integers(0,256,(160,640),dtype=np.uint8)
        image[160:] = cv2.GaussianBlur(
            rng.integers(112,145,(320,640),dtype=np.uint8), (3,3), 0.6)
        return image
    def test_roi_local_occlusion_is_not_a_separate_gate(self):
        image = self.weak_roi_scene()
        moved = cv2.warpPerspective(image,self.H,(640,480))
        moved[180:] = 128
        H, reason = NS['estimate_alignment_homography'](
            image,moved,[],[[100,270],[100,420]])
        self.assertIsNotNone(H, reason)
    def test_feature_orientation_duplicates_do_not_inflate_support(self):
        src = np.float32([[10,10],[10.1,10.1],[30,30],[50,50]])
        dst = np.float32([[20,20],[20.1,20.1],[40,40],[40.1,40.1]])
        unique_src, unique_dst = NS['unique_roi_matches'](src,dst,3)
        self.assertEqual(len(unique_src),2)
        self.assertEqual(len(unique_dst),2)
    def test_saved_incident_accepts_global_sift_without_rails(self):
        directory=SOURCE.parent/'artifacts/roi_review_20260907'
        a=cv2.imread(str(directory/'CAM4_20260907_182506_suspect_anchor_base.jpg'),0)
        b=cv2.imread(str(directory/'CAM4_20260907_183507_confirm_current.jpg'),0)
        if a is None or b is None: self.skipTest('incident images not present')
        trace=[]
        H, reason=NS['estimate_alignment_homography'](a,b,[],[[47,463],[37,268]],trace)
        self.assertIsNotNone(H,reason)
        self.assertEqual([attempt['method'] for attempt in trace], ['SIFT'])
        self.assertEqual(sum(trace[0]['inliers']), 10)
        self.assertEqual(NS['transform_roi_points_h']([[47,463],[37,268]],H),
                         [[272,462],[285,241]])

class CommitTests(unittest.TestCase):
    def test_cam4_global_sift_commits_without_boundary_fallback(self):
        directory=SOURCE.parent/'artifacts/roi_review_20260907'
        base=cv2.imread(str(directory/'CAM4_20260907_182506_suspect_anchor_base.jpg'),0)
        current=cv2.imread(str(directory/'CAM4_20260907_183507_confirm_current.jpg'),0)
        if base is None or current is None:
            self.skipTest('incident images not present')
        env=self.env((None,'unused'))
        cam=env['self']
        reference=dict(gray=base,poly=[],lines=[[47,463],[37,268]],shape=base.shape)
        cam.roi_correction_reference=reference
        cam.aligned_roi_lines=[[47,463],[37,268]]
        env['frame']=current
        env['estimate_alignment_homography']=NS['estimate_alignment_homography']
        cv2.setRNGSeed(42)
        exec(BLOCK,env)
        self.assertTrue(cam.roi_auto_corrected)
        self.assertEqual(cam.aligned_roi_lines,[[272,462],[285,241]])
        self.assertIs(cam.roi_correction_reference,reference)
        self.assertEqual(env['roi_correct_method'],'homography')
        cam._inject_roi_to_handlers.assert_called_once()
        cam.aligner.refresh_grid_anchor.assert_called_once()
        self.assertTrue(cam.roi_setup_pending and cam.roi_align_untrusted)
        exec(BLOCK,env)
        cam._inject_roi_to_handlers.assert_called_once()



    def test_failed_confirmation_retries_pending_then_stops_after_success(self):
        scope = dict(math=math)
        exec(compile(ast.Module(body=[methods['_record_roi_auto_correct_observation']],
                               type_ignores=[]),str(SOURCE),'exec'),scope)
        env=self.env((None,'occluded'))
        cam=env['self']; cam.roi_auto_correct_candidate=None; cam.roi_auto_correct_observations=[]
        observe=scope['_record_roi_auto_correct_observation']
        for kind in ('suspect','suspect','suspect'):
            candidate=observe(cam,{'observed_decision':kind},{'median_dx':20,'median_dy':10})
        env['auto_correct_shift']=candidate
        exec(BLOCK,env)
        self.assertFalse(cam.roi_auto_corrected)
        env['auto_correct_shift']=observe(cam,{'pending':True, 'latched_abnormal_kind':'suspect',
                                             'observed_decision':'normal'}, {})
        self.assertIsNotNone(env['auto_correct_shift'])
        env['estimate_alignment_homography'].return_value=(np.eye(3),'ok')
        exec(BLOCK,env)
        self.assertTrue(cam.roi_auto_corrected)
        exec(BLOCK,env)
        self.assertEqual(env['estimate_alignment_homography'].call_count,2)
        observe(cam,{'decision':'normal'}, {})
        self.assertIsNone(cam.roi_auto_correct_candidate)
    def test_three_consecutive_suspects_required(self):
        scope = dict(math=math)
        exec(compile(ast.Module(body=[methods['_record_roi_auto_correct_observation']],
                               type_ignores=[]), str(SOURCE), 'exec'), scope)
        for sequence in itertools.product(('suspect', 'normal'), repeat=3):
            cam = types.SimpleNamespace(roi_auto_correct_observations=[], roi_auto_correct_candidate=None)
            for i, observed in enumerate(sequence):
                result = scope['_record_roi_auto_correct_observation'](
                    cam, {'decision': 'confirm' if i == 2 else observed,
                          'observed_decision': observed}, {'median_dx': 0, 'median_dy': 0})
                if i < 2:
                    self.assertIsNone(result)
            self.assertEqual(result is not None, sequence.count('suspect') == 3)

    def test_initial_reference_is_copied(self):
        assignment = next(n for n in ast.walk(methods['_update_alignment'])
                          if isinstance(n, ast.Assign) and any(
                              isinstance(t, ast.Attribute) and t.attr == 'roi_correction_reference'
                              for t in n.targets))
        gray = np.zeros((480, 640), np.uint8)
        cam = types.SimpleNamespace(base_roi_poly=[], base_roi_lines=[[10,20],[30,40]],
            aligner=types.SimpleNamespace(anchor_slots={'base': {'gray': gray}}))
        scope = dict(self=cam,frame=gray,ANCHOR_BASE='base')
        exec(compile(ast.Module(body=[assignment],type_ignores=[]),str(SOURCE),'exec'),scope)
        gray[:] = 100
        cam.base_roi_lines[0][0] = 99
        self.assertEqual(int(cam.roi_correction_reference['gray'].max()),0)
        self.assertEqual(cam.roi_correction_reference['lines'][0],[10,20])

    def test_normal_refresh_tracks_updated_image_and_current_roi(self):
        refresh = next(n for n in body if isinstance(n, ast.If)
                       and isinstance(n.test, ast.Name) and n.test.id == 'refresh_allowed')
        block = compile(ast.Module(body=[refresh], type_ignores=[]), str(SOURCE), 'exec')
        for allowed, action in ((True, 'grid_refresh'), (True, 'skip_refresh_low_texture:0'),
                                (False, 'grid_refresh')):
            with self.subTest(allowed=allowed, action=action):
                updated = np.full((480, 640), 100, np.uint8)
                previous = dict(gray=np.zeros_like(updated), lines=[[10,20],[30,40]])
                cam = types.SimpleNamespace(roi_correction_reference=previous,
                    aligned_roi_poly=[], aligned_roi_lines=[[15,25],[35,45]],
                    aligner=types.SimpleNamespace(anchor_slots={'updated': {'gray': updated}},
                        refresh_grid_anchor=Mock(return_value=action)))
                scope = dict(self=cam, frame=updated, refresh_allowed=allowed,
                             ANCHOR_UPDATED='updated')
                exec(block, scope)
                ref = cam.roi_correction_reference
                if allowed and action == 'grid_refresh':
                    self.assertIsNot(ref, previous)
                    np.testing.assert_array_equal(ref['gray'], updated)
                    updated[:] = 200
                    cam.aligned_roi_lines[0][0] = 99
                    self.assertEqual(int(ref['gray'].max()), 100)
                    self.assertEqual(ref['lines'], [[15,25],[35,45]])
                else:
                    self.assertIs(ref, previous)

    def env(self, result):
        reference=dict(gray=np.zeros((480,640),np.uint8),poly=[],lines=[[100,130],[100,340]],shape=(480,640))
        cam=types.SimpleNamespace(events=['roi_change_apply'],roi_auto_corrected=False,
            roi_correction_reference=reference,base_roi_poly=[],base_roi_lines=reference['lines'],
            aligned_roi_poly=[],aligned_roi_lines=[[100,130],[100,340]],roi_shift=[0.,0.],
            roi_setup_pending=True,roi_align_untrusted=True,cam_id=4,ip='test',camera_key='test',
            _inject_roi_to_handlers=Mock(),aligner=types.SimpleNamespace(
            _gray_plain=lambda f:f,refresh_grid_anchor=Mock()))
        env = dict(NS,self=cam,grid=dict(median_dx=-22,median_dy=-19),
            auto_correct_shift=(-22.,-19.),ROI_CHANGE_APPLY_EVENT='roi_change_apply',
            frame=np.zeros((480,640),np.uint8),estimate_alignment_homography=Mock(return_value=result),
            logger=Mock(),append_roi_change_log=Mock(),check_id='test',n_mov=3,n_meas=3,consistent=3,consistent_quorum=4)
        env['estimate_saved_alignment_homography'] = lambda ref, frame, check_id: env['estimate_alignment_homography'](
            ref['gray'], frame, ref['poly'], ref['lines'])
        return env
    def test_rejection_keeps_roi_anchor_and_pending(self):
        env=self.env((None,'homography_rejected:roi_unsupported'))
        exec(BLOCK,env); cam=env['self']
        self.assertEqual(cam.aligned_roi_lines,[[100,130],[100,340]])
        self.assertFalse(cam.roi_auto_corrected)
        cam.aligner.refresh_grid_anchor.assert_not_called()
        cam._inject_roi_to_handlers.assert_not_called()
        self.assertTrue(cam.roi_setup_pending and cam.roi_align_untrusted)
    def test_verified_transform_applied_once(self):
        H=np.float64([[1,0,250],[0,1,0],[0,0,1]])
        env=self.env((H,'homography_ok'))
        exec(BLOCK,env); cam=env['self']
        self.assertEqual(cam.aligned_roi_lines,[[350,130],[350,340]])
        self.assertTrue(cam.roi_auto_corrected)
        self.assertTrue(cam.roi_setup_pending and cam.roi_align_untrusted)
        exec(BLOCK,env)
        self.assertEqual(env['estimate_alignment_homography'].call_count,1)
    def test_resolution_change_defers(self):
        env=self.env((None,'unused')); env['frame']=np.zeros((240,320),np.uint8)
        exec(BLOCK,env)
        env['estimate_alignment_homography'].assert_not_called()
        self.assertFalse(env['self'].roi_auto_corrected)
    def test_configuration_change_during_matching_discards_result(self):
        env=self.env((None,'unused'))
        def match(*args):
            env['self'].roi_correction_reference=None
            return np.eye(3),'ok'
        env['estimate_alignment_homography']=match
        exec(BLOCK,env)
        self.assertFalse(env['self'].roi_auto_corrected)
        self.assertEqual(env['h_status'],'homography_reference_changed')

if __name__ == '__main__': unittest.main()
