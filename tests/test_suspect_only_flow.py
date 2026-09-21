"""Exercise actual grid and state-machine code without device/API startup."""
import ast
import math
from pathlib import Path
import threading
import types
import unittest
import cv2
import numpy as np

SOURCE=Path(__file__).resolve().parents[1]/'multi_event_irfilter.py'
TREE=ast.parse(SOURCE.read_text(encoding='utf-8-sig'))
NS=dict(cv2=cv2,np=np,math=math,ANCHOR_BASE='base',ANCHOR_UPDATED='updated',
        SYS_CFG={},ROI_ALIGN_LEARNING_DEFAULTS={})
for n in TREE.body:
    if isinstance(n,ast.Assign):
        for t in n.targets:
            if isinstance(t,ast.Name) and (t.id.startswith('GRID_') or t.id=='ROI_DRIFT_CONFIRM_COUNT'):
                NS[t.id]=ast.literal_eval(n.value)
classes={n.name:n for n in TREE.body if isinstance(n,ast.ClassDef)}
observe=next(n for n in classes['Camera'].body if isinstance(n,ast.FunctionDef)
             and n.name=='_record_roi_auto_correct_observation')
exec(compile(ast.Module(body=[classes['AnchorTrackingROIAligner'],classes['ROIAlignLearningStore'],observe],
                       type_ignores=[]),str(SOURCE),'exec'),NS)


class SuspectOnlyTests(unittest.TestCase):
    def grid(self,vectors):
        image=(np.indices((90,90)).sum(axis=0)%50).astype(np.uint8)
        aligner=NS['AnchorTrackingROIAligner']()
        aligner.anchor_slots['updated']={'gray':image}
        aligner._gray_plain=lambda frame:frame
        it=iter(vectors)
        def phase(a,b):
            dx,dy=next(it)
            return dict(dx=dx,dy=dy,shift=math.hypot(dx,dy),response=.9)
        aligner._cell_phase=phase
        return aligner.detect_grid_camera_motion(image)

    def store(self):
        cls=NS['ROIAlignLearningStore']
        store=cls.__new__(cls)
        store.data={};store.lock=threading.Lock();store._now_iso=lambda:'test-time'
        return store

    def test_opposite_directions_all_become_suspect(self):
        result=self.grid([(8,0),(-8,0),(0,8),(0,-8)]*2+[(8,0)])
        self.assertTrue(result['moved'])
        self.assertNotIn('disturbed',result)
        self.assertTrue(all('cos' not in cell for cell in result['cells']))

    def test_threshold_is_strict_and_all_measured_cells_must_move(self):
        self.assertFalse(self.grid([(8,0)]*8+[(2.0,0)])['moved'])
        self.assertFalse(self.grid([(8,0)]*8+[(0,0)])['moved'])
        self.assertTrue(self.grid([(2.0001,0)]*9)['moved'])

    def test_three_suspects_confirm_and_enable_correction_together(self):
        store=self.store()
        cam=types.SimpleNamespace(roi_auto_correct_observations=[],roi_auto_correct_candidate=None)
        statuses=[];candidates=[];requests=[]
        for i in range(3):
            # Obsolete per-camera early-confirm settings cannot bypass the requested 3 checks.
            d=store.record_check('cam',{'roi_align_learning':{'confirm_count_required':1,
                'abnormal_count_required':1}},True)
            statuses.append(d['decision']);requests.append(d['healthcheck'])
            candidates.append(NS['_record_roi_auto_correct_observation'](cam,d,{'median_dx':8,'median_dy':0}))
        self.assertEqual(statuses,['suspect','suspect','confirm'])
        self.assertEqual(requests,[False,False,True])
        self.assertEqual(candidates[:2],[None,None])
        self.assertIsNotNone(candidates[2])
        self.assertFalse(store.record_check('cam',{},True)['healthcheck'])

    def test_normal_breaks_streak_before_confirmation(self):
        store=self.store()
        results=[store.record_check('cam',{},m) for m in (True,True,False,True,True,True)]
        self.assertEqual([d['suspect_count'] for d in results],[1,2,0,1,2,3])
        self.assertEqual([d['decision'] for d in results],['suspect','suspect','normal','suspect','suspect','confirm'])
        self.assertTrue(all(d['observed_decision'] in ('normal','suspect') for d in results))

    def test_legacy_disturbed_latch_never_emits_disturbed(self):
        store=self.store()
        store.data={'cameras':{'cam':dict(awaiting_roi_setup=True,
            latched_abnormal_kind='disturbed',consecutive_disturbed=3)}}
        result=store.record_check('cam',{},True)
        self.assertEqual(result['observed_decision'],'suspect')
        self.assertEqual(result['latched_abnormal_kind'],'suspect')
        self.assertEqual(result['disturbed_count'],0)


if __name__=='__main__':unittest.main()
