"""Direct two-file replay. Never uses suspect_current or an updated reference."""
import json
from pathlib import Path
import runpy
import sys
import time
import cv2
import numpy as np

D=Path(__file__).resolve().parent
sys.path.insert(0,str(D.parents[1]))
from conveyor_roi import estimate_conveyor_centerline, conveyor_search_hints
NS=runpy.run_path(str(D.parents[1]/'tests/test_roi_auto_correct_irfilter.py'),run_name='replay')['NS']
base_path=D/'CAM4_20260907_182506_suspect_anchor_base.jpg'
current_path=D/'CAM4_20260907_183507_confirm_current.jpg'
base,current=[cv2.imread(str(p),0) for p in (base_path,current_path)]
roi=[[47,463],[37,268]]
trace=[]
cv2.setRNGSeed(42)
started=time.perf_counter()
H,old_status=NS['estimate_alignment_homography'](base,current,[],roi,trace)
diagnostics=[]
line,status=estimate_conveyor_centerline(base,current,roi,
    conveyor_search_hints(trace,roi,current.shape),diagnostics)
result=dict(base_file=base_path.name,current_file=current_path.name,
    intermediate_frame_used=False,base_roi=roi,corrected_roi=line,
    point_homography_accepted=H is not None,point_status=old_status,
    centerline_accepted=line is not None,centerline_status=status,
    runtime_seconds=time.perf_counter()-started,
    application_policy='Candidate computed from these two files; production commit also requires a second distinct current observation.',
    diagnostics=diagnostics)
canvas=cv2.cvtColor(np.hstack([base,current]),cv2.COLOR_GRAY2BGR)
cv2.polylines(canvas,[np.int32(roi)],False,(255,255,0),3)
if line is not None:
    cv2.polylines(canvas,[np.int32(np.rint(line)+[base.shape[1],0])],False,(255,255,0),3)
cv2.putText(canvas,'ORIGINAL BASE: original ROI',(10,25),cv2.FONT_HERSHEY_SIMPLEX,.6,(255,255,0),2)
cv2.putText(canvas,'CONFIRM CURRENT: belt centreline',(650,25),cv2.FONT_HERSHEY_SIMPLEX,.6,(255,255,0),2)
cv2.imwrite(str(D/'CAM4_BASE_to_CONFIRM_centerline.jpg'),canvas)
(D/'CAM4_BASE_to_CONFIRM_result.json').write_text(json.dumps(result,indent=2),encoding='utf8')
print(json.dumps({k:v for k,v in result.items() if k!='diagnostics'},indent=2))
if line is None:
    raise SystemExit(1)
