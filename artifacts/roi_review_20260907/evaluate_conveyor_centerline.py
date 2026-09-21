import json
from pathlib import Path
import runpy
import sys
import cv2
import numpy as np
D=Path(__file__).resolve().parent
sys.path.insert(0,str(D.parents[1]))
from conveyor_roi import estimate_conveyor_centerline, detect_conveyor_rails, conveyor_search_hints
NS=runpy.run_path(str(D.parents[1]/'tests/test_roi_auto_correct_irfilter.py'),run_name='replay')['NS']
roi=[[47,463],[37,268]]
a=cv2.imread(str(D/'CAM4_20260907_182506_suspect_anchor_base.jpg'),0)
results=[]
base,reason=detect_conveyor_rails(a,roi,reference=True)
print('BASE',reason,flush=True)
for stage in ('182506_suspect','183507_confirm'):
    b=cv2.imread(str(D/f'CAM4_20260907_{stage}_current.jpg'),0)
    trace=[]
    H,status=NS['estimate_alignment_homography'](a,b,[],roi,trace)
    seeds=conveyor_search_hints(trace,roi,b.shape)
    diag=[]
    line,reason=estimate_conveyor_centerline(a,b,roi,seeds,diag)
    print(stage,line,reason,flush=True)
    result=dict(stage=stage,line=line,reason=reason,diagnostics=diag)
    results.append(result)
    canvas=cv2.cvtColor(b,cv2.COLOR_GRAY2BGR)
    for r in diag:
        if r['result'] is None:continue
        r=r['result']
        for side in ('left','right'):
            p=np.column_stack([r[side],r['ts']])@np.float64(r['basis']).T+np.float64(r['origin'])
            cv2.polylines(canvas,[p.astype(int)],False,(0,200,0),2)
    if line is not None:cv2.polylines(canvas,[np.int32(line)],False,(255,255,0),2)
    cv2.imwrite(str(D/f'centerline_{stage}.jpg'),canvas)
(D/'conveyor_centerline_results.json').write_text(json.dumps(results,indent=2),encoding='utf8')
