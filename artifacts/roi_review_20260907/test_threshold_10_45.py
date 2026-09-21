"""Offline threshold-only replay: no production settings or centreline fallback."""
import json
from pathlib import Path
import runpy
import cv2
import numpy as np

D=Path(__file__).resolve().parent
ns=runpy.run_path(str(D.parents[1]/'tests/test_roi_auto_correct_irfilter.py'),run_name='replay')['NS']
ns['GRID_HOMOGRAPHY_MIN_INLIERS']=10
ns['GRID_HOMOGRAPHY_MIN_INLIER_RATIO']=.45
base_path=D/'CAM4_20260907_182506_suspect_anchor_base.jpg'
current_path=D/'CAM4_20260907_183507_confirm_current.jpg'
base,current=[cv2.imread(str(p),0) for p in (base_path,current_path)]
roi=np.float32([[47,463],[37,268]])
trace=[]
cv2.setRNGSeed(42)
accepted_H,status=ns['estimate_alignment_homography'](base,current,[],roi.tolist(),trace)
row=next(r for r in trace if r['method']=='SIFT' and r.get('H') is not None)
H=np.float64(row['H'])
mapped=cv2.perspectiveTransform(roi.reshape(-1,1,2),H).reshape(-1,2)
good=np.asarray(row['inliers'],bool)
src=np.float32(row['src']);dst=np.float32(row['dst'])
samples=ns['roi_validation_samples']([],roi.tolist())
local=[int(np.sum(np.linalg.norm(src[good]-p,axis=1)<=80)) for p in samples]
count=int(good.sum());ratio=count/len(good)
result=dict(base=base_path.name,current=current_path.name,min_inliers=10,min_ratio=.45,
    matches=len(good),inliers=count,inlier_ratio=ratio,count_ratio_pass=count>=10 and ratio>=.45,
    homography_accepted=accepted_H is not None,status=status,method=row['method'],
    reason=row['reason'],base_roi=roi.tolist(),mapped_roi=mapped.tolist(),
    local_inliers_at_endpoints_and_midpoint=local,attempts=trace)
(D/'threshold_10_45_result.json').write_text(json.dumps(result,indent=2),encoding='utf8')
canvas=np.full((570,1280,3),24,np.uint8)
canvas[48:528,:640]=cv2.cvtColor(base,cv2.COLOR_GRAY2BGR)
canvas[48:528,640:]=cv2.cvtColor(current,cv2.COLOR_GRAY2BGR)
cyan=(255,255,0);orange=(0,165,255)
for points,offset,color in ((roi,[0,48],cyan),(mapped,[640,48],orange)):
    pixels=np.int32(np.rint(points+offset))
    cv2.polylines(canvas,[pixels],False,color,3)
    for p in pixels:cv2.circle(canvas,tuple(p),5,color,-1)
cv2.putText(canvas,'ORIGINAL BASE / original ROI',(12,31),cv2.FONT_HERSHEY_SIMPLEX,.65,cyan,2)
cv2.putText(canvas,'CONFIRM / SIFT homography ROI',(652,31),cv2.FONT_HERSHEY_SIMPLEX,.65,orange,2)
label=f'{count}/{len(good)} inliers ({ratio:.1%}) | 10 / 45% gate: PASS | ROI validation: {row["reason"]}'
cv2.putText(canvas,label,(12,554),cv2.FONT_HERSHEY_SIMPLEX,.6,(235,235,235),1)
cv2.imwrite(str(D/'CAM4_SIFT_10inliers_45percent.png'),canvas)
print(json.dumps({k:v for k,v in result.items() if k!='attempts'},indent=2))
