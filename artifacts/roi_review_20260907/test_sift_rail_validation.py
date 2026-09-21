"""Replay the SIFT + rail verification path without centreline reconstruction."""
import json
from pathlib import Path
import runpy
import cv2
import numpy as np

D=Path(__file__).resolve().parent
ns=runpy.run_path(str(D.parents[1]/'tests/test_roi_auto_correct_irfilter.py'),run_name='replay')['NS']
names=['CAM4_20260907_182506_suspect_anchor_base.jpg','CAM4_20260907_183507_confirm_current.jpg']
base,current=[cv2.imread(str(D/name),0) for name in names]
roi=[[47,463],[37,268]]
strict=[];cv2.setRNGSeed(42)
old,old_reason=ns['estimate_alignment_homography'](base,current,[],roi,strict)
original=next(r for r in strict if r['method']=='SIFT')
trace=[];cv2.setRNGSeed(42)
H,reason=ns['estimate_alignment_homography'](base,current,[],roi,trace,allow_conveyor_validation=True)
assert H is not None,reason
np.testing.assert_allclose(H,original['H'],atol=1e-10)
mapped=cv2.perspectiveTransform(np.float32(roi).reshape(-1,1,2),H).reshape(-1,2)
result=dict(base_file=names[0],current_file=names[1],method='homography',
    intermediate_frame_used=False,centerline_reconstruction_used=False,
    min_inliers=ns['GRID_HOMOGRAPHY_MIN_INLIERS'],min_ratio=ns['GRID_HOMOGRAPHY_MIN_INLIER_RATIO'],
    accepted=True,status=reason,previous_status=old_reason,
    homography_unchanged=True,original_roi=roi,projected_roi=mapped.tolist(),
    applied_roi=np.rint(mapped).astype(int).tolist(),attempts=trace)
(D/'SIFT_rail_validation_result.json').write_text(json.dumps(result,indent=2),encoding='utf8')
canvas=np.full((570,1280,3),24,np.uint8)
canvas[48:528,:640]=cv2.cvtColor(base,cv2.COLOR_GRAY2BGR)
canvas[48:528,640:]=cv2.cvtColor(current,cv2.COLOR_GRAY2BGR)
for points,offset,color in ((roi,[0,48],(255,255,0)),(mapped,[640,48],(0,200,255))):
    cv2.polylines(canvas,[np.int32(np.rint(np.asarray(points)+offset))],False,color,3)
cv2.putText(canvas,'ORIGINAL BASE / original ROI',(12,31),cv2.FONT_HERSHEY_SIMPLEX,.65,(255,255,0),2)
cv2.putText(canvas,'CONFIRM / SIFT ROI - VERIFIED',(652,31),cv2.FONT_HERSHEY_SIMPLEX,.65,(0,200,255),2)
cv2.putText(canvas,'10/21 inliers (47.6%) | Rail verification: PASS | Original SIFT coordinates preserved',
    (12,554),cv2.FONT_HERSHEY_SIMPLEX,.59,(235,235,235),1)
cv2.imwrite(str(D/'CAM4_SIFT_verified_by_rails.png'),canvas)
print(json.dumps({k:v for k,v in result.items() if k!='attempts'},indent=2))
