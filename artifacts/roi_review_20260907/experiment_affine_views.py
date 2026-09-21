"""Test viewpoint normalization without weakening correspondence checks."""
import json
from pathlib import Path
import runpy
import cv2
import numpy as np

D=Path(__file__).resolve().parent
NS=runpy.run_path(str(D.parents[1]/'tests/test_roi_auto_correct_irfilter.py'),run_name='replay')['NS']
roi=[[47,463],[37,268]]
a,b=[cv2.imread(str(D/f),0) for f in ('CAM4_20260907_182506_suspect_anchor_base.jpg','CAM4_20260907_183507_confirm_current.jpg')]
sift=cv2.SIFT_create(nfeatures=6000,contrastThreshold=.01)
kb,db=sift.detectAndCompute(b,None)
rows=[]
all_src=[];all_dst=[]
for sx in (.5,.7,1.,1.4,2.):
    for shear in (-.3,0.,.3):
        A=np.float32([[sx,shear,max(0,-shear*a.shape[0])],[0,1,0]])
        im=cv2.warpAffine(a,A,(int(sx*a.shape[1]+abs(shear)*a.shape[0]),a.shape[0]))
        ka,da=sift.detectAndCompute(im,None)
        bf=cv2.BFMatcher()
        fw=bf.knnMatch(da,db,k=2);bw=bf.knnMatch(db,da,k=2)
        rev={(m.trainIdx,m.queryIdx) for m,n in bw if m.distance<.7*n.distance}
        ms=[m for m,n in fw if m.distance<.7*n.distance and (m.queryIdx,m.trainIdx) in rev]
        p=np.float32([ka[m.queryIdx].pt for m in ms]).reshape(-1,1,2)
        src=cv2.transform(p,cv2.invertAffineTransform(A)).reshape(-1,2)
        dst=np.float32([kb[m.trainIdx].pt for m in ms])
        all_src.extend(src);all_dst.extend(dst)
        counts=[int(np.sum(np.linalg.norm(src-p,axis=1)<80)) for p in np.float32(roi)]
        rows.append(dict(sx=sx,shear=shear,matches=len(ms),local_matches=counts))
src,dst=NS['unique_roi_matches'](np.float32(all_src),np.float32(all_dst),3)
H,mask=cv2.findHomography(src,dst,cv2.USAC_MAGSAC,3,maxIters=10000)
ok,reason=NS['validate_roi_homography'](H,src,dst,mask,[],roi,a.shape)
good=mask.ravel().astype(bool)
result=dict(views=rows,matches=len(src),inliers=int(good.sum()),accepted=ok,reason=reason,
    local_counts=[int(np.sum(np.linalg.norm(src[good]-p,axis=1)<80)) for p in np.float32(roi)],
    src=src.tolist(),dst=dst.tolist(),mask=mask.ravel().tolist(),H=H.tolist())
(D/'affine_views_results.json').write_text(json.dumps(result,indent=2),encoding='utf8')
print({k:v for k,v in result.items() if k not in ('src','dst','mask','H','views')})
