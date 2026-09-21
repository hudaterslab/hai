import json
import runpy
from pathlib import Path
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
D = Path(__file__).resolve().parent
ns = runpy.run_path(str(ROOT / 'tests/test_roi_auto_correct_irfilter.py'), run_name='replay')['NS']
a = cv2.imread(str(D / 'CAM4_20260907_182506_suspect_anchor_base.jpg'))
b = cv2.imread(str(D / 'CAM4_20260907_183507_confirm_current.jpg'))
roi = np.float32([[47,463],[37,268]])
rows = []
for preprocess in ('gray', 'jpeg_gray', 'clahe', 'upscale2'):
    ga, gb = [cv2.cvtColor(im,cv2.COLOR_BGR2GRAY) for im in (a,b)]
    if preprocess == 'jpeg_gray':
        ga=cv2.imread(str(D/'CAM4_20260907_182506_suspect_anchor_base.jpg'),0)
        gb=cv2.imread(str(D/'CAM4_20260907_183507_confirm_current.jpg'),0)
    if preprocess == 'clahe':
        c = cv2.createCLAHE(clipLimit=2.0,tileGridSize=(8,8))
        ga,gb=c.apply(ga),c.apply(gb)
    scale=2 if preprocess=='upscale2' else 1
    if scale>1:
        ga,gb=[cv2.resize(im,None,fx=scale,fy=scale,interpolation=cv2.INTER_CUBIC) for im in (ga,gb)]
    detector = cv2.SIFT_create(nfeatures=3000)
    ka,da=detector.detectAndCompute(ga,None)
    kb,db=detector.detectAndCompute(gb,None)
    matcher=cv2.BFMatcher(cv2.NORM_L2)
    ab,ba=matcher.knnMatch(da,db,k=2),matcher.knnMatch(db,da,k=2)
    for ratio in (.7,.75,.8,.85,.9):
        fw=[p[0] for p in ab if len(p)==2 and p[0].distance<ratio*p[1].distance]
        rev={(p[0].trainIdx,p[0].queryIdx) for p in ba if len(p)==2 and p[0].distance<ratio*p[1].distance}
        for mutual in (True,False):
            matches=[m for m in fw if not mutual or (m.queryIdx,m.trainIdx) in rev]
            src=np.float32([ka[m.queryIdx].pt for m in matches])/scale; dst=np.float32([kb[m.trainIdx].pt for m in matches])/scale
            cv2.setRNGSeed(42)
            H,mask=cv2.findHomography(src,dst,cv2.RANSAC,3.0)
            if H is None: continue
            valid,reason=ns['validate_roi_homography'](H,src,dst,mask,[],roi,a.shape[:2])
            good=mask.ravel().astype(bool)
            mapped=cv2.perspectiveTransform(roi.reshape(-1,1,2),H).reshape(-1,2)
            # Diagnostic only: show the next gate without changing production thresholds.
            old=ns['GRID_HOMOGRAPHY_MIN_INLIERS'],ns['GRID_HOMOGRAPHY_MIN_INLIER_RATIO']
            ns['GRID_HOMOGRAPHY_MIN_INLIERS'],ns['GRID_HOMOGRAPHY_MIN_INLIER_RATIO']=4,0
            _,next_reason=ns['validate_roi_homography'](H,src,dst,mask,[],roi,a.shape[:2])
            ns['GRID_HOMOGRAPHY_MIN_INLIERS'],ns['GRID_HOMOGRAPHY_MIN_INLIER_RATIO']=old
            row=dict(preprocess=preprocess,ratio=ratio,mutual=mutual,matches=len(matches),inliers=int(good.sum()),accepted=valid,reason=reason,next_gate=next_reason,mapped_roi=mapped.tolist(),local_counts=[int((np.linalg.norm(src[good]-p,axis=1)<80).sum()) for p in roi])
            rows.append(row)
            if preprocess=='gray' and mutual and ratio in (.7,.8,.9):
                canvas=np.hstack([a.copy(),b.copy()])
                for i,(s,t,g) in enumerate(zip(src,dst,good)):
                    color=(0,220,0) if g else (0,0,220)
                    s=tuple(np.round(s).astype(int)); t=tuple(np.round(t+[640,0]).astype(int))
                    cv2.line(canvas,s,t,color,1)
                    cv2.circle(canvas,s,3,color,-1);cv2.circle(canvas,t,3,color,-1)
                cv2.polylines(canvas,[roi.astype(int)],False,(255,255,0),3)
                cv2.polylines(canvas,[(mapped+[640,0]).astype(int)],False,(255,255,0),3)
                cv2.imwrite(str(D/f'matches_{ratio}.jpg'),canvas)
                overlay=b.copy()
                cv2.polylines(overlay,[mapped.astype(int)],False,(0,0,255),3)
                cv2.imwrite(str(D/f'roi_proposed_{ratio}.jpg'),overlay)
(D/'matching_experiments.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
for row in rows: print(json.dumps(row))
