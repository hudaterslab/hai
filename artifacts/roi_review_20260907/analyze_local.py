"""Offline ROI neighbourhood experiment; does not modify camera configuration."""
import json
from pathlib import Path
import cv2
import numpy as np

D=Path(__file__).resolve().parent
a=cv2.imread(str(D/'CAM4_20260907_182506_suspect_anchor_base.jpg'))
b=cv2.imread(str(D/'CAM4_20260907_183507_confirm_current.jpg'))
roi=np.float32([[47,463],[37,268]])
rows=[]
for region in ('no_roof','roi_near'):
 for scale in (1,2):
  for contrast in (.04,.01):
   ga,gb=[cv2.cvtColor(im,cv2.COLOR_BGR2GRAY) for im in (a,b)]
   masks=[np.zeros(ga.shape,np.uint8),np.zeros(gb.shape,np.uint8)]
   masks[0][160:,:160 if region=='roi_near' else 640]=255
   masks[1][160:,:]=255
   if scale==2:
    ga,gb=[cv2.resize(im,None,fx=2,fy=2,interpolation=cv2.INTER_CUBIC) for im in (ga,gb)]
    masks=[cv2.resize(im,None,fx=2,fy=2,interpolation=cv2.INTER_NEAREST) for im in masks]
   sift=cv2.SIFT_create(nfeatures=5000,contrastThreshold=contrast)
   ka,da=sift.detectAndCompute(ga,masks[0]);kb,db=sift.detectAndCompute(gb,masks[1])
   bf=cv2.BFMatcher(cv2.NORM_L2)
   ab,ba=bf.knnMatch(da,db,k=2),bf.knnMatch(db,da,k=2)
   for ratio in (.7,.8,.9):
    rev={(p[0].trainIdx,p[0].queryIdx) for p in ba if len(p)==2 and p[0].distance<ratio*p[1].distance}
    matches=[p[0] for p in ab if len(p)==2 and p[0].distance<ratio*p[1].distance and (p[0].queryIdx,p[0].trainIdx) in rev]
    row=dict(region=region,scale=scale,contrast=contrast,ratio=ratio,features=[len(ka),len(kb)],matches=len(matches))
    if len(matches)<4:
     rows.append(row);continue
    src=np.float32([ka[m.queryIdx].pt for m in matches])/scale;dst=np.float32([kb[m.trainIdx].pt for m in matches])/scale
    cv2.setRNGSeed(42)
    H,mask=cv2.findHomography(src,dst,cv2.RANSAC,3)
    if H is None:
     rows.append(row);continue
    good=mask.ravel().astype(bool)
    mapped=cv2.perspectiveTransform(roi.reshape(-1,1,2),H).reshape(-1,2)
    hull=cv2.convexHull(src[good]); distances=[cv2.pointPolygonTest(hull,tuple(map(float,p)),True) for p in roi]
    row.update(inliers=int(good.sum()),mapped_roi=mapped.tolist(),local_counts=[int((np.linalg.norm(src[good]-p,axis=1)<80).sum()) for p in roi],hull_distances=distances)
    rows.append(row)
    if region=='roi_near' and scale==1 and contrast==.04 and ratio in (.7,.9):
     canvas=np.hstack([a.copy(),b.copy()])
     for s,t,g in zip(src,dst,good):
      color=(0,220,0) if g else (0,0,220)
      s=tuple(np.round(s).astype(int));t=tuple(np.round(t+[640,0]).astype(int))
      cv2.line(canvas,s,t,color,1);cv2.circle(canvas,s,3,color,-1);cv2.circle(canvas,t,3,color,-1)
     cv2.polylines(canvas,[roi.astype(int)],False,(255,255,0),3)
     cv2.polylines(canvas,[(mapped+[640,0]).astype(int)],False,(255,255,0),3)
     cv2.imwrite(str(D/f'local_matches_{ratio}.jpg'),canvas)
(D/'local_experiments.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
for r in rows: print(json.dumps(r))
