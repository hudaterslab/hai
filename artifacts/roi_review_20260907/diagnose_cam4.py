"""Stage-by-stage audit: detection, ratio test, mutual check, RANSAC."""
import json
from pathlib import Path
import cv2
import numpy as np

D = Path(__file__).resolve().parent
ROI = np.float32([[47,463],[37,268],[42,365.5]])


def audit(a, b, label, samples=ROI):
    detector = cv2.SIFT_create(nfeatures=0, contrastThreshold=0.01)
    ka, da = detector.detectAndCompute(a,None)
    kb, db = detector.detectAndCompute(b,None)
    src = np.float32([k.pt for k in ka])
    dst = np.float32([k.pt for k in kb])
    bf = cv2.BFMatcher()
    fw, bw = bf.knnMatch(da,db,k=2), bf.knnMatch(db,da,k=2)
    ratio = np.array([m.distance/max(n.distance,1e-9) for m,n in fw])
    reverse = {(m.trainIdx,m.queryIdx) for m,n in bw if m.distance < .7*n.distance}
    passes = ratio < .7
    mutual = passes & np.array([(m.queryIdx,m.trainIdx) in reverse for m,n in fw])
    matches = [m for i,(m,n) in enumerate(fw) if mutual[i]]
    H, mask = cv2.findHomography(np.float32([ka[m.queryIdx].pt for m in matches]),
                                np.float32([kb[m.trainIdx].pt for m in matches]),cv2.RANSAC,3)
    good = np.zeros(len(ka),bool)
    for m,g in zip(matches,mask.ravel()):
        good[m.queryIdx] = bool(g)
    rows = []
    for point in samples:
        near = np.linalg.norm(src-point,axis=1)<80
        rows.append(dict(point=point.tolist(),detected=int(near.sum()),
            ratio_pass=int((near&passes).sum()),mutual_pass=int((near&mutual).sum()),
            inliers=int((near&good).sum()),
            median_ratio=float(np.median(ratio[near])) if near.any() else None,
            best_candidates=[dict(source=src[i].tolist(),target=dst[fw[i][0].trainIdx].tolist(),
                                 ratio=float(ratio[i])) for i in np.flatnonzero(near)]))
    canvas = cv2.cvtColor(np.hstack([a,b]),cv2.COLOR_GRAY2BGR)
    for i,p in enumerate(src):
        if not np.any(np.linalg.norm(samples-p,axis=1)<80): continue
        color=(0,220,0) if good[i] else ((0,200,255) if mutual[i] else (0,0,255))
        cv2.circle(canvas,tuple(np.int32(p)),2,color,-1)
    cv2.polylines(canvas,[np.int32(samples[:2])],False,(255,255,0),2)
    cv2.imwrite(str(D/f'{label}_detections.jpg'),canvas)
    return dict(label=label,features=[len(ka),len(kb)],samples=rows)


if __name__ == '__main__':
    a,b,c = [cv2.imread(str(D/f),0) for f in (
        'CAM4_20260907_182506_suspect_anchor_base.jpg',
        'CAM4_20260907_182506_suspect_current.jpg',
        'CAM4_20260907_183507_confirm_current.jpg')]
    results = [audit(a,c,'cam4_base_confirm'),
               audit(b,c,'cam4_suspect_confirm',np.float32([[270,440],[280,250],[275,345]]))]
    (D/'cam4_matching_diagnosis.json').write_text(json.dumps(results,indent=2),encoding='utf8')
    for r in results:
        print(r['label'],r['features'])
        for row in r['samples']:
            print({k:v for k,v in row.items() if k!='best_candidates'})
