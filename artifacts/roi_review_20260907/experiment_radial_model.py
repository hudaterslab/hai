"""Test whether lens distortion explains loss of a single homography consensus."""
import json
from pathlib import Path
import cv2
import numpy as np
from scipy.optimize import least_squares

D=Path(__file__).resolve().parent
z=np.load(D/'loftr_outdoor_183507_confirm.npz')
src,dst=z['src'],z['dst']
center=np.float64([320,240]);scale=640
roi=np.float64([[47,463],[37,268]])


def undistort(p,k):
    q=(p-center)/scale
    return q/(1+k*np.sum(q*q,axis=1)[:,None])


def distort(q,k):
    r2=np.sum(q*q,axis=1)
    factor=2/(1+np.sqrt(np.maximum(1-4*k*r2,1e-9)))
    return q*factor[:,None]*scale+center


def project(p,H,k):
    u=undistort(p,k)
    v=cv2.perspectiveTransform(u.reshape(-1,1,2),H).reshape(-1,2)
    return distort(v,k)


def residual(params,s,t):
    H=np.append(params[:8],1).reshape(3,3)
    return ((project(s,H,params[8])-t)).ravel()


rows=[]
for k in np.linspace(-1.2,.4,17):
    u,v=undistort(src,k),undistort(dst,k)
    cv2.setRNGSeed(42)
    H,mask=cv2.findHomography(u,v,cv2.USAC_MAGSAC,4/scale,maxIters=10000)
    error=np.linalg.norm(project(src,H,k)-dst,axis=1)
    good=error<8
    p=np.r_[H.ravel()[:8],k]
    fit=least_squares(residual,p,args=(src[good],dst[good]),loss='soft_l1',f_scale=2,
                      bounds=([-np.inf]*8+[-1.5],[np.inf]*8+[.5]),max_nfev=300)
    H=np.append(fit.x[:8],1).reshape(3,3);k=float(fit.x[8])
    err=np.linalg.norm(project(src,H,k)-dst,axis=1)
    good=err<3
    rows.append(dict(k=k,H=H.tolist(),inliers=int(good.sum()),error_p90=float(np.percentile(err[good],90)),
                     mapped=project(roi,H,k).tolist(),mask=good.tolist()))
rows.sort(key=lambda r:r['inliers'],reverse=True)
(D/'radial_model_results.json').write_text(json.dumps(rows,indent=2))
for row in rows[:5]:print({k:v for k,v in row.items() if k not in ('H','mask')})
best=rows[0];H=np.float64(best['H']);k=best['k']
a,b=[cv2.imread(str(D/f)) for f in ('CAM4_20260907_182506_suspect_anchor_base.jpg','CAM4_20260907_183507_confirm_current.jpg')]
canvas=b.copy()
lines=json.loads((D/'structural_lines.json').read_text())[0]
for i in (48,51,55,84):
    l=np.float64(lines[i]).reshape(2,2)
    p=np.linspace(l[0],l[1],40)
    mapped=project(p,H,k)
    cv2.polylines(canvas,[mapped.astype(int)],False,(0,0,255),2)
cv2.polylines(canvas,[project(roi,H,k).astype(int)],False,(255,255,0),2)
cv2.imwrite(str(D/'radial_model_conveyor.jpg'),canvas)
