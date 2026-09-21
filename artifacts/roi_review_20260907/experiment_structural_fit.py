"""Diagnostic point + rail fit; no operational acceptance based on manually selected rails."""
import json
from pathlib import Path
import cv2
import numpy as np
from scipy.optimize import least_squares
import runpy

D=Path(__file__).resolve().parent
# Explicit diagnostic annotations: these rail identities must NOT become camera-specific production constants.
lines=json.loads((D/'structural_lines.json').read_text())
source_left=np.float64([lines[0][i] for i in (48,55)]).reshape(-1,2)
source_right=np.float64([lines[0][84]]).reshape(-1,2)
target_left=np.float64([lines[1][i] for i in (34,37,46,51)]).reshape(-1,2)
target_right=np.float64([lines[1][i] for i in (31,74)]).reshape(-1,2)
rail_src=[np.concatenate([np.linspace(s[i],s[i+1],8) for i in range(0,len(s),2)]) for s in (source_left,source_right)]
rail_target=[]
for points in (target_left,target_right):
    vx,vy,x,y=cv2.fitLine(points.astype(np.float32),cv2.DIST_L2,0,.01,.01).ravel()
    rail_target.append(np.float64([-vy,vx,vy*x-vx*y]))
center=np.float64([320,240]);scale=640
z=np.load(D/'loftr_outdoor_183507_confirm.npz');src,dst=z['src'],z['dst']
initial=json.loads((D/'radial_model_results.json').read_text())[0]
H=np.float64(initial['H']);k=initial['k']
def project(p,H,k):
    q=(p-center)/scale
    q=q/(1+k*np.sum(q*q,axis=1)[:,None])
    q=cv2.perspectiveTransform(q.reshape(-1,1,2),H).reshape(-1,2)
    r2=np.sum(q*q,axis=1)
    factor=2/(1+np.sqrt(np.maximum(1-4*k*r2,1e-9)))
    return q*factor[:,None]*scale+center
good=np.linalg.norm(project(src,H,k)-dst,axis=1)<5
s,t=src[good],dst[good]
def residual(p,s,t,weight):
    H=np.append(p[:8],1).reshape(3,3);k=p[8]
    out=[(project(s,H,k)-t).ravel()]
    for pts,line in zip(rail_src,rail_target):
        mapped=project(pts,H,k)
        out.append(weight*(mapped@line[:2]+line[2]))
    return np.concatenate(out)
rows=[]
roi=np.float64([[47,463],[37,268]])
for weight in (1,2,4,8):
    fit=least_squares(residual,np.r_[H.ravel()[:8],k],args=(s,t,weight),loss='soft_l1',f_scale=2,max_nfev=300)
    Hfit=np.append(fit.x[:8],1).reshape(3,3);kfit=fit.x[8]
    rows.append(dict(weight=weight,k=float(kfit),mapped_roi=project(roi,Hfit,kfit).tolist(),
        point_p90=float(np.percentile(np.linalg.norm(project(s,Hfit,kfit)-t,axis=1),90)),
        rail_error=[float(np.mean(np.abs(project(pts,Hfit,kfit)@line[:2]+line[2]))) for pts,line in zip(rail_src,rail_target)]))
    canvas=cv2.imread(str(D/'CAM4_20260907_183507_confirm_current.jpg'))
    for pts in rail_src:
        cv2.polylines(canvas,[project(pts,Hfit,kfit).astype(int)],False,(0,0,255),2)
    cv2.polylines(canvas,[project(roi,Hfit,kfit).astype(int)],False,(255,255,0),2)
    cv2.imwrite(str(D/f'structural_fit_{weight}.jpg'),canvas)
(D/'structural_fit_results.json').write_text(json.dumps(rows,indent=2))
for row in rows:print(row)
