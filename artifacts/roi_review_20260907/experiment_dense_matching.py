"""Local LoFTR experiment. Images stay on this computer; downloads model weights only."""
import argparse
import copy
import json
from pathlib import Path
import runpy
import time
import urllib.request
import cv2
import numpy as np
import torch
from kornia.feature import LoFTR
from kornia.feature.loftr.loftr import default_cfg

D = Path(__file__).resolve().parent
NS = runpy.run_path(str(D.parents[1]/'tests/test_roi_auto_correct_irfilter.py'),run_name='replay')['NS']
ROI = np.float32([[47,463],[37,268]])


def assess(a,b,src,dst,confidence,label):
    order = np.argsort(-confidence)
    src,dst = NS['unique_roi_matches'](src[order],dst[order],3)
    rows=[]
    for region in ('global','roi'):
        keep = np.ones(len(src),bool)
        if region=='roi':
            m=NS['roi_feature_mask'](a.shape,[],ROI.tolist())
            keep=m[np.clip(src[:,1].astype(int),0,a.shape[0]-1),np.clip(src[:,0].astype(int),0,a.shape[1]-1)]>0
        s,t=src[keep],dst[keep]
        if len(s)<4: continue
        cv2.setRNGSeed(42)
        H,mask=cv2.findHomography(s,t,cv2.USAC_MAGSAC,3,maxIters=10000,confidence=.999)
        ok,reason=NS['validate_roi_homography'](H,s,t,mask,[],ROI.tolist(),a.shape)
        good=mask.ravel().astype(bool)
        mapped=cv2.perspectiveTransform(ROI.reshape(-1,1,2),H).reshape(-1,2)
        samples=NS['roi_validation_samples']([],ROI.tolist())
        counts=[int(np.sum(np.linalg.norm(s[good]-p,axis=1)<80)) for p in samples]
        row=dict(label=label,region=region,matches=len(s),inliers=int(good.sum()),accepted=ok,
                 reason=reason,local_counts=counts,mapped_roi=mapped.tolist(),H=H.tolist(),
                 src=s.tolist(),dst=t.tolist(),mask=mask.ravel().tolist())
        rows.append(row)
        print({k:v for k,v in row.items() if k not in ('H','src','dst','mask')},flush=True)
        canvas=cv2.cvtColor(np.hstack([a,b]),cv2.COLOR_GRAY2BGR)
        for p,q in zip(s[good],t[good]):
            cv2.line(canvas,tuple(np.int32(p)),tuple(np.int32(q+[a.shape[1],0])),(0,180,0),1)
        cv2.polylines(canvas,[np.int32(ROI)],False,(255,255,0),2)
        cv2.polylines(canvas,[np.int32(mapped+[a.shape[1],0])],False,(255,255,0),2)
        cv2.putText(canvas,label+' '+region+(' ACCEPTED' if ok else ' REJECTED'),(8,24),cv2.FONT_HERSHEY_SIMPLEX,.5,(0,0,255),1)
        cv2.imwrite(str(D/f'{label}_{region}.jpg'),canvas)
    return rows


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--model',default='outdoor',choices=['outdoor','indoor_new'])
    args=parser.parse_args()
    torch.set_num_threads(4)
    filename='loftr_outdoor.ckpt' if args.model=='outdoor' else 'loftr_indoor_ds_new.ckpt'
    weights=Path.home()/'.cache/codex-roi-matching'/filename
    if not weights.exists():
        print('Downloading',filename,flush=True)
        urllib.request.urlretrieve('https://huggingface.co/kornia/loftr/resolve/main/'+filename,weights)
    config=copy.deepcopy(default_cfg)
    config['coarse']['temp_bug_fix']=args.model=='indoor_new'
    model=LoFTR(pretrained=None,config=config).eval()
    model.load_state_dict(torch.load(weights,map_location='cpu',weights_only=True)['state_dict'])
    a=cv2.imread(str(D/'CAM4_20260907_182506_suspect_anchor_base.jpg'),0)
    results=[]
    for stage in ('182506_suspect','183507_confirm'):
        b=cv2.imread(str(D/f'CAM4_20260907_{stage}_current.jpg'),0)
        started=time.perf_counter()
        with torch.inference_mode():
            pred=model(dict(image0=torch.from_numpy(a.copy()).float()[None,None]/255,
                            image1=torch.from_numpy(b.copy()).float()[None,None]/255))
        src,dst,conf=[pred[k].cpu().numpy() for k in ('keypoints0','keypoints1','confidence')]
        label=f'loftr_{args.model}_{stage}'
        np.savez_compressed(D/f'{label}.npz',src=src,dst=dst,confidence=conf)
        print(label,'matches',len(src),'seconds',time.perf_counter()-started,flush=True)
        results.extend(assess(a,b,src,dst,conf,label))
    (D/f'loftr_{args.model}_results.json').write_text(json.dumps(results,indent=2),encoding='utf8')
