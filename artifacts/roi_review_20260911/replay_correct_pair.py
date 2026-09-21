import json,runpy,hashlib
from pathlib import Path
import cv2,numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[2]
ns=runpy.run_path(str(ROOT/"tests/test_roi_auto_correct_irfilter.py"))["NS"]
D=ROOT/"images/20260911"
paths=[D/"CAM4_20260911_082751_suspect_anchor_base.jpg",D/"CAM4_20260911_082751_suspect_current.jpg"]
frames=[cv2.imread(str(p)) for p in paths]
a,b=[cv2.cvtColor(f,cv2.COLOR_BGR2GRAY) for f in frames]
normalized=[[0.003125,0.572917],[0.014063,0.989583]]
h,w=a.shape;roi=[[int(round(x*w)),int(round(y*h))] for x,y in normalized]
trace=[];cv2.setRNGSeed(42)
H,status=ns["estimate_alignment_homography"](a,b,[],roi,trace)
candidate=ns["transform_roi_points_h"](roi,H) if H is not None else (ns["transform_roi_points_h"](roi,np.array(trace[-1]["H"])) if trace and trace[-1].get("H") is not None else None)
result=dict(base=str(paths[0]),current=str(paths[1]),base_roi=roi,roi_source="CAM4 cameras.json read from terminal on 2026-09-11",accepted=H is not None,status=status,candidate=candidate,applied_roi=candidate if H is not None else roi,preprocessing="BGR to gray",qualification="Offline projection only; not a live three-suspect confirmation",attempts=trace)
fig,axes=plt.subplots(1,2,figsize=(12,5.4),dpi=160)
for ax,f in zip(axes,frames):
 ax.imshow(cv2.cvtColor(f,cv2.COLOR_BGR2RGB));ax.set_xlim(-5,w);ax.set_ylim(h,0);ax.axis("off")
pts=np.array(roi)
axes[0].plot(pts[:,0],pts[:,1],color="#00e5ff",lw=2,label="Original ROI")
axes[0].set_title("2026-09-11 08:27:51 | BASE",fontsize=11)
axes[1].plot(pts[:,0],pts[:,1],color="#00e5ff",lw=1.7,ls=":",label="Original coordinates")
if candidate is not None:
 pts=np.array(candidate);color="#ffdf00" if H is not None else "#ff3e3e"
 axes[1].plot(pts[:,0],pts[:,1],color=color,lw=2.5,ls="-" if H is not None else "--",label="SIFT projected ROI" if H is not None else "Rejected candidate")
 for x,y in pts:
  axes[1].scatter([x],[y],s=20,color=color)
  axes[1].annotate(str((int(x),int(y))),(x,y),xytext=(8,-12),textcoords="offset points",fontsize=10,color=color,bbox=dict(facecolor="black",alpha=.65,pad=2))
axes[1].set_title("CURRENT | SIFT + homography | "+("PASS" if H is not None else "REJECTED"),fontsize=11)
for ax in axes:ax.legend(loc="upper right",fontsize=8)
fig.text(.03,.025,"Original ROI: "+str(roi)+"   |   Projected ROI: "+str(candidate),fontsize=10)
fig.tight_layout(rect=(0,.06,1,1));fig.savefig(Path(__file__).with_name("CAM4_20260911_082751_roi_projection.png"));plt.close(fig)
Path(__file__).with_name("CAM4_20260911_082751_roi_projection.json").write_text(json.dumps(result,indent=2),encoding="utf8")
print(json.dumps({k:v for k,v in result.items() if k!="attempts"}))
print([(x["matches"],sum(x.get("inliers",[]))) for x in trace])
