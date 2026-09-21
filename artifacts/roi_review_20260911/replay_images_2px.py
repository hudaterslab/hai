import json,runpy,hashlib
from pathlib import Path
import cv2,numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[2]
ns=runpy.run_path(str(ROOT/"tests/test_roi_auto_correct_irfilter.py"))["NS"]
gns=runpy.run_path(str(ROOT/"tests/test_suspect_only_flow.py"))["NS"]
D=ROOT/"images/20260907"
paths=[D/"CAM4_20260907_182506_suspect_anchor_base.jpg",D/"CAM4_20260907_182506_suspect_current.jpg",D/"CAM4_20260907_183507_confirm_current.jpg"]
frames=[cv2.imread(str(p)) for p in paths]
roi=[[47,463],[37,268]]
a=gns["AnchorTrackingROIAligner"]()
base=a._gray_plain(frames[0]);a.anchor_slots["updated"]={"gray":base}
result=dict(threshold=gns["GRID_SHAKE_THRESHOLD_PX"],base_roi=roi,base_file=str(paths[0]),source_sha256=hashlib.sha256((ROOT/"multi_event_irfilter.py").read_bytes()).hexdigest(),note="Offline candidates; saved images do not establish three consecutive suspect observations.",observations=[])
fig,axes=plt.subplots(1,3,figsize=(18,5.3),dpi=150)
for i,(ax,frame,path) in enumerate(zip(axes,frames,paths)):
 ax.imshow(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB));ax.set_xlim(0,639);ax.set_ylim(575,0);ax.axis("off")
 old=np.array(roi);ax.plot(old[:,0],old[:,1],color="#00e5ff",lw=2,label="Original ROI")
 if i==0:
  ax.set_title("BASE | original ROI",fontsize=12)
 else:
  grid=a.detect_grid_camera_motion(frame);cv2.setRNGSeed(42);trace=[]
  H,status=ns["estimate_alignment_homography"](base,a._gray_plain(frame),[],roi,trace)
  mapped=ns["transform_roi_points_h"](roi,H) if H is not None else None
  result["observations"].append(dict(file=str(path),grid=grid,status=status,accepted=H is not None,projected_roi=mapped,attempts=trace))
  if mapped is not None:
   pts=np.array(mapped);ax.plot(pts[:,0],pts[:,1],color="#ffdf00",lw=2.5,label="SIFT ROI (accepted)");ax.scatter(pts[:,0],pts[:,1],s=22,color="#ffdf00")
  else:
   if trace and trace[-1].get("H") is not None:
    rejected=ns["transform_roi_points_h"](roi,np.array(trace[-1]["H"]))
    result["observations"][-1]["rejected_candidate"]=rejected
    pts=np.array(rejected);ax.plot(pts[:,0],pts[:,1],color="#ff3e3e",lw=2,ls="--",label="Rejected candidate")
    ax.scatter(pts[:,0],pts[:,1],s=22,color="#ff3e3e")
    for x,y in pts:ax.text(x+8,y,str((int(x),int(y))),fontsize=8,color="#c00000")
  ax.set_title(("SUSPECT image" if i==1 else "CONFIRM image")+" | "+("PASS" if H is not None else "REJECTED"),fontsize=12)
  ax.text(.02,-.08,f"Grid >2px: {grid['n_moving']}/{grid['n_measurable']} | ROI: {mapped}",transform=ax.transAxes,fontsize=9)
 ax.legend(loc="upper right",fontsize=8)
fig.suptitle("Current SIFT + homography | cyan: unchanged ROI; red dashed: rejected SIFT candidate",fontsize=13)
fig.tight_layout();fig.savefig(Path(__file__).with_name("CAM4_images_2px.png"),bbox_inches="tight");plt.close(fig)
Path(__file__).with_name("CAM4_images_2px.json").write_text(json.dumps(result,indent=2),encoding="utf8")
for r in result["observations"]:print(Path(r["file"]).name,r["grid"]["n_moving"],r["grid"]["n_measurable"],r["status"],r["projected_roi"])

# Compare legacy JPEG grayscale decoding with the runtime BGR-to-gray preprocessing.
gray0=cv2.imread(str(paths[0]),0);gray1=cv2.imread(str(paths[2]),0)
cv2.setRNGSeed(42)
legacyH,legacyStatus=ns["estimate_alignment_homography"](gray0,gray1,[],roi)
legacy=ns["transform_roi_points_h"](roi,legacyH) if legacyH is not None else None
result["legacy_jpeg_gray_replay"]=dict(status=legacyStatus,roi=legacy)
fig,axes=plt.subplots(1,2,figsize=(12,5.8),dpi=150)
for ax in axes:
 ax.imshow(cv2.cvtColor(frames[2],cv2.COLOR_BGR2RGB));ax.set_xlim(0,639);ax.set_ylim(575,0);ax.axis("off")
old=np.array(roi);axes[1].plot(old[:,0],old[:,1],color="#00e5ff",lw=2,label="Actual ROI: unchanged")
pts=np.array(legacy);axes[0].plot(pts[:,0],pts[:,1],color="#ffdf00",lw=2.5,label="Accepted by earlier replay")
pts=np.array(result["observations"][-1]["rejected_candidate"]);axes[1].plot(pts[:,0],pts[:,1],color="#ff3e3e",lw=2,ls="--",label="Rejected candidate")
for ax,pts,title in [(axes[0],np.array(legacy),"Earlier replay: JPEG decoded as grayscale"),(axes[1],pts,"Current code preprocessing: BGR to gray")]:
 ax.set_title(title,fontsize=11)
 for x,y in pts:ax.scatter([x],[y],s=18,color="red");ax.text(x+8,y,str((int(x),int(y))),fontsize=9,color="red")
 ax.legend(loc="upper right",fontsize=8)
axes[0].text(.02,-.02,"10/21 inliers | PASS",transform=axes[0].transAxes)
axes[1].text(.02,-.02,"10/22 inliers | REJECTED: endpoint y=545 > 479",transform=axes[1].transAxes)
fig.tight_layout();fig.savefig(Path(__file__).with_name("CAM4_preprocessing_comparison.png"),bbox_inches="tight");plt.close(fig)
Path(__file__).with_name("CAM4_images_2px.json").write_text(json.dumps(result,indent=2),encoding="utf8")
