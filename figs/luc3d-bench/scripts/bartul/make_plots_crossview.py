#!/usr/bin/env python
"""Combined Bartul figure: within-view IDF1 + CROSS-VIEW IDF1 + 2D ID switches."""
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path("/root/vast/eric/luc3d-bench/outputs/bartul_smoke")
PLOTS = OUT/"plots"; PLOTS.mkdir(exist_ok=True)
SER = ["21241563","21369048","21372315","21372316","22085397"]
COL = {"Liezl — 3D-weight×6":"#d24a00","Liezl (3D)":"#8bc34a","LUC3D (3D)":"#1f6fd0",
       "SLEAP per-camera 2D (ref)":"#8a8a8a","ByteTrack":"#c2185b","3D-MuPPET (SORT+xview)":"#00897b"}

cv = pd.read_csv(OUT/"bartul_crossview_metric.csv")
ev = pd.read_csv(OUT/"bartul_eval_per_camera.csv", dtype={"camera":str})
sw = ev.groupby("tracker")["num_switches"].sum()
cv["sw2d"] = cv["tracker"].map(sw).fillna(0)
cv = cv.sort_values("crossview_idf1", ascending=True)   # best cross-view on top (barh)
labels = cv.tracker.tolist(); y = np.arange(len(labels))
colors = [COL.get(l,"#777") for l in labels]

fig, ax = plt.subplots(1, 3, figsize=(18, 0.62*len(labels)+2.4))
# A: within-view IDF1
ax[0].barh(y, cv.per_camera_idf1, color=colors, edgecolor="black", lw=.7)
for i,v in enumerate(cv.per_camera_idf1): ax[0].text(v+.01,i,f"{v:.3f}",va="center",fontsize=10)
ax[0].set_xlim(0,1.05); ax[0].set_yticks(y); ax[0].set_yticklabels(labels,fontsize=10)
ax[0].set_title("A. Within-view Identity F1\n(per-camera, current metric)",fontweight="bold")
ax[0].set_xlabel("IDF1 (higher better)")
# B: cross-view IDF1  (NEW)
ax[1].barh(y, cv.crossview_idf1, color=colors, edgecolor="black", lw=.7)
for i,v in enumerate(cv.crossview_idf1): ax[1].text(v+.01,i,f"{v:.3f}",va="center",fontweight="bold",fontsize=10)
ax[1].set_xlim(0,1.05); ax[1].set_yticks(y); ax[1].set_yticklabels([])
ax[1].set_title("B. CROSS-VIEW Identity F1\n(one global ID per animal across all 5 cams)",fontweight="bold")
ax[1].set_xlabel("IDF1 (higher better)")
ax[1].axvspan(0,0.35,color="red",alpha=0.05)
# C: 2D per-camera ID switches
mx=cv.sw2d.max()
ax[2].barh(y, cv.sw2d, color=colors, edgecolor="black", lw=.7)
for i,v in enumerate(cv.sw2d): ax[2].text(v+mx*0.01,i,f"{int(v):,}",va="center",fontsize=10)
ax[2].set_yticks(y); ax[2].set_yticklabels([]); ax[2].set_xlim(0,mx*1.18+1)
ax[2].set_title("C. 2D ID switches\n(per-camera, raw sum)",fontweight="bold")
ax[2].set_xlabel("switches (lower better)")

fig.suptitle("Bartul session 20250827_141755 (2-animal, 5 cams) — within-view vs CROSS-VIEW identity\n"
    "Detections = raw SLEAP keypoints (identity stripped); GT = reprojected proofread 3D; first 9,000 frames.\n"
    "Panel B is the new cross-view metric: SLEAP/ByteTrack are per-camera (no global identity) → collapse to ~0.2; "
    "Liezl holds one identity across all cameras.", fontsize=10.5, fontweight="bold")
fig.tight_layout(rect=[0,0,1,0.90])
for e in ("png","pdf"): fig.savefig(PLOTS/f"figure_crossview_bartul.{e}", dpi=140, bbox_inches="tight")
print("wrote", PLOTS/"figure_crossview_bartul.png")
EOF_MARKER = None
