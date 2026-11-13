#!/usr/bin/env python
"""Aggregate BMimica figure: Liezl x6 vs ByteTrack over 50 sessions —
within-view IDF1, cross-view IDF1, 2D switches."""
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

BM = Path("/root/vast/eric/luc3d-bench/outputs/bmimica")
df = pd.read_csv(BM/"bmimica_eval_per_session.csv")
n = len(df)
rows = []
for tr, lab, col in [("liezl", "Liezl — 3D-weight×6", "#d24a00"),
                     ("muppet", "3D-MuPPET (SORT+xview)", "#00897b"),
                     ("byte", "ByteTrack", "#c2185b")]:
    if f"{tr}_idf1" not in df: continue
    rows.append(dict(label=lab, color=col,
        within=df[f"{tr}_idf1"].mean(), xview=df[f"{tr}_xview_idf1"].mean(),
        sw2d=df[f"{tr}_sw2d"].sum()))
R = pd.DataFrame(rows)
fig, ax = plt.subplots(1, 3, figsize=(16, 2.6))
y = np.arange(len(R))
for a, key, title, xlim in [(ax[0], "within", "A. Within-view IDF1 (per-camera)", 1.05),
                             (ax[1], "xview", "B. CROSS-VIEW IDF1 (one global ID/animal)", 1.05),
                             (ax[2], "sw2d", "C. 2D ID switches (total)", None)]:
    vals = R[key].values
    a.barh(y, vals, color=R.color, edgecolor="black", lw=.7)
    for i, v in enumerate(vals):
        a.text(v + (0.01 if xlim else vals.max()*0.01), i, f"{v:.3f}" if xlim else f"{int(v):,}", va="center", fontsize=10)
    a.set_yticks(y); a.set_yticklabels(R.label if a is ax[0] else [], fontsize=10)
    a.set_title(title, fontweight="bold", fontsize=10)
    if xlim: a.set_xlim(0, xlim)
fig.suptitle(f"BMimica — Liezl ×6 vs ByteTrack over {n} sessions (2-animal, 5 cams each)\n"
             "Detections = sleap-nn 0.3.0 filter+track pool; GT = reprojected proofread 3D.",
             fontsize=11, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.86])
for e in ("png", "pdf"): fig.savefig(BM/f"figure_bmimica_liezl_vs_bytetrack.{e}", dpi=140, bbox_inches="tight")
print("wrote", BM/"figure_bmimica_liezl_vs_bytetrack.png")
