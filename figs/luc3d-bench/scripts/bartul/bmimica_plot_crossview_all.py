#!/usr/bin/env python
"""Aggregate cross-view figure over ALL runnable Bartul/BMimica sessions (same rig/
dataset family — Bartul's single session is one of these). Replaces the old
single-session figure_crossview_bartul.png (which only covered session
20250827_141755) and drops the retired trackers: the old broken flat-JS "LUC3D (3D)"
snapshot and the standalone python "Liezl" reimplementation — both superseded by the
real talmolab/luc3d tracker (LUC3D), which now implements the same cross-view model."""
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

BM = Path("/root/vast/eric/luc3d-bench/outputs/bmimica")
PLOTS = BM / "plots"; PLOTS.mkdir(exist_ok=True)
SUBGROUP = Path("/root/vast/eric/luc3d-bench/outputs/subgroup_7_7_2026/plots")

df = pd.read_csv(BM / "bmimica_crossview_all_eval.csv")
n = len(df)

COL = {
    "LUC3D": "#d24a00",
    "3D-MuPPET (SORT+xview)": "#00897b",
    "SLEAP per-camera 2D (ref)": "#8a8a8a",
    "ByteTrack": "#c2185b",
}
rows = []
for tr, lab in [("luc3d", "LUC3D"), ("muppet", "3D-MuPPET (SORT+xview)"),
                ("sleap", "SLEAP per-camera 2D (ref)"), ("byte", "ByteTrack")]:
    if f"{tr}_idf1" not in df:
        continue
    rows.append(dict(label=lab, color=COL[lab],
                      within=df[f"{tr}_idf1"].mean(), xview=df[f"{tr}_xview_idf1"].mean(),
                      sw2d=df[f"{tr}_sw2d"].sum()))
R = pd.DataFrame(rows).sort_values("xview", ascending=True)   # best cross-view on top (barh)
labels = R.label.tolist(); y = np.arange(len(labels))
colors = R.color.tolist()

fig, ax = plt.subplots(1, 3, figsize=(18, 0.62 * len(labels) + 2.4))
ax[0].barh(y, R.within, color=colors, edgecolor="black", lw=.7)
for i, v in enumerate(R.within): ax[0].text(v + .01, i, f"{v:.3f}", va="center", fontsize=10)
ax[0].set_xlim(0, 1.05); ax[0].set_yticks(y); ax[0].set_yticklabels(labels, fontsize=10)
ax[0].set_title("A. Within-view Identity F1\n(per-camera, mean over sessions)", fontweight="bold")
ax[0].set_xlabel("IDF1 (higher better)")

ax[1].barh(y, R.xview, color=colors, edgecolor="black", lw=.7)
for i, v in enumerate(R.xview): ax[1].text(v + .01, i, f"{v:.3f}", va="center", fontweight="bold", fontsize=10)
ax[1].set_xlim(0, 1.05); ax[1].set_yticks(y); ax[1].set_yticklabels([])
ax[1].set_title("B. CROSS-VIEW Identity F1\n(one global ID per animal across all 5 cams)", fontweight="bold")
ax[1].set_xlabel("IDF1 (higher better)")
ax[1].axvspan(0, 0.35, color="red", alpha=0.05)

mx = R.sw2d.max()
ax[2].barh(y, R.sw2d, color=colors, edgecolor="black", lw=.7)
for i, v in enumerate(R.sw2d): ax[2].text(v + mx * 0.01, i, f"{int(v):,}", va="center", fontsize=10)
ax[2].set_yticks(y); ax[2].set_yticklabels([]); ax[2].set_xlim(0, mx * 1.18 + 1)
ax[2].set_title("C. 2D ID switches\n(per-camera, raw sum over all sessions)", fontweight="bold")
ax[2].set_xlabel("switches (lower better)")

fig.suptitle(
    f"Bartul/BMimica corpus — {n} sessions (2-animal, 5 cams each) — within-view vs CROSS-VIEW identity\n"
    "Detections = sleap-nn 0.3.0 filter+track pool; GT = reprojected proofread 3D; full-length sessions.\n"
    "Panel B is the cross-view metric: SLEAP/ByteTrack are per-camera (no global identity) -> collapse toward ~0.2 (1/5 cams); "
    "LUC3D (real talmolab/luc3d tracker) holds one identity across all cameras.",
    fontsize=10.5, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.88])
for e in ("png", "pdf"):
    fig.savefig(PLOTS / f"figure_crossview_bartul_all.{e}", dpi=140, bbox_inches="tight")
    fig.savefig(SUBGROUP / f"figure_crossview_bartul.{e}", dpi=140, bbox_inches="tight")
print("wrote", PLOTS / "figure_crossview_bartul_all.png")
print("wrote", SUBGROUP / "figure_crossview_bartul.png")
print(R[["label", "within", "xview", "sw2d"]].to_string(index=False))
