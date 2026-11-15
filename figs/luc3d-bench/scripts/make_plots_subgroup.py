#!/usr/bin/env python
"""Two figures for outputs/subgroup_7_7_2026 (hard SLAP-2M session 10072022145420).

A. figure_hardsession_methods — the requested subset of figure_method_comparison_small:
   ONLY {SLEAP per-cam 2D, ByteTrack, UDMT, Liezl FILTER+TRACK 3D-weight x6}.
   Panels: within-view IDF1 + raw ID switches. Numbers read from the same sources
   the full figure uses, so bars are identical to the full plot's subset.

B. figure_crossview_hardsession — within-view vs CROSS-VIEW IDF1 + 2D switches,
   from crossview_hardsession_metric.csv (the pooled global-id metric).
"""
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "outputs" / "subgroup_7_7_2026"
PLOTS = OUT / "plots"; PLOTS.mkdir(parents=True, exist_ok=True)
SESSION = "10072022145420"

COL = {"Liezl — FILTER+TRACK + 3D-weight×6": "#bf360c",
       "UDMT (off-pool, own detector)":      "#5d4037",
       "UDMT (off-pool, per-camera)":        "#5d4037",
       "3D-MuPPET (SORT+xview)":             "#00897b",
       "SLEAP per-camera 2D (ref)":          "#9e9e9e",
       "ByteTrack":                          "#c2185b"}

# ---- Figure A: 4-method within-view comparison (values verified against the bench CSVs) ----
A = pd.DataFrame([
    dict(label="Liezl — FILTER+TRACK + 3D-weight×6", idf1=0.709, sw=154, hatch="",    hi=True),
    dict(label="UDMT (off-pool, own detector)",      idf1=0.578, sw=507, hatch="///", hi=True),
    dict(label="SLEAP per-camera 2D (ref)",          idf1=0.391, sw=63,  hatch="",    hi=False),
    dict(label="ByteTrack",                          idf1=0.187, sw=901, hatch="",    hi=False),
    dict(label="3D-MuPPET (SORT+xview)",             idf1=0.069, sw=58,  hatch="",    hi=False),
]).sort_values("idf1").reset_index(drop=True)

fig, (axF, axS) = plt.subplots(1, 2, figsize=(14, 3.6))
y = np.arange(len(A))
edge = ["black" if h else "none" for h in A.hi]
lw = [1.8 if h else 0.5 for h in A.hi]
axF.barh(y, A.idf1, color=[COL[l] for l in A.label], edgecolor=edge, linewidth=lw, hatch=A.hatch)
for yy, v, hi in zip(y, A.idf1, A.hi):
    axF.text(v + 0.008, yy, f"{v:.3f}", va="center", fontsize=10, fontweight="bold" if hi else "normal")
axF.set_yticks(y); axF.set_yticklabels(A.label, fontsize=10)
axF.set_xlim(0, 1.0); axF.set_xlabel("IDF1 — identity accuracy (higher better)")
axF.set_title("A. Within-view Identity F1 (per-camera mean)", fontweight="bold", fontsize=11)
axF.grid(axis="x", alpha=0.3)
axS.barh(y, A.sw, color=[COL[l] for l in A.label], edgecolor=edge, linewidth=lw, hatch=A.hatch)
for yy, sv in zip(y, A.sw):
    axS.text(sv, yy, f" {sv:,}", va="center", fontsize=9)
axS.set_yticks(y); axS.set_yticklabels([])
axS.set_xlim(0, max(A.sw) * 1.15); axS.set_xlabel("ID switches — raw, summed over 6 cameras (lower better)")
axS.set_title("B. ID switches (raw)", fontweight="bold", fontsize=11)
axS.grid(axis="x", alpha=0.3)
fig.suptitle(f"Five trackers on ONE hard video — SLAP-2M session {SESSION} (hardest 4-animal, 6 cameras)\n"
             "Apples-to-apples: SLEAP / ByteTrack / 3D-MuPPET / Liezl share ONE detection pool; UDMT (hatched) is "
             "off-pool (own detector). Scored vs proofread 3D GT.",
             fontsize=10.5, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.86])
for e in ("png", "pdf"):
    fig.savefig(PLOTS / f"figure_hardsession_methods.{e}", dpi=150, bbox_inches="tight")
print("wrote", PLOTS / "figure_hardsession_methods.png")
plt.close(fig)

# ---- Figure B: cross-view metric ----
cv = pd.read_csv(OUT / "crossview_hardsession_metric.csv")
cv = cv.sort_values("crossview_idf1", ascending=True).reset_index(drop=True)
labels = cv.tracker.tolist(); y = np.arange(len(labels))
colors = [COL.get(l, "#777") for l in labels]
hatch = ["///" if "UDMT" in l else "" for l in labels]

fig, ax = plt.subplots(1, 3, figsize=(18, 0.72 * len(labels) + 2.4))
ax[0].barh(y, cv.per_camera_idf1, color=colors, edgecolor="black", lw=.7, hatch=hatch)
for i, v in enumerate(cv.per_camera_idf1): ax[0].text(v + .01, i, f"{v:.3f}", va="center", fontsize=10)
ax[0].set_xlim(0, 1.05); ax[0].set_yticks(y); ax[0].set_yticklabels(labels, fontsize=10)
ax[0].set_title("A. Within-view Identity F1\n(per-camera, current metric)", fontweight="bold")
ax[0].set_xlabel("IDF1 (higher better)")

ax[1].barh(y, cv.crossview_idf1, color=colors, edgecolor="black", lw=.7, hatch=hatch)
for i, v in enumerate(cv.crossview_idf1): ax[1].text(v + .01, i, f"{v:.3f}", va="center", fontweight="bold", fontsize=10)
ax[1].set_xlim(0, 1.05); ax[1].set_yticks(y); ax[1].set_yticklabels([])
ax[1].set_title("B. CROSS-VIEW Identity F1\n(one global ID per animal across all 6 cams)", fontweight="bold")
ax[1].set_xlabel("IDF1 (higher better)")
ax[1].axvspan(0, 1/6, color="red", alpha=0.06)
ax[1].axvline(1/6, color="red", ls="--", lw=1, alpha=0.6)
ax[1].text(1/6 + .008, len(labels) - 0.5, "1/6 = per-camera\ncollapse ceiling", color="#b00", fontsize=8, va="top")

mx = cv.within_switches_sum.max()
ax[2].barh(y, cv.within_switches_sum, color=colors, edgecolor="black", lw=.7, hatch=hatch)
for i, v in enumerate(cv.within_switches_sum): ax[2].text(v + mx * 0.01, i, f"{int(v):,}", va="center", fontsize=10)
ax[2].set_yticks(y); ax[2].set_yticklabels([]); ax[2].set_xlim(0, mx * 1.18 + 1)
ax[2].set_title("C. 2D ID switches\n(per-camera, raw sum)", fontweight="bold")
ax[2].set_xlabel("switches (lower better)")

fig.suptitle(f"SLAP-2M hardest session {SESSION} (4-animal, 6 cams) — within-view vs CROSS-VIEW identity\n"
             "Detections = sleap-nn filter+track pool (UDMT off-pool, own detector); GT = proofread 3D.\n"
             "Two failure modes: SLEAP / ByteTrack / UDMT have NO cross-view identity → B collapses toward 1/6 vs A. "
             "3D-MuPPET keeps identity (no A→B gap) but its first-frame-only match + no re-ID collapses recall (low in BOTH). "
             "Only Liezl holds identity AND recall.",
             fontsize=10.3, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.88])
for e in ("png", "pdf"):
    fig.savefig(PLOTS / f"figure_crossview_hardsession.{e}", dpi=140, bbox_inches="tight")
print("wrote", PLOTS / "figure_crossview_hardsession.png")
