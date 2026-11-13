#!/usr/bin/env python3
"""Figure: WHERE/HOW do Liezl ID switches happen? (champion = keeptrack + corr3d6)

Answers Eric's two questions, against the 3D ground truth (points3d.h5):
  (1) switch counts are TRANSITIONS, not per-frame inflation.
  (2) switches happen at CLOSE encounters / small apparent size, NOT far from camera.

Inputs: outputs/PAF_3d_kalman/_diag/G_keeptrack_3d6_{sw_cov,all_cov}.npy
        + ..._switch_events.tsv (session,difficulty,cam,frame,gt_oid,depth,inter,area)
Output: outputs/PAF_3d_kalman/plots/figure_switch_diagnosis.{png,pdf}
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[1]
PRE = REPO / "outputs/PAF_3d_kalman/_diag/G_keeptrack_3d6"
sw = np.load(f"{PRE}_sw_cov.npy")          # (n_sw, 3) depth, inter, area
al = np.load(f"{PRE}_all_cov.npy")         # (n_sample, 3) baseline
ev = pd.read_csv(f"{PRE}_switch_events.tsv", sep="\t")

GREY = "#8a8a8a"; RED = "#d1495b"
fig, ax = plt.subplots(2, 3, figsize=(18, 9.5))
fig.suptitle("WHERE & HOW do the ID switches happen?  champion = Liezl keeptrack + 3D-weight×6  "
             "(42 multi-animal sessions; covariates from 3D ground truth)\n"
             "3,594 raw / 1,005 sustained switches over 7.43M matched frames  —  "
             "switches cluster at CLOSE encounters & small apparent size, NOT far from camera",
             fontweight="bold", fontsize=12.5)


def kde_pair(a, col, title, xlabel, xmax, note=None, note_xy=None):
    sw_v = sw[:, col]; al_v = al[:, col]
    sw_v = sw_v[np.isfinite(sw_v)]; al_v = al_v[np.isfinite(al_v)]
    bins = np.linspace(0, xmax, 60)
    a.hist(al_v, bins=bins, density=True, alpha=0.55, color=GREY, label=f"all matched frames")
    a.hist(sw_v, bins=bins, density=True, alpha=0.6, color=RED, label=f"switch frames")
    a.axvline(np.median(al_v), color="#444", ls="--", lw=1.6)
    a.axvline(np.median(sw_v), color=RED, ls="--", lw=1.6)
    a.set_title(title, fontweight="bold", fontsize=10.5); a.set_xlabel(xlabel); a.set_ylabel("density")
    a.legend(fontsize=8)
    if note:
        a.annotate(note, xy=note_xy[0], xytext=note_xy[1],
                   arrowprops=dict(arrowstyle="->", color=note_xy[2]), color=note_xy[2], fontsize=9)
    return sw_v, al_v

# A inter-animal distance — the key result
v, b = kde_pair(ax[0, 0], 1, "A. Inter-animal 3D distance — switch med=120mm vs baseline 165mm",
                "min pairwise inter-animal distance (mm)", 500)
ax[0, 0].annotate("switches shift LEFT\n= animals CLOSER", xy=(120, ax[0, 0].get_ylim()[1]*0.55),
                  xytext=(240, ax[0, 0].get_ylim()[1]*0.8),
                  arrowprops=dict(arrowstyle="->", color=RED), color=RED, fontsize=9.5)

# B depth — no far-away effect
kde_pair(ax[0, 1], 0, "B. Distance from camera (depth) — switch med=659mm vs baseline 701mm",
         "camera-frame depth Z (mm)", 1000)
ax[0, 1].annotate("nearly identical →\nNOT 'far from camera'", xy=(659, ax[0, 1].get_ylim()[1]*0.5),
                  xytext=(90, ax[0, 1].get_ylim()[1]*0.85),
                  arrowprops=dict(arrowstyle="->", color="#333"), fontsize=9.5)

# C apparent size
kde_pair(ax[0, 2], 2, "C. Apparent size in frame — switch med=15.5k vs baseline 23k px²",
         "GT bbox area (px²)", 70000)
ax[0, 2].annotate("smaller at switches\n→ LOOKS far, but depth is\n   normal → occlusion / partial det.",
                  xy=(15500, ax[0, 2].get_ylim()[1]*0.5), xytext=(30000, ax[0, 2].get_ylim()[1]*0.8),
                  arrowprops=dict(arrowstyle="->", color=RED), color=RED, fontsize=9)

# D cumulative concentration of close-pass switches vs baseline
axc = ax[1, 0]
sw_i = sw[:, 1]; sw_i = np.sort(sw_i[np.isfinite(sw_i)])
al_i = al[:, 1]; al_i = np.sort(al_i[np.isfinite(al_i)])
axc.plot(sw_i, np.linspace(0, 1, len(sw_i)), color=RED, lw=2, label="switch frames")
axc.plot(al_i, np.linspace(0, 1, len(al_i)), color=GREY, lw=2, label="all matched frames")
axc.axvline(150, color="#444", ls=":", lw=1)
f150 = (sw_i < 150).mean(); fb150 = (al_i < 150).mean()
axc.set_title("D. Cumulative — fraction of frames within X of the other animal",
              fontweight="bold", fontsize=10.5)
axc.set_xlabel("inter-animal distance (mm)"); axc.set_ylabel("cumulative fraction")
axc.set_xlim(0, 500); axc.legend(fontsize=8)
axc.annotate(f"{f150*100:.0f}% of switches happen <150mm apart\n(vs {fb150*100:.0f}% of all frames)",
             xy=(150, f150), xytext=(180, 0.35),
             arrowprops=dict(arrowstyle="->", color=RED), color=RED, fontsize=9.5)

# E regime split by difficulty (both-visible close pass vs single-visible re-entry)
axe = ax[1, 1]
ev["single"] = ev["inter_mm"].isna()
g = ev.groupby("difficulty").agg(n=("frame", "size"), single=("single", "sum"))
g["both"] = g["n"] - g["single"]
xk = g.index.tolist(); x = np.arange(len(xk))
axe.bar(x, g["both"], color="#3a7", label="both visible (close pass)")
axe.bar(x, g["single"], bottom=g["both"], color="#d29", label="single visible (re-entry/recovery)")
for xi, (b_, s_) in enumerate(zip(g["both"], g["single"])):
    axe.text(xi, b_ + s_, f"{b_+s_}", ha="center", va="bottom", fontsize=8)
axe.set_xticks(x); axe.set_xticklabels([f"diff {k}" for k in xk], fontsize=8)
axe.set_title("E. Switch count by difficulty & regime", fontweight="bold", fontsize=10.5)
axe.set_ylabel("switch events"); axe.legend(fontsize=8)
axe.annotate("diff-7 sessions dominate;\nclose passes are the majority everywhere",
             xy=(len(xk)-1, g["n"].iloc[-1]*0.6), xytext=(0.2, g["n"].max()*0.7), fontsize=9)

# F counting answer text
axd = ax[1, 2]; axd.axis("off")
n_single = int(ev["single"].sum())
txt = (
    "F. Are these real switches?\n"
    "──────────────────────────────\n"
    "A motmetrics SWITCH = one identity TRANSITION,\n"
    "not one per mislabeled frame.\n\n"
    "• 3,594 raw switches over 7,430,605 matched frames\n"
    "  (per-frame counting would give ~7.4M).\n"
    "• median identity run = 62 frames; mean = 1,787.\n"
    "• your 1→2→1 over 300 frames is counted as 2, not 300.\n\n"
    "BUT 72% of raw switches are <30-frame FLICKER:\n"
    "  raw 3,594  →  SUSTAINED 1,005  (the honest count).\n\n"
    "Regime split of the 3,594:\n"
    f"  • {3594-n_single} both-visible CLOSE passes (crossing)\n"
    f"  • {n_single} ({100*n_single/3594:.0f}%) re-entry (other animal not\n"
    "    even detected in that camera)."
)
axd.text(0.0, 1.0, txt, va="top", ha="left", fontsize=10.5, family="monospace")

plt.tight_layout(rect=[0, 0, 1, 0.93])
out = REPO / "outputs/PAF_3d_kalman/plots/figure_switch_diagnosis.png"
out.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(out, dpi=120); plt.savefig(str(out).replace(".png", ".pdf"))
print("wrote", out)
