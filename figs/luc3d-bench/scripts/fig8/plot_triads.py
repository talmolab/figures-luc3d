"""Plot the TRIADS identity results: IDF1 (within vs cross view), identity
accuracy, and the purity/coverage curve.

Panel c is the one that carries a negative result and it is the most important of
the three: purity is FLAT in the confidence threshold, so the reprojection residual
of an assignment says essentially nothing about whether its identity is right. That
is because the errors are not per-detection geometric failures -- after an
unrecovered switch every detection is confidently and self-consistently assigned to
the wrong animal, with a perfectly good residual. No per-frame confidence measure
can catch that, which is precisely the argument for a proofreading tool with a
timeline and a 3D view rather than an automatic confidence filter.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUT = Path("/root/vast/eric/luc3d-bench/outputs/fig8")
TEAL, ORANGE, PURPLE = "#66C2A5", "#FC8D62", "#8DA0CB"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "axes.linewidth": 0.8, "xtick.major.width": 0.8, "ytick.major.width": 0.8,
    "lines.linewidth": 2.0, "savefig.transparent": True,
})


def despine(ax, left=True):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if not left:
        ax.spines["left"].set_visible(False)


def main():
    import sys as _sys
    win = "_20k" if "--20k" in _sys.argv else ""
    df = pd.read_csv(OUT / f"triads_idf1{win}.csv")
    cur = pd.read_csv(OUT / f"triads_purity_curve{win}.csv")
    base = pd.read_csv(OUT / "triads_baselines.csv")
    df = df.sort_values("idf1_cross_view").reset_index(drop=True)
    n_an = 3
    chance = 1.0 / n_an

    fig, axes = plt.subplots(1, 3, figsize=(9.2, 2.7))

    # ---- a: within-view vs cross-view IDF1, three trackers ---------------- #
    # Fig 7a's construction, on third-party data: same detections, three trackers,
    # each scored with a per-camera mapping and then with ONE global mapping.
    ax = axes[0]
    series = [
        ("LUC3D", df.idf1_within_view.mean(), df.idf1_cross_view.mean(), TEAL),
        ("centroid (SLEAP-style)",
         base[base.method == "simple"].idf1_within_view.mean(),
         base[base.method == "simple"].idf1_cross_view.mean(), ORANGE),
        ("ByteTrack",
         base[base.method == "bytetrack"].idf1_within_view.mean(),
         base[base.method == "bytetrack"].idf1_cross_view.mean(), PURPLE),
    ]
    # Stagger the labels: the two baselines' cross-view values are close enough
    # that placing both at their own y collides them.
    offsets = [0.0, +0.055, -0.055]
    for (name, w, c, col), dy in zip(series, offsets):
        ax.plot([0, 1], [w, c], color=col, marker="o", markersize=4, lw=2.2)
        ax.text(1.06, c + dy, f"{name}\n{c:.3f}  ({c / w:.2f}x)", fontsize=6.2,
                color=col, va="center", fontweight="bold", linespacing=1.25)
    ax.axhline(1.0 / 6, ls="--", lw=0.8, color="0.5")
    ax.text(0.02, 1.0 / 6 + 0.02, "1/C = 0.17  ceiling for a per-camera tracker",
            fontsize=5.8, color="0.35")
    ax.set_xlim(-0.15, 2.4)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["within-view", "cross-view"])
    ax.set_ylabel("IDF1")
    ax.set_ylim(0, 1)
    ax.set_title("a  Only LUC3D's identity survives pooling", loc="left", fontsize=8)
    despine(ax)

    # ---- b: identity accuracy --------------------------------------------- #
    ax = axes[1]
    y = np.arange(len(df))
    ax.barh(y, df.accuracy_pct, color=TEAL, height=0.62)
    ax.axvline(100 * chance, ls="--", lw=0.8, color=ORANGE)
    ax.text(100 * chance + 1.5, len(df) - 0.4, "chance\n(3 animals)",
            fontsize=6.5, color=ORANGE, va="top")
    ax.set_yticks(y)
    ax.set_yticklabels([s.replace("2023_03_", "") for s in df.session], fontsize=6)
    ax.set_xlabel("identity accuracy (%)")
    ax.set_xlim(0, 100)
    pooled = (df.accuracy_pct * df.detections).sum() / df.detections.sum()
    ax.set_title(f"b  LUC3D: correct animal of 3 — pooled {pooled:.1f}%",
                 loc="left", fontsize=8)
    despine(ax)

    # ---- c: purity vs coverage -------------------------------------------- #
    ax = axes[2]
    for s, g in cur.groupby("session"):
        g = g.sort_values("coverage")
        ax.plot(100 * g.coverage, 100 * g.purity, color=PURPLE, alpha=0.55, lw=1.2)
    m = cur.groupby("threshold_px")[["coverage", "purity"]].mean().sort_values("coverage")
    ax.plot(100 * m.coverage, 100 * m.purity, color=PURPLE, lw=2.4, label="mean")
    ax.axhline(100 * chance, ls="--", lw=0.8, color=ORANGE)
    ax.set_xlabel("coverage: assignments kept (%)")
    ax.set_ylabel("purity: correct identity (%)")
    ax.set_ylim(0, 100)
    ax.set_xlim(0, 100)
    ax.set_title("c  Confidence cannot buy purity", loc="left", fontsize=8)
    lo, hi = 100 * m.purity.min(), 100 * m.purity.max()
    ax.annotate(f"purity is FLAT ({lo:.1f}-{hi:.1f}%) across the whole\n"
                "threshold range: after an un-recovered switch every\n"
                "detection is confidently assigned to the WRONG animal\n"
                "with a good residual, so geometry cannot flag it",
                xy=(0.03, 0.06), xycoords="axes fraction", fontsize=6, color="0.25")
    despine(ax)

    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"triads_identity.{ext}", dpi=300, bbox_inches="tight")
    print("wrote", OUT / "triads_identity.png")


if __name__ == "__main__":
    sys.exit(main())
