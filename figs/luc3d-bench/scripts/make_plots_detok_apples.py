"""Apples-to-apples detections-only benchmark figure (MULTI-ANIMAL sessions only).

Every tracker here consumes the IDENTICAL filter-only per-frame detection pool
(no tracking, no top-N drop) — so this is a fair tracker comparison. 1-animal
sessions are excluded (identity tracking is trivial there); we keep only
`animals >= 2` cam-sessions.

Data sources (all filtered to animals>=2):
  - per-camera eval CSVs for the LUC3D/Liezl variants (tracker=='luc3d' rows)
  - the registered detok_bytetrack / detok_sleap_ref rows in the metrics CSV

`detok_sleap_ref` is NOT detections-only (it is single-view SLEAP tracking on the
OLD tracked H5). It is shown ONLY as a reference for what per-camera temporal
linking buys — visually set apart (hatched/grey) — because the headline finding
is that detections-only strips that temporal signal.

Output: outputs/PAF_3d_kalman/plots/figure_detections_only_multianimal.{png,pdf}
"""
from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
PAF = REPO / "outputs" / "PAF_3d_kalman"
METRICS = PAF / "metrics" / "per_camera_session_metrics.csv"
PLOTS = PAF / "plots"

# (label, source, color, is_reference, hatch)
# source: ("csv", path)  -> per-camera eval CSV, take tracker=='luc3d' rows
#         ("metrics", tracker_name) -> rows from the registered metrics CSV
VARIANTS = [
    ("ByteTrack [detok]",        ("metrics", "detok_bytetrack"), "#c2185b", False, None),
    ("LUC3D [detok] (bare)",     ("csv", PAF / "_eval_detok_luc3d.csv"), "#1565c0", False, None),
    ("LUC3D [detok] +3D-KF/PAF", ("csv", PAF / "_eval_detok_luc3d_3dkf.csv"), "#6a1b9a", False, None),
    ("Liezl [detok] (bare)",     ("csv", PAF / "_eval_detok_liezl.csv"), "#2e7d32", False, None),
    ("Liezl [detok] +motionfix", ("csv", PAF / "_eval_detok_liezl_motionfix.csv"), "#00897b", False, None),
    ("SLEAP single-view (ref, temporal)", ("metrics", "detok_sleap_ref"), "#9e9e9e", True, "////"),
]

MIN_ANIMALS = 2


def load_rows(source) -> pd.DataFrame:
    kind, ref = source
    if kind == "csv":
        df = pd.read_csv(ref)
        df = df[df.tracker == "luc3d"]
    else:
        df = pd.read_csv(METRICS)
        df = df[df.tracker == ref]
    return df[df.animals >= MIN_ANIMALS].copy()


def main():
    recs = []
    per_session = {}
    for label, source, color, is_ref, hatch in VARIANTS:
        df = load_rows(source)
        if df.empty:
            print(f"WARN: no rows for {label}; skipping")
            continue
        recs.append({
            "label": label, "color": color, "is_ref": is_ref, "hatch": hatch,
            "n": len(df),
            "switches": int(df.num_switches.sum()),
            "sw_per": df.num_switches.sum() / len(df),
            "idf1": df.idf1.mean(),
            "idf1_sem": df.idf1.sem(),
            "frag": int(df.num_fragmentations.sum()),
        })
        per_session[label] = df.num_switches.values
    s = pd.DataFrame(recs)
    print(s[["label", "n", "switches", "sw_per", "idf1", "frag"]].to_string(index=False))

    mpl.rcParams["font.family"] = "DejaVu Sans"
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    y = np.arange(len(s))
    colors = s.color.tolist()

    # ---- Panel A: total switches (primary). 3D-KF is off-scale → clip + annotate.
    axA = axes[0, 0]
    XCAP = 18000
    disp = s.switches.clip(upper=XCAP)
    bars = axA.barh(y, disp, color=colors, edgecolor="black", linewidth=0.5)
    for b, h in zip(bars, s.hatch):
        if isinstance(h, str) and h:
            b.set_hatch(h)
    for i, sw in enumerate(s.switches):
        txt = f" {sw:,}" + (" →(off-scale)" if sw > XCAP else "")
        axA.text(min(sw, XCAP), i, txt, va="center", fontsize=7.5)
    axA.set_yticks(y); axA.set_yticklabels(s.label, fontsize=8); axA.invert_yaxis()
    axA.set_xlabel("Total ID switches (lower is better)")
    axA.set_xlim(0, XCAP * 1.18)
    axA.set_title("A. Total switches — multi-animal sessions only", fontweight="bold")
    axA.grid(axis="x", alpha=0.3)

    # ---- Panel B: IDF1 with 95% CI
    axB = axes[0, 1]
    bars = axB.barh(y, s.idf1, xerr=s.idf1_sem * 1.96, color=colors,
                    edgecolor="black", linewidth=0.5)
    for b, h in zip(bars, s.hatch):
        if isinstance(h, str) and h:
            b.set_hatch(h)
    axB.set_yticks(y); axB.set_yticklabels(s.label, fontsize=8); axB.invert_yaxis()
    axB.set_xlabel("IDF1 (mean ± 95% CI)")
    axB.set_xlim(0, 1.0)
    axB.set_title("B. Identity F1 — multi-animal sessions only", fontweight="bold")
    axB.grid(axis="x", alpha=0.3)

    # ---- Panel C: fragmentations — the temporal-contiguity diagnostic
    axC = axes[1, 0]
    bars = axC.barh(y, s.frag, color=colors, edgecolor="black", linewidth=0.5)
    for b, h in zip(bars, s.hatch):
        if isinstance(h, str) and h:
            b.set_hatch(h)
    for i, fr in enumerate(s.frag):
        axC.text(fr, i, f" {fr:,}", va="center", fontsize=7.5)
    axC.set_yticks(y); axC.set_yticklabels(s.label, fontsize=8); axC.invert_yaxis()
    axC.set_xlabel("Total fragmentations (lower = better temporal contiguity)")
    axC.set_xlim(0, s.frag.max() * 1.22)
    axC.set_title("C. Fragmentation — the temporal-contiguity signal", fontweight="bold")
    axC.grid(axis="x", alpha=0.3)

    # ---- Panel D: Pareto IDF1 vs switches (log-x to fit 3D-KF)
    axD = axes[1, 1]
    for _, r in s.iterrows():
        axD.scatter(r.switches, r.idf1, s=140, color=r.color, edgecolor="black",
                    linewidth=0.6, zorder=3, hatch=(r.hatch if isinstance(r.hatch, str) else ""), label=r.label)
    axD.set_xscale("log")
    axD.set_xlabel("Total switches (log scale, lower better)")
    axD.set_ylabel("Mean IDF1 (higher better)")
    axD.set_title("D. Pareto: IDF1 vs switches", fontweight="bold")
    axD.legend(fontsize=7, loc="lower left")
    axD.grid(alpha=0.3, which="both")

    n_sess = s.n.iloc[0] if len(s) else 0
    fig.suptitle(
        f"Detections-Only Benchmark — Apples-to-Apples, Multi-Animal Only "
        f"({n_sess} cam-sessions, animals≥{MIN_ANIMALS})   "
        f"[{pd.Timestamp.now('UTC'):%Y-%m-%d %H:%M} UTC]",
        fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    PLOTS.mkdir(parents=True, exist_ok=True)
    png = PLOTS / "figure_detections_only_multianimal.png"
    pdf = PLOTS / "figure_detections_only_multianimal.pdf"
    fig.savefig(png, dpi=150, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"\nwrote {png}\nwrote {pdf}")


if __name__ == "__main__":
    main()
