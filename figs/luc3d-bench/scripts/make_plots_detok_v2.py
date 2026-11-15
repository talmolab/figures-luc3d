"""Detections-only multi-animal figure v2 — clear labels + honest switch metric.

Fixes from v1 (Eric feedback 2026-06-22):
  - No jargon. "detok" -> "detections-only"; the SLEAP row is spelled out as a
    per-camera 2D tracker REFERENCE (not part of the shared pool).
  - Switches shown as RAW vs SUSTAINED. Raw motmetrics switches count every identity
    transition, but ~half-to-two-thirds are brief <30-frame flickers that revert.
    SUSTAINED = transitions between identity runs >=30 frames (a swap-once-and-stay
    counts once; oscillation flicker is removed). This is the honest tracker-error
    count and it reverses the motionfix "win": motionfix beats bare Liezl on raw
    switches but is WORSE on sustained ones.
  - MULTI-ANIMAL sessions only (animals>=2). 1-animal sessions are trivially solved
    (IDF1 0.91, 0 switches) and only inflate the averages.

Every tracker here consumes the IDENTICAL filtered per-frame detection pool (no
tracking, no top-N drop) EXCEPT the SLEAP reference, which uses its own per-camera
2D tracking on the original tracked H5 (shown hatched as context).

Inputs:
  outputs/PAF_3d_kalman/_liezlsweep/switch_breakdown.tsv   (tracker, raw, sustained)
  per-camera eval CSVs / metrics CSV (IDF1, animals>=2)
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
BREAKDOWN = PAF / "_liezlsweep" / "switch_breakdown.tsv"
PLOTS = PAF / "plots"

# row key in breakdown.tsv -> (label, idf1 source, color, is_reference)
# idf1 source: ("csv", path) reads tracker=='luc3d'; ("metrics", name) reads metrics CSV
ROWS = [
    ("detok_sleap_ref", "SLEAP per-camera 2D tracker\n(reference — not shared pool)",
     ("metrics", "detok_sleap_ref"), "#9e9e9e", True),
    ("detok_bytetrack", "ByteTrack (2D + cross-view)",
     ("metrics", "detok_bytetrack"), "#c2185b", False),
    ("detok_luc3d_results", "LUC3D (multi-view 3D)",
     ("csv", PAF / "_eval_detok_luc3d.csv"), "#1565c0", False),
    ("detok_liezl_results", "Liezl CrossView",
     ("csv", PAF / "_eval_detok_liezl.csv"), "#2e7d32", False),
    ("detok_liezl_motionfix_results", "Liezl + motion-vote",
     ("csv", PAF / "_eval_detok_liezl_motionfix.csv"), "#00897b", False),
]
MIN_ANIMALS = 2


def idf1_for(source):
    kind, ref = source
    if kind == "csv":
        df = pd.read_csv(ref); df = df[df.tracker == "luc3d"]
    else:
        df = pd.read_csv(METRICS); df = df[df.tracker == ref]
    df = df[df.animals >= MIN_ANIMALS]
    a = df.idf1.dropna().values
    return df.idf1.mean(), (1.96 * a.std(ddof=1) / np.sqrt(len(a)) if len(a) > 1 else 0.0)


def main():
    bd = pd.read_csv(BREAKDOWN, sep="\t", header=None, names=["key", "raw", "sustained"])
    bd = bd.drop_duplicates("key", keep="last").set_index("key")

    recs = []
    for key, label, src, color, is_ref in ROWS:
        if key not in bd.index:
            print(f"WARN missing breakdown for {key}"); continue
        idf1, ci = idf1_for(src)
        recs.append(dict(label=label, color=color, is_ref=is_ref,
                         raw=int(bd.loc[key, "raw"]), sustained=int(bd.loc[key, "sustained"]),
                         idf1=idf1, idf1_ci=ci))
    s = pd.DataFrame(recs)
    print(s[["label", "raw", "sustained", "idf1"]].to_string(index=False))

    mpl.rcParams["font.family"] = "DejaVu Sans"
    fig, (axS, axF) = plt.subplots(1, 2, figsize=(16, 6.5))
    y = np.arange(len(s))[::-1]
    h = 0.38

    # ---- Panel A: RAW vs SUSTAINED switches (grouped) ----
    axS.barh(y + h/2, s["raw"], height=h, color=s.color, alpha=0.35,
             edgecolor="black", linewidth=0.5, label="raw (all transitions)")
    bars = axS.barh(y - h/2, s["sustained"], height=h, color=s.color,
                    edgecolor="black", linewidth=0.5, label="sustained (>=30-frame runs)")
    for is_ref, yy, sv, rv in zip(s.is_ref, y, s["sustained"], s["raw"]):
        axS.text(rv, yy + h/2, f" {rv:,}", va="center", fontsize=7.5, alpha=0.7)
        axS.text(sv, yy - h/2, f" {sv:,}", va="center", fontsize=8, fontweight="bold")
    axS.set_yticks(y); axS.set_yticklabels(s.label, fontsize=8.5)
    axS.set_xlabel("ID switches (lower is better)")
    axS.set_title("A. Identity switches — raw vs SUSTAINED\n"
                  "(sustained removes <30-frame flicker = the honest error count)",
                  fontweight="bold", fontsize=10)
    axS.legend(loc="lower right", fontsize=8)
    axS.grid(axis="x", alpha=0.3)
    axS.set_xlim(0, s["raw"].max() * 1.12)

    # ---- Panel B: IDF1 ----
    axF.barh(y, s["idf1"], height=0.6, xerr=s["idf1_ci"], color=s.color,
             edgecolor="black", linewidth=0.5, error_kw=dict(ecolor="#333", capsize=3))
    for yy, v in zip(y, s["idf1"]):
        axF.text(v + 0.01, yy, f"{v:.3f}", va="center", fontsize=8.5, fontweight="bold")
    axF.set_yticks(y); axF.set_yticklabels(s.label, fontsize=8.5)
    axF.set_xlim(0, 1.0)
    axF.set_xlabel("IDF1 — identity accuracy (higher is better)")
    axF.set_title("B. Identity F1 (handles swap-once-and-stay correctly)",
                  fontweight="bold", fontsize=10)
    axF.grid(axis="x", alpha=0.3)

    fig.suptitle(
        "Detections-only ID tracking — multi-animal sessions (≥2 animals; 252 cam-sessions)\n"
        "Every tracker gets the IDENTICAL filtered per-frame detection pool (no tracking, no data dropped); "
        "SLEAP row = per-camera 2D reference",
        fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.93])

    PLOTS.mkdir(parents=True, exist_ok=True)
    png = PLOTS / "figure_detections_only_multianimal.png"
    pdf = PLOTS / "figure_detections_only_multianimal.pdf"
    fig.savefig(png, dpi=150, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"\nwrote {png}\nwrote {pdf}")


if __name__ == "__main__":
    main()
