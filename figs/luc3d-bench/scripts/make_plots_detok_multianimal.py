#!/usr/bin/env python3
"""Fresh, fair-comparison figure: detections-only ID tracking on MULTI-ANIMAL sessions.

Deliberately excludes every legacy/confounded tracker (the top-4-drop runs, the
filter sweeps, etc.). Shows ONLY the apples-to-apples set:

  - SLEAP (single-view ref)          legitimate single-view baseline (old tracked H5)
  - ByteTrack [det-only]             shared filtered detection pool
  - LUC3D [det-only]                 shared filtered detection pool, identities from scratch
  - Liezl [det-only]                 shared filtered detection pool, identities from scratch
  - sleap-nn track -> Liezl          keep-all-tracklets: sleap-nn temporal linking + Liezl

1-animal sessions are trivially solved (single identity) and only dilute the
signal, so we restrict to sessions with >=2 animals.

Usage:
  PYTHONPATH=. .venv/bin/python scripts/make_plots_detok_multianimal.py [out_stem]
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[1]
CSV = REPO / "outputs/PAF_3d_kalman/metrics/per_camera_session_metrics.csv"
PLOTS = REPO / "outputs/PAF_3d_kalman/plots"

# (csv tracker name, display label, color) — order top->bottom in the bars.
# Grouped: single-view ref, then det-only (NO temporal), then TEMPORAL variants.
TRACKERS = [
    ("detok_sleap_ref",         "SLEAP (single-view ref)",       "#999999"),
    # --- detections-only, no temporal model (identities from per-frame geometry) ---
    ("detok_bytetrack",         "ByteTrack [det-only]",          "#d62728"),
    ("detok_luc3d",             "LUC3D [det-only]",              "#1f77b4"),
    ("detok_liezl",             "Liezl [det-only, no temporal]", "#2ca02c"),
    # --- temporal tracking added ---
    ("detok_luc3d_3dkf",        "LUC3D + 3D Kalman (temporal)",  "#17becf"),
    ("detok_liezl_motionfix",   "Liezl + 3D-motion vote",        "#ff7f0e"),
    ("detok_keepall_liezl",     "sleap-nn track → Liezl",       "#9467bd"),
]
MIN_ANIMALS = 2


def ci95(series):
    a = series.dropna().values
    if len(a) < 2:
        return 0.0
    return 1.96 * a.std(ddof=1) / np.sqrt(len(a))


def main(out_stem="figure_detok_multianimal"):
    df = pd.read_csv(CSV)
    df = df[df["animals"] >= MIN_ANIMALS].copy()
    present = set(df["tracker"].unique())
    rows = [t for t in TRACKERS if t[0] in present]
    missing = [t[0] for t in TRACKERS if t[0] not in present]
    if missing:
        print(f"WARNING: not yet in metrics CSV (skipped): {missing}")

    # per-session totals (sum switches over cameras), then aggregate per tracker
    agg = []
    for name, label, color in rows:
        d = df[df["tracker"] == name]
        per_sess_sw = d.groupby("session")["num_switches"].sum()
        agg.append(dict(
            name=name, label=label, color=color,
            n_camsess=len(d),
            n_sess=d["session"].nunique(),
            switches=int(d["num_switches"].sum()),
            sw_per_sess=per_sess_sw.mean(),
            idf1=d["idf1"].mean(), idf1_ci=ci95(d["idf1"]),
            idp=d["idp"].mean(), idr=d["idr"].mean(),
            frag=int(d["num_fragmentations"].sum()),
        ))
    A = pd.DataFrame(agg)
    print("\n=== Multi-animal (>= %d) fair comparison ===" % MIN_ANIMALS)
    print(A[["label", "n_camsess", "n_sess", "switches", "sw_per_sess",
             "idf1", "idp", "idr", "frag"]].round(4).to_string(index=False))

    labels = A["label"].tolist()
    colors = A["color"].tolist()
    y = np.arange(len(A))[::-1]  # first row at top

    fig, ax = plt.subplots(2, 2, figsize=(15, 8))
    n_sess = A["n_sess"].iloc[0] if len(A) else 0
    fig.suptitle(
        f"Detections-only ID tracking — MULTI-ANIMAL sessions only "
        f"(≥{MIN_ANIMALS} animals; {n_sess} sessions)",
        fontsize=14, fontweight="bold")

    # A: total switches
    ax[0, 0].barh(y, A["switches"], color=colors)
    ax[0, 0].set_yticks(y); ax[0, 0].set_yticklabels(labels)
    ax[0, 0].set_title("A. Total ID switches (lower is better)")
    ax[0, 0].set_xlabel("ID switches")
    for yi, v in zip(y, A["switches"]):
        ax[0, 0].text(v, yi, f" {v}", va="center", fontsize=9)

    # B: switches per session
    ax[0, 1].barh(y, A["sw_per_sess"], color=colors)
    ax[0, 1].set_yticks(y); ax[0, 1].set_yticklabels(labels)
    ax[0, 1].set_title("B. ID switches per session (lower is better)")
    ax[0, 1].set_xlabel("switches / session")
    for yi, v in zip(y, A["sw_per_sess"]):
        ax[0, 1].text(v, yi, f" {v:.1f}", va="center", fontsize=9)

    # C: IDF1 with CI
    ax[1, 0].barh(y, A["idf1"], xerr=A["idf1_ci"], color=colors,
                  error_kw=dict(ecolor="#333", capsize=3))
    ax[1, 0].set_yticks(y); ax[1, 0].set_yticklabels(labels)
    ax[1, 0].set_title("C. IDF1 (higher is better, 95% CI)")
    ax[1, 0].set_xlabel("IDF1"); ax[1, 0].set_xlim(0, 1)
    for yi, v in zip(y, A["idf1"]):
        ax[1, 0].text(v, yi, f" {v:.3f}", va="center", fontsize=9)

    # D: Pareto switches/sess vs IDF1
    ax[1, 1].scatter(A["sw_per_sess"], A["idf1"], c=colors, s=120, zorder=3)
    for _, r in A.iterrows():
        ax[1, 1].annotate(r["label"], (r["sw_per_sess"], r["idf1"]),
                          fontsize=8, xytext=(5, 4), textcoords="offset points")
    ax[1, 1].set_title("D. Pareto: switches/session vs IDF1")
    ax[1, 1].set_xlabel("switches / session (lower better)")
    ax[1, 1].set_ylabel("IDF1 (higher better)")
    ax[1, 1].grid(alpha=0.3)

    plt.tight_layout(rect=(0, 0, 1, 0.96))
    PLOTS.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        p = PLOTS / f"{out_stem}.{ext}"
        fig.savefig(p, dpi=140, bbox_inches="tight")
        print("wrote", p)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "figure_detok_multianimal")
