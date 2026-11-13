"""Additional panels for the PAF benchmark — per-session improvement scatter,
PAF intervention statistics, etc. Run AFTER scripts/make_plots_paf.py."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

REPO = Path(__file__).resolve().parents[1]
PAF_DIR = REPO / "outputs" / "PAF_3d_kalman"
METRICS_DIR = PAF_DIR / "metrics"
PLOTS_DIR = PAF_DIR / "plots"
SRC = METRICS_DIR / "per_camera_session_metrics.csv"

PALETTE = {
    "sleap": "#7F7F7F",
    "bytetrack": "#E69F00",
    "luc3d": "#56B4E9",
    "luc3d_paf_l1": "#009E73",
    "luc3d_paf_full": "#0072B2",
}
TRACKER_LABEL = {
    "sleap": "SLEAP",
    "bytetrack": "ByteTrack",
    "luc3d": "LUC3D (baseline)",
    "luc3d_paf_l1": "LUC3D + PAF (L1)",
    "luc3d_paf_full": "LUC3D + PAF (L1+2+3)",
}


def set_style():
    sns.set_theme(context="paper", style="ticks", font="DejaVu Sans")
    mpl.rcParams.update({
        "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
        "savefig.facecolor": "white", "axes.facecolor": "white",
        "axes.edgecolor": "0.15", "axes.linewidth": 0.8,
        "axes.titlesize": 9.0, "axes.labelsize": 7.5,
        "xtick.labelsize": 7, "ytick.labelsize": 7,
        "axes.spines.top": False, "axes.spines.right": False,
        "pdf.fonttype": 42, "ps.fonttype": 42, "font.size": 7.5,
    })


def panel_per_session_idf1_delta(df: pd.DataFrame, out_path: Path):
    """For each session, plot (PAF IDF1 − baseline LUC3D IDF1) sorted by Δ."""
    pivot = df.groupby(["session", "tracker"])["idf1"].mean().unstack("tracker")
    needed = ["luc3d", "luc3d_paf_l1", "luc3d_paf_full"]
    pivot = pivot.dropna(subset=[c for c in needed if c in pivot.columns], how="any")
    if pivot.empty: return

    deltas_full = (pivot["luc3d_paf_full"] - pivot["luc3d"]).sort_values()
    deltas_l1 = (pivot["luc3d_paf_l1"] - pivot["luc3d"]).reindex(deltas_full.index)

    fig, ax = plt.subplots(figsize=(7.5, 3.2))
    x = np.arange(len(deltas_full))
    ax.bar(x, deltas_full.values, color=PALETTE["luc3d_paf_full"], alpha=0.85, width=0.85,
           label=TRACKER_LABEL["luc3d_paf_full"])
    ax.plot(x, deltas_l1.values, marker="o", markersize=2.5, linewidth=0,
            color=PALETTE["luc3d_paf_l1"], alpha=0.8, label=TRACKER_LABEL["luc3d_paf_l1"])
    ax.axhline(0, color="0.4", linewidth=0.8)
    ax.set_xlabel("session (sorted by full-PAF Δ IDF1)")
    ax.set_ylabel("IDF1 − baseline LUC3D")
    ax.set_title("Per-session IDF1 improvement over baseline LUC3D (sorted)")
    ax.legend(loc="upper left", fontsize=7)
    sns.despine(ax=ax, trim=False, offset=2)
    fig.savefig(out_path, bbox_inches="tight"); plt.close(fig)


def panel_per_session_switches_delta(df: pd.DataFrame, out_path: Path):
    pivot = df.groupby(["session", "tracker"])["num_switches"].mean().unstack("tracker")
    needed = ["luc3d", "luc3d_paf_l1", "luc3d_paf_full"]
    pivot = pivot.dropna(subset=[c for c in needed if c in pivot.columns], how="any")
    if pivot.empty: return
    deltas_full = (pivot["luc3d_paf_full"] - pivot["luc3d"]).sort_values()
    deltas_l1 = (pivot["luc3d_paf_l1"] - pivot["luc3d"]).reindex(deltas_full.index)

    fig, ax = plt.subplots(figsize=(7.5, 3.2))
    x = np.arange(len(deltas_full))
    ax.bar(x, deltas_full.values, color=PALETTE["luc3d_paf_full"], alpha=0.85, width=0.85,
           label=TRACKER_LABEL["luc3d_paf_full"])
    ax.plot(x, deltas_l1.values, marker="o", markersize=2.5, linewidth=0,
            color=PALETTE["luc3d_paf_l1"], alpha=0.8, label=TRACKER_LABEL["luc3d_paf_l1"])
    ax.axhline(0, color="0.4", linewidth=0.8)
    ax.set_xlabel("session (sorted by full-PAF Δ switches)")
    ax.set_ylabel("ID switches − baseline LUC3D")
    ax.set_title("Per-session Δ switches vs baseline (negative = improvement)")
    ax.legend(loc="upper left", fontsize=7)
    sns.despine(ax=ax, trim=False, offset=2)
    fig.savefig(out_path, bbox_inches="tight"); plt.close(fig)


def panel_paf_intervention_stats(out_path: Path):
    """Read per-session PAF stats from log files."""
    log_dir = REPO / "logs" / "luc3d_track_all"
    rows = []
    for p in log_dir.glob("*.log"):
        try:
            txt = p.read_text()
            # last "temporal-3D-PAF" line wins
            kf, l2, l3 = 0, 0, 0
            for line in txt.splitlines():
                if "temporal-3D-PAF:" in line and "KF updates" in line:
                    parts = line.split(",")
                    try:
                        kf = int(parts[0].split()[-3])
                        l2 = int(parts[1].split()[-3])
                        l3 = int(parts[2].split()[-3])
                    except: pass
            if kf > 0:
                rows.append({"session": p.stem, "kf_updates": kf, "l2_swaps": l2, "l3_flips": l3})
        except: pass
    df = pd.DataFrame(rows)
    if df.empty: return
    df.to_csv(METRICS_DIR / "paf_intervention_stats.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.2))
    sub = df.sort_values("l2_swaps")
    axes[0].bar(range(len(sub)), sub["l2_swaps"], color=PALETTE["luc3d_paf_full"], alpha=0.85)
    axes[0].set_xlabel("session (sorted)"); axes[0].set_ylabel("L2 hypothesis swaps")
    axes[0].set_title(f"L2 swaps per session (median={int(df['l2_swaps'].median())})")
    sub = df.sort_values("l3_flips")
    axes[1].bar(range(len(sub)), sub["l3_flips"], color=PALETTE["luc3d_paf_l1"], alpha=0.85)
    axes[1].set_xlabel("session (sorted)"); axes[1].set_ylabel("L3 retroactive flips")
    axes[1].set_title(f"L3 flips per session (median={int(df['l3_flips'].median())})")
    for ax in axes:
        sns.despine(ax=ax, trim=False, offset=2)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight"); plt.close(fig)


def main():
    set_style()
    if not SRC.exists():
        print(f"missing {SRC}; run scripts/make_plots_paf.py first")
        return
    df = pd.read_csv(SRC)
    panel_per_session_idf1_delta(df, PLOTS_DIR / "panel_per_session_idf1_delta.png")
    panel_per_session_switches_delta(df, PLOTS_DIR / "panel_per_session_switches_delta.png")
    panel_paf_intervention_stats(PLOTS_DIR / "panel_paf_intervention_stats.png")
    print(f"wrote extra panels to {PLOTS_DIR}")


if __name__ == "__main__":
    main()
