"""Zero-swaps strategy comparison figure.

Reads outputs/PAF_3d_kalman/metrics/per_camera_session_metrics.csv (which
should include the new rows: sleap_nn_filtered, luc3d_paf_l1_dedupe,
luc3d_paf_l1_dedupe_tracklet3d, luc3d_paf_l1_dedupe_tracklet3d_stitched).

Renders a 4-panel figure:
  Panel A: per-tracker mean IDF1 with 95% CI
  Panel B: per-tracker total switches (across all cam-sessions)
  Panel C: per-tracker switches-per-session histogram
  Panel D: pareto IDF1 vs switches, all trackers labelled

Output: outputs/PAF_3d_kalman/plots/figure_main_zero_swaps.{png,pdf}
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
PAF_DIR = REPO / "outputs" / "PAF_3d_kalman"
METRICS = PAF_DIR / "metrics" / "per_camera_session_metrics.csv"
PLOTS = PAF_DIR / "plots"

# In display order; only trackers actually present get drawn.
# Eric's preference (2026-05-20): NO PAF L1 rows on the final figure.
TRACKER_ORDER = [
    # Single-view baselines
    "sleap",
    "bytetrack",
    # Multi-view baselines (no PAF)
    "luc3d",
    "liezl_cross_view",
    # New pipeline (sleap-nn filter + downstream, no PAF)
    "sleap_nn_filtered_no_paf",
    "liezl_on_sleap_nn",
    "liezl_on_sleap_nn_tracklet",
    # Best-filter sweep winners (added by post-sweep automation if available)
    "sleap_nn_filtered_best_no_paf",
    "liezl_on_sleap_nn_best",
    "liezl_on_sleap_nn_best_tracklet",
    # Detections-only benchmark (identical per-frame detection pool, no data dropped)
    "detok_sleap_ref",
    "detok_bytetrack",
    "detok_luc3d",
    "detok_liezl",
]
LABEL = {
    "sleap": "SLEAP",
    "bytetrack": "ByteTrack",
    "luc3d": "LUC3D (baseline)",
    "liezl_cross_view": "Liezl CrossView (raw)",
    "sleap_nn_filtered_no_paf": "sleap-nn → LUC3D",
    "liezl_on_sleap_nn": "sleap-nn → Liezl",
    "liezl_on_sleap_nn_tracklet": "sleap-nn → Liezl + tracklet_vote",
    "sleap_nn_filtered_best_no_paf": "sleap-nn (tuned filter) → LUC3D",
    "liezl_on_sleap_nn_best": "sleap-nn (tuned filter) → Liezl",
    "liezl_on_sleap_nn_best_tracklet": "sleap-nn (tuned) → Liezl + tracklet_vote",
    "detok_sleap_ref": "SLEAP (single-view ref)",
    "detok_bytetrack": "ByteTrack [detections-only]",
    "detok_luc3d": "LUC3D [detections-only]",
    "detok_liezl": "Liezl [detections-only]",
}
COLOR = {
    "sleap": "#888888",
    "bytetrack": "#aa6666",
    "luc3d": "#3366cc",
    "liezl_cross_view": "#9c27b0",
    "sleap_nn_filtered_no_paf": "#ef6c00",
    "liezl_on_sleap_nn": "#2e7d32",
    "liezl_on_sleap_nn_tracklet": "#1b5e20",
    "sleap_nn_filtered_best_no_paf": "#bf360c",
    "liezl_on_sleap_nn_best": "#388e3c",
    "liezl_on_sleap_nn_best_tracklet": "#004d40",
    "detok_sleap_ref": "#888888",
    "detok_bytetrack": "#c2185b",
    "detok_luc3d": "#1565c0",
    "detok_liezl": "#2e7d32",
}


def load() -> pd.DataFrame:
    if not METRICS.exists():
        raise SystemExit(f"missing {METRICS}")
    df = pd.read_csv(METRICS)
    present = [t for t in TRACKER_ORDER if t in df.tracker.unique()]
    return df[df.tracker.isin(present)], present


def summary(df, present):
    rows = []
    for t in present:
        s = df[df.tracker == t]
        rows.append({
            "tracker": t,
            "label": LABEL[t],
            "n": len(s),
            "switches_total": int(s["num_switches"].sum()),
            "switches_mean": s["num_switches"].mean(),
            "switches_sem": s["num_switches"].sem(),
            "idf1_mean": s["idf1"].mean(),
            "idf1_sem": s["idf1"].sem(),
            "idp_mean": s["idp"].mean(),
            "frag_total": int(s["num_fragmentations"].sum()),
        })
    return pd.DataFrame(rows)


def panel_idf1(ax, summary_df):
    y = np.arange(len(summary_df))
    ax.barh(y, summary_df["idf1_mean"], xerr=summary_df["idf1_sem"] * 1.96,
            color=[COLOR[t] for t in summary_df["tracker"]],
            edgecolor="black", linewidth=0.5)
    ax.set_yticks(y)
    ax.set_yticklabels(summary_df["label"], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("IDF1 (mean ± 95% CI)")
    ax.set_title("A. Identity F1 by tracker", fontweight="bold")
    ax.axvline(0.7360, color="grey", ls="--", lw=0.8, label="LUC3D baseline (no PAF, 3710 sw)")
    ax.axvline(0.803, color="purple", ls=":", lw=0.8, label="Liezl raw (360 sess, 1384 sw)")
    ax.legend(loc="lower right", fontsize=7)
    ax.set_xlim(0.5, 1.0)
    ax.grid(axis="x", alpha=0.3)


def panel_switches(ax, summary_df):
    y = np.arange(len(summary_df))
    totals = summary_df["switches_total"]
    bars = ax.barh(y, totals, color=[COLOR[t] for t in summary_df["tracker"]],
                   edgecolor="black", linewidth=0.5)
    for i, (b, n) in enumerate(zip(bars, totals)):
        ax.text(n, i, f" {int(n):,}", va="center", fontsize=7)
    ax.set_yticks(y)
    ax.set_yticklabels(summary_df["label"], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Total ID switches (lower is better)")
    ax.set_title("B. Total switches by tracker", fontweight="bold")
    ax.axvline(3710, color="grey", ls="--", lw=0.8, label="LUC3D baseline (no PAF, 3710)")
    ax.axvline(1384, color="purple", ls=":", lw=0.8, label="Liezl raw (1384 on 360 subset)")
    ax.legend(loc="lower right", fontsize=7)
    ax.grid(axis="x", alpha=0.3)


def panel_pareto(ax, df, present):
    for t in present:
        s = df[df.tracker == t]
        ax.scatter(s["num_switches"].sum(), s["idf1"].mean(),
                   s=120, color=COLOR[t], edgecolor="black",
                   linewidth=0.6, zorder=3, label=LABEL[t])
    ax.set_xlabel("Total switches (lower better)")
    ax.set_ylabel("Mean IDF1 (higher better)")
    ax.set_title("D. Pareto: IDF1 vs switches", fontweight="bold")
    ax.legend(fontsize=7, loc="lower left")
    ax.grid(alpha=0.3)


def panel_switches_hist(ax, df, present):
    bins = np.linspace(0, 60, 30)
    for t in present:
        if t in ("bytetrack",):  # off-scale
            continue
        s = df[df.tracker == t]
        ax.hist(s["num_switches"], bins=bins, alpha=0.55,
                label=LABEL[t], color=COLOR[t], histtype="step", linewidth=1.5)
    ax.set_xlabel("Switches per cam-session")
    ax.set_ylabel("Number of cam-sessions")
    ax.set_title("C. Per cam-session switch distribution", fontweight="bold")
    ax.set_xlim(0, 30)
    ax.legend(fontsize=7, loc="upper right")
    ax.grid(alpha=0.3)


def main(out_stem: Optional[str] = None):
    df, present = load()
    s = summary(df, present)
    print(s[["label", "n", "switches_total", "idf1_mean", "idp_mean"]].to_string(index=False))

    mpl.rcParams["font.family"] = "DejaVu Sans"
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    panel_idf1(axes[0, 0], s)
    panel_switches(axes[0, 1], s)
    panel_switches_hist(axes[1, 0], df, present)
    panel_pareto(axes[1, 1], df, present)
    fig.suptitle(
        f"Zero-Swaps Strategy Benchmark ({pd.Timestamp.now('UTC'):%Y-%m-%d %H:%M} UTC)",
        fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    PLOTS.mkdir(parents=True, exist_ok=True)
    stem = out_stem or "figure_main_zero_swaps"
    png = PLOTS / f"{stem}.png"
    pdf = PLOTS / f"{stem}.pdf"
    fig.savefig(png, dpi=150, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"\nwrote {png}")
    print(f"wrote {pdf}")


if __name__ == "__main__":
    import sys
    stem = sys.argv[1] if len(sys.argv) > 1 else None
    main(stem)
