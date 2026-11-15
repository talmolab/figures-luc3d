"""Multi-panel figure for the LUC3D vs ByteTrack vs SLEAP benchmark.

Reads outputs/metrics/per_camera_session_metrics.csv and writes:
  - outputs/metrics/plots/figure_main.{png,pdf}  — composed 3x3 panel figure
  - outputs/metrics/plots/figure_supplementary.{png,pdf}
  - outputs/metrics/plots/panel_*.png            — each panel individually

MOTA / MOTP are intentionally NOT in the main figure: all three trackers consume
the same SLEAP detections, so MOTA is FN-dominated (>99% of error) and
essentially identical across trackers. The interesting axes are IDF1 / IDR /
IDP / ID-switches.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

REPO = Path(__file__).resolve().parents[1]
METRICS_DIR = REPO / "outputs" / "metrics"
PLOTS_DIR = METRICS_DIR / "plots"
SRC = METRICS_DIR / "per_camera_session_metrics.csv"

TRACKERS = ["sleap", "bytetrack", "luc3d"]
TRACKER_LABEL = {"sleap": "SLEAP", "bytetrack": "ByteTrack", "luc3d": "LUC3D"}
# Talmo/SLEAP-style sober palette — colorblind-safe (Okabe-Ito-adjacent).
PALETTE = {"sleap": "#7F7F7F", "bytetrack": "#E69F00", "luc3d": "#0072B2"}


def set_style():
    sns.set_theme(context="paper", style="ticks", font="DejaVu Sans")
    mpl.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": "0.15",
        "axes.linewidth": 0.8,
        "axes.titlesize": 8.5,
        "axes.titleweight": "regular",
        "axes.titlepad": 4,
        "axes.labelsize": 7.5,
        "axes.labelpad": 2.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.labelsize": 6.8,
        "ytick.labelsize": 6.8,
        "xtick.color": "0.15",
        "ytick.color": "0.15",
        "xtick.major.size": 2.6,
        "ytick.major.size": 2.6,
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "legend.fontsize": 6.8,
        "legend.frameon": False,
        "legend.handlelength": 1.1,
        "legend.handletextpad": 0.4,
        "legend.borderaxespad": 0.2,
        "lines.linewidth": 1.3,
        "lines.markersize": 3.6,
        "axes.grid": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "font.size": 7.5,
    })


def add_panel_label(ax, text):
    ax.text(-0.22, 1.08, text, transform=ax.transAxes,
            fontsize=9.5, fontweight="bold", ha="left", va="bottom")


def fig_legend(fig, axes_for_handles):
    handles = [mpl.lines.Line2D([], [], color=PALETTE[t], marker="o",
                                markersize=4, markeredgecolor="white",
                                markeredgewidth=0.5, label=TRACKER_LABEL[t],
                                linewidth=1.6) for t in TRACKERS]
    fig.legend(handles=handles, loc="upper center", ncol=3,
               bbox_to_anchor=(0.5, 1.0), frameon=False,
               fontsize=8, columnspacing=2.2, handletextpad=0.5)


# ---------- panels ----------

def panel_overall_bars(ax, df: pd.DataFrame):
    """Headline bars: IDF1 / IDP / IDR.

    IDF1 is the global F1 of the GT↔predicted trajectory assignment (= 2·IDP·IDR/(IDP+IDR));
    IDP is identity precision (of the predicted identities, how many matched the GT
    identity they were assigned to); IDR is identity recall (of the GT detections,
    how many got the right identity).
    """
    metrics = ["idf1", "idp", "idr"]
    labels = ["IDF1\n(identity F1)", "IDP\n(precision)", "IDR\n(recall / accuracy)"]
    means = df.groupby("tracker")[metrics].mean()
    x = np.arange(len(metrics))
    w = 0.26
    for i, t in enumerate(TRACKERS):
        vals = means.loc[t, metrics].values
        ax.bar(x + (i - 1) * w, vals, w, color=PALETTE[t],
               edgecolor="white", linewidth=0.5, label=TRACKER_LABEL[t])
        if t == "luc3d":
            for xi, v in zip(x, vals):
                ax.text(xi + (i - 1) * w, v + 0.02, f"{v:.2f}",
                        ha="center", va="bottom", fontsize=6.2, color=PALETTE[t])
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.05)
    ax.set_yticks(np.arange(0, 1.01, 0.2))
    ax.set_ylabel("score")
    ax.set_title("Identity tracking (higher = better)")


def panel_switches_bar(ax, df: pd.DataFrame):
    means = df.groupby("tracker")["num_switches"].mean()
    vals = [means[t] for t in TRACKERS]
    ax.bar(range(len(TRACKERS)), vals,
           color=[PALETTE[t] for t in TRACKERS],
           edgecolor="white", linewidth=0.5, width=0.62)
    ax.set_xticks(range(len(TRACKERS)))
    ax.set_xticklabels([TRACKER_LABEL[t] for t in TRACKERS])
    ax.set_ylabel("mean ID switches")
    ax.set_title("ID switches per camera-session\n(lower = better)")
    ymax = max(vals) if max(vals) > 0 else 1.0
    for i, v in enumerate(vals):
        ax.text(i, v + ymax * 0.025, f"{v:.1f}",
                ha="center", va="bottom", fontsize=6.5, color="0.2")
    ax.set_ylim(0, ymax * 1.18)


def panel_idf1_distribution(ax, df: pd.DataFrame):
    data = [df[df["tracker"] == t]["idf1"].values for t in TRACKERS]
    positions = np.arange(len(TRACKERS))
    bp = ax.boxplot(
        data, positions=positions, widths=0.55, patch_artist=True,
        showmeans=True, medianprops=dict(color="black", linewidth=1.0),
        meanprops=dict(marker="D", markerfacecolor="white",
                       markeredgecolor="black", markersize=3.4),
        flierprops=dict(marker="o", markerfacecolor="0.5",
                        markeredgecolor="none", markersize=2.2, alpha=0.6),
        whiskerprops=dict(color="0.2", linewidth=0.7),
        capprops=dict(color="0.2", linewidth=0.7),
        boxprops=dict(linewidth=0.7),
    )
    for patch, t in zip(bp["boxes"], TRACKERS):
        patch.set_facecolor(PALETTE[t])
        patch.set_alpha(0.75)
        patch.set_edgecolor("0.2")
    ax.set_xticks(positions)
    ax.set_xticklabels([TRACKER_LABEL[t] for t in TRACKERS])
    ax.set_ylim(-0.02, 1.05)
    ax.set_yticks(np.arange(0, 1.01, 0.2))
    ax.set_ylabel("IDF1 (per camera-session)")
    ax.set_title("Distribution of IDF1")


def panel_by_group(ax, df: pd.DataFrame, group_col: str, metric: str,
                   ylabel: str, title: str):
    grouped = df.groupby(["tracker", group_col])[metric].mean().reset_index()
    for t in TRACKERS:
        sub = grouped[grouped["tracker"] == t].sort_values(group_col)
        ax.plot(sub[group_col], sub[metric], marker="o",
                label=TRACKER_LABEL[t], color=PALETTE[t],
                markeredgecolor="white", markeredgewidth=0.5)
    ax.set_xlabel(group_col)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(sorted(grouped[group_col].unique()))


def _survival_curve(values: np.ndarray, n_steps: int = 201):
    """Return (thresholds, survival) for survival = P(X >= τ).

    Area under the survival curve over τ ∈ [0,1] equals E[X] (= mean) for
    X ∈ [0,1]. This is the "AUC" interpretation: a tracker that holds high IDF1
    on more sessions has a fatter survival curve and a larger AUC, which
    numerically equals its mean IDF1.
    """
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    taus = np.linspace(0.0, 1.0, n_steps)
    surv = np.array([(vals >= t).mean() if len(vals) else 0.0 for t in taus])
    return taus, surv


def _auc(taus, surv):
    # np.trapz is deprecated; np.trapezoid is the >=2.0 replacement.
    return float(np.trapezoid(surv, taus))


def panel_survival(ax, df: pd.DataFrame, metric: str, title: str,
                   xlabel: str | None = None):
    """Survival/AUC curve: fraction of camera-sessions with metric ≥ τ.

    For a metric on [0,1], area under the curve = mean of the metric. This
    gives one figure that shows both central tendency AND tail behavior — a
    tracker can win on the mean but lose on the "fraction of sessions where I
    got at least 0.8 IDF1" question, or vice versa. AUC is annotated in the
    legend so you can read it off directly.
    """
    for t in TRACKERS:
        vals = df[df["tracker"] == t][metric].values
        taus, surv = _survival_curve(vals)
        auc = _auc(taus, surv)
        ax.plot(taus, surv, color=PALETTE[t], linewidth=1.5,
                label=f"{TRACKER_LABEL[t]} (AUC={auc:.3f})")
        ax.fill_between(taus, 0, surv, color=PALETTE[t], alpha=0.08)
    ax.set_xlim(0, 1.0)
    ax.set_ylim(0, 1.02)
    ax.set_xticks(np.arange(0, 1.01, 0.2))
    ax.set_yticks(np.arange(0, 1.01, 0.2))
    ax.set_xlabel(xlabel or f"{metric.upper()} threshold τ")
    ax.set_ylabel(f"frac. sessions with {metric.upper()} ≥ τ")
    ax.set_title(title)
    ax.legend(loc="lower left", frameon=False, fontsize=6.2)


def panel_switches_survival(ax, df: pd.DataFrame, kmax: int = 60):
    """Survival on switches (lower better): fraction with ≤ K switches.

    Reads "fraction of camera-sessions with at most K switches". A larger area
    under this curve = better. AUC normalized by kmax so it's on [0,1].
    """
    ks = np.arange(0, kmax + 1)
    for t in TRACKERS:
        vals = df[df["tracker"] == t]["num_switches"].values
        surv = np.array([(vals <= k).mean() for k in ks])
        auc_norm = float(np.trapezoid(surv, ks) / kmax)
        ax.plot(ks, surv, color=PALETTE[t], linewidth=1.5,
                label=f"{TRACKER_LABEL[t]} (AUC={auc_norm:.3f})")
        ax.fill_between(ks, 0, surv, color=PALETTE[t], alpha=0.08)
    ax.set_xlim(0, kmax)
    ax.set_ylim(0, 1.02)
    ax.set_yticks(np.arange(0, 1.01, 0.2))
    ax.set_xlabel("max switches K")
    ax.set_ylabel("frac. sessions with switches ≤ K")
    ax.set_title("Switch survival (higher = better)")
    ax.legend(loc="lower right", frameon=False, fontsize=6.2)


def panel_by_camera(ax, df: pd.DataFrame):
    grouped = df.groupby(["tracker", "camera"])["idf1"].mean().unstack("tracker")
    cameras = sorted(grouped.index.tolist())
    x = np.arange(len(cameras))
    w = 0.26
    for i, t in enumerate(TRACKERS):
        ax.bar(x + (i - 1) * w, grouped.loc[cameras, t].values, w,
               color=PALETTE[t], edgecolor="white", linewidth=0.5,
               label=TRACKER_LABEL[t])
    ax.set_xticks(x)
    ax.set_xticklabels(cameras, rotation=0)
    ax.set_ylim(0, 1.0)
    ax.set_yticks(np.arange(0, 1.01, 0.2))
    ax.set_ylabel("IDF1")
    ax.set_title("IDF1 by camera view")


def panel_scatter(ax, df: pd.DataFrame, ref_tracker: str = "sleap"):
    pivot = df.groupby(["session", "tracker"])["idf1"].mean().unstack("tracker")
    pivot = pivot.dropna(subset=["luc3d", ref_tracker])
    ax.plot([0, 1], [0, 1], ls="--", color="0.6", lw=0.8, zorder=1)
    ax.scatter(pivot[ref_tracker], pivot["luc3d"],
               s=16, color=PALETTE["luc3d"], alpha=0.75,
               edgecolor="white", linewidth=0.4, zorder=2)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks(np.arange(0, 1.01, 0.25))
    ax.set_yticks(np.arange(0, 1.01, 0.25))
    ax.set_xlabel(f"{TRACKER_LABEL[ref_tracker]} IDF1")
    ax.set_ylabel("LUC3D IDF1")
    n_above = int((pivot["luc3d"] > pivot[ref_tracker]).sum())
    n_total = len(pivot)
    ax.set_title(f"LUC3D vs {TRACKER_LABEL[ref_tracker]}\n({n_above}/{n_total} sessions favor LUC3D)")


# ---------- supplementary panels ----------

def panel_pr_scatter(ax, df: pd.DataFrame):
    """ID precision vs ID recall scatter. Each tracker plotted as the per-session
    cloud + the cross-session mean (large dot). Position toward the upper-right =
    better. This visualizes the precision/recall *tradeoff* that the user asked
    about, where high IDP with low IDR (= ByteTrack's failure mode) sits left
    and the ideal upper-right is what LUC3D approaches.
    """
    for t in TRACKERS:
        sub = df[df["tracker"] == t]
        ax.scatter(sub["idr"], sub["idp"], s=6, color=PALETTE[t], alpha=0.18,
                   edgecolor="none")
        mr, mp = sub["idr"].mean(), sub["idp"].mean()
        ax.scatter([mr], [mp], s=80, color=PALETTE[t], edgecolor="white",
                   linewidth=0.8, zorder=5, label=f"{TRACKER_LABEL[t]} (mean)")
    ax.plot([0, 1], [0, 1], ls=":", color="0.7", lw=0.6, zorder=1)
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.02)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks(np.arange(0, 1.01, 0.25))
    ax.set_yticks(np.arange(0, 1.01, 0.25))
    ax.set_xlabel("IDR (identity recall / accuracy)")
    ax.set_ylabel("IDP (identity precision)")
    ax.set_title("Per-session identity precision vs recall")
    ax.legend(loc="lower left", frameon=False, fontsize=6.0)


def panel_fragmentations(ax, df: pd.DataFrame):
    grouped = df.groupby(["tracker", "difficulty"])["num_fragmentations"].mean().reset_index()
    for t in TRACKERS:
        sub = grouped[grouped["tracker"] == t].sort_values("difficulty")
        ax.plot(sub["difficulty"], sub["num_fragmentations"], marker="o",
                color=PALETTE[t], label=TRACKER_LABEL[t],
                markeredgecolor="white", markeredgewidth=0.5)
    ax.set_xlabel("difficulty")
    ax.set_ylabel("mean fragmentations")
    ax.set_title("Track fragmentations by difficulty")
    ax.set_xticks(sorted(grouped["difficulty"].unique()))


def panel_coverage(ax, df: pd.DataFrame):
    """Stacked MT / PT / ML."""
    means = df.groupby("tracker")[["mostly_tracked", "partially_tracked", "mostly_lost"]].mean()
    means = means.reindex(TRACKERS)
    colors = {"mostly_tracked": "#009E73", "partially_tracked": "#F0E442", "mostly_lost": "#999999"}
    bottoms = np.zeros(len(TRACKERS))
    for col, label in [("mostly_tracked", "MT"), ("partially_tracked", "PT"), ("mostly_lost", "ML")]:
        vals = means[col].fillna(0).values
        ax.bar(range(len(TRACKERS)), vals, bottom=bottoms,
               color=colors[col], edgecolor="white", linewidth=0.5, width=0.62,
               label=label)
        bottoms += vals
    ax.set_xticks(range(len(TRACKERS)))
    ax.set_xticklabels([TRACKER_LABEL[t] for t in TRACKERS])
    ax.set_ylabel("mean GT tracks per session")
    ax.set_title("Track coverage (MT/PT/ML)")
    ax.legend(loc="upper right", fontsize=6, ncol=3)


def panel_by_bedding(ax, df: pd.DataFrame):
    grouped = df.groupby(["tracker", "bedding"])["idf1"].mean().unstack("tracker")
    bedding_cats = sorted(grouped.index.tolist())
    x = np.arange(len(bedding_cats))
    w = 0.26
    for i, t in enumerate(TRACKERS):
        if t not in grouped.columns:
            continue
        ax.bar(x + (i - 1) * w, grouped.loc[bedding_cats, t].fillna(0).values, w,
               color=PALETTE[t], edgecolor="white", linewidth=0.5,
               label=TRACKER_LABEL[t])
    ax.set_xticks(x)
    ax.set_xticklabels(bedding_cats, rotation=0)
    ax.set_ylim(0, 1.0)
    ax.set_yticks(np.arange(0, 1.01, 0.2))
    ax.set_ylabel("IDF1")
    ax.set_title("IDF1 by bedding type")


# ---------- composition ----------

def make_figure(df: pd.DataFrame, out_stem: Path):
    """Main 3×3 figure. MOTA panels removed; replaced with switches-by-difficulty
    (now first-class) and IDF1/switches survival (AUC) curves."""
    fig = plt.figure(figsize=(7.8, 7.8))
    gs = fig.add_gridspec(
        nrows=3, ncols=3,
        hspace=0.65, wspace=0.50,
        left=0.085, right=0.985, top=0.92, bottom=0.07,
    )
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[0, 2])

    ax_d = fig.add_subplot(gs[1, 0])
    ax_e = fig.add_subplot(gs[1, 1])
    ax_f = fig.add_subplot(gs[1, 2])

    ax_g = fig.add_subplot(gs[2, 0])
    ax_h = fig.add_subplot(gs[2, 1])
    ax_i = fig.add_subplot(gs[2, 2])

    # Row 1 — headline
    panel_overall_bars(ax_a, df)
    panel_switches_bar(ax_b, df)
    panel_idf1_distribution(ax_c, df)

    # Row 2 — difficulty stratification (switches by difficulty is the focal panel)
    panel_by_group(ax_d, df, "difficulty", "idf1", "IDF1", "IDF1 by difficulty")
    ax_d.set_ylim(0, 1.0)
    ax_d.set_yticks(np.arange(0, 1.01, 0.2))

    panel_by_group(ax_e, df, "difficulty", "num_switches", "ID switches",
                   "ID switches by difficulty")

    panel_by_group(ax_f, df, "difficulty", "num_fragmentations",
                   "fragmentations", "Fragmentations by difficulty")

    # Row 3 — AUC curves + the LUC3D vs SLEAP head-to-head
    panel_survival(ax_g, df, "idf1", "IDF1 survival (higher = better)",
                   xlabel="IDF1 threshold τ")
    panel_switches_survival(ax_h, df)
    panel_scatter(ax_i, df, ref_tracker="sleap")

    for ax in (ax_a, ax_b, ax_c, ax_d, ax_e, ax_f, ax_g, ax_h, ax_i):
        sns.despine(ax=ax, trim=False, offset=2)

    for lab, ax in zip("abcdefghi",
                       [ax_a, ax_b, ax_c, ax_d, ax_e, ax_f, ax_g, ax_h, ax_i]):
        add_panel_label(ax, lab)

    fig_legend(fig, [ax_d])

    fig.savefig(out_stem.with_suffix(".png"))
    fig.savefig(out_stem.with_suffix(".pdf"))
    plt.close(fig)


def make_supplementary_figure(df: pd.DataFrame, out_stem: Path):
    """Supplementary panels: precision/recall scatter, IDR/IDP survival curves,
    error decomposition, track coverage, IDF1 by animal count + camera + bedding,
    LUC3D vs ByteTrack scatter. Still no MOTA/MOTP — those are not informative
    when detection sets are shared."""
    fig = plt.figure(figsize=(7.8, 7.8))
    gs = fig.add_gridspec(
        nrows=3, ncols=3,
        hspace=0.65, wspace=0.50,
        left=0.085, right=0.985, top=0.92, bottom=0.07,
    )
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[0, 2])
    ax_d = fig.add_subplot(gs[1, 0])
    ax_e = fig.add_subplot(gs[1, 1])
    ax_f = fig.add_subplot(gs[1, 2])
    ax_g = fig.add_subplot(gs[2, 0])
    ax_h = fig.add_subplot(gs[2, 1])
    ax_i = fig.add_subplot(gs[2, 2])

    panel_pr_scatter(ax_a, df)
    panel_survival(ax_b, df, "idr", "IDR (accuracy) survival",
                   xlabel="IDR threshold τ")
    panel_survival(ax_c, df, "idp", "IDP (precision) survival",
                   xlabel="IDP threshold τ")

    panel_by_group(ax_d, df, "animals", "idf1", "IDF1", "IDF1 by animal count")
    ax_d.set_ylim(0, 1.0)
    ax_d.set_yticks(np.arange(0, 1.01, 0.2))

    panel_by_group(ax_e, df, "animals", "num_switches", "ID switches",
                   "Switches by animal count")

    panel_by_camera(ax_f, df)

    panel_by_bedding(ax_g, df)
    panel_coverage(ax_h, df)
    panel_scatter(ax_i, df, ref_tracker="bytetrack")

    for ax in (ax_a, ax_b, ax_c, ax_d, ax_e, ax_f, ax_g, ax_h, ax_i):
        sns.despine(ax=ax, trim=False, offset=2)
    for lab, ax in zip("abcdefghi",
                       [ax_a, ax_b, ax_c, ax_d, ax_e, ax_f, ax_g, ax_h, ax_i]):
        add_panel_label(ax, lab)

    fig.savefig(out_stem.with_suffix(".png"))
    fig.savefig(out_stem.with_suffix(".pdf"))
    plt.close(fig)


def save_individual_panels(df: pd.DataFrame, out_dir: Path):
    individuals = [
        ("overall_bars", lambda ax: panel_overall_bars(ax, df), (3.3, 2.6)),
        ("switches", lambda ax: panel_switches_bar(ax, df), (3.0, 2.6)),
        ("idf1_distribution", lambda ax: panel_idf1_distribution(ax, df), (3.0, 2.8)),
        ("by_camera", lambda ax: panel_by_camera(ax, df), (3.6, 2.6)),
        ("by_bedding", lambda ax: panel_by_bedding(ax, df), (3.6, 2.6)),
        ("idf1_by_difficulty", lambda ax: panel_by_group(ax, df, "difficulty", "idf1", "IDF1", "IDF1 by difficulty"), (3.4, 2.6)),
        ("switches_by_difficulty", lambda ax: panel_by_group(ax, df, "difficulty", "num_switches", "ID switches", "ID switches by difficulty"), (3.4, 2.6)),
        ("fragmentations_by_difficulty", lambda ax: panel_fragmentations(ax, df), (3.4, 2.6)),
        ("idf1_by_animals", lambda ax: panel_by_group(ax, df, "animals", "idf1", "IDF1", "IDF1 by animal count"), (3.4, 2.6)),
        ("switches_by_animals", lambda ax: panel_by_group(ax, df, "animals", "num_switches", "ID switches", "Switches by animal count"), (3.4, 2.6)),
        ("scatter_luc3d_vs_bytetrack", lambda ax: panel_scatter(ax, df, "bytetrack"), (3.2, 3.2)),
        ("scatter_luc3d_vs_sleap", lambda ax: panel_scatter(ax, df, "sleap"), (3.2, 3.2)),
        ("pr_scatter", lambda ax: panel_pr_scatter(ax, df), (3.4, 3.4)),
        ("idf1_survival", lambda ax: panel_survival(ax, df, "idf1", "IDF1 survival (AUC = mean IDF1)"), (3.4, 2.8)),
        ("idr_survival", lambda ax: panel_survival(ax, df, "idr", "IDR (accuracy) survival"), (3.4, 2.8)),
        ("idp_survival", lambda ax: panel_survival(ax, df, "idp", "IDP (precision) survival"), (3.4, 2.8)),
        ("switches_survival", lambda ax: panel_switches_survival(ax, df), (3.4, 2.8)),
        ("track_coverage", lambda ax: panel_coverage(ax, df), (3.2, 2.8)),
    ]
    for name, fn, size in individuals:
        fig, ax = plt.subplots(figsize=size)
        fn(ax)
        sns.despine(ax=ax, trim=False, offset=2)
        if "by_difficulty" in name or "by_animals" in name:
            handles = [mpl.lines.Line2D([], [], color=PALETTE[t], marker="o",
                                        markersize=4, label=TRACKER_LABEL[t])
                       for t in TRACKERS]
            ax.legend(handles=handles, loc="best", frameon=False)
            if name.startswith("idf1"):
                ax.set_ylim(0, 1.0)
        fig.savefig(out_dir / f"panel_{name}.png", bbox_inches="tight")
        plt.close(fig)


def write_auc_summary(df: pd.DataFrame, out_path: Path):
    """Dump the survival AUCs as a small TSV the report can include."""
    rows = []
    for t in TRACKERS:
        sub = df[df["tracker"] == t]
        row = {"tracker": t}
        for metric in ["idf1", "idr", "idp"]:
            taus, surv = _survival_curve(sub[metric].values)
            row[f"{metric}_auc"] = _auc(taus, surv)
            row[f"{metric}_mean"] = float(sub[metric].mean())
        # switches AUC normalized to [0,60]
        ks = np.arange(0, 61)
        surv_k = np.array([(sub["num_switches"].values <= k).mean() for k in ks])
        row["switches_auc_norm"] = float(np.trapezoid(surv_k, ks) / 60)
        row["switches_mean"] = float(sub["num_switches"].mean())
        rows.append(row)
    pd.DataFrame(rows).to_csv(out_path, sep="\t", index=False, float_format="%.4f")


def main():
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(SRC)
    set_style()
    make_figure(df, PLOTS_DIR / "figure_main")
    make_supplementary_figure(df, PLOTS_DIR / "figure_supplementary")
    save_individual_panels(df, PLOTS_DIR)
    write_auc_summary(df, METRICS_DIR / "auc_summary.tsv")
    n_sessions = df["session"].nunique()
    n_trackers = df["tracker"].nunique()
    print(f"wrote {PLOTS_DIR / 'figure_main.png'} + figure_supplementary "
          f"({n_sessions} sessions, {n_trackers} trackers)")


if __name__ == "__main__":
    main()
