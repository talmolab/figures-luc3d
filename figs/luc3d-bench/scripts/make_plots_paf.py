"""4-way multi-panel figure: SLEAP vs ByteTrack vs LUC3D-baseline vs LUC3D+PAF.

Reads outputs/PAF_3d_kalman/metrics/per_camera_session_metrics.csv (which
combines all four trackers) and writes the headline figure + AUC table.

Same panel layout as scripts/make_plots.py but with a 4th tracker color.
"""
from __future__ import annotations

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

# Tracker key → display label and color.
# Removed +L1 flip (subsumed by tracklet) and +PAF L1+2+3 (over-corrects, distracts).
TRACKERS = ["sleap", "bytetrack", "luc3d", "luc3d_paf_l1", "luc3d_paf_l1_tracklet"]
TRACKER_LABEL = {
    "sleap": "SLEAP",
    "bytetrack": "ByteTrack",
    "luc3d": "LUC3D",
    "luc3d_paf_l1": "+PAF L1",
    "luc3d_paf_l1_flip": "+PAF L1+flip",
    "luc3d_paf_l1_tracklet": "+PAF L1+tracklet",
    "luc3d_paf_l1_phantom": "+PAF L1+phantom",
    "luc3d_paf_full": "+PAF L1+2+3",
}
PALETTE = {
    "sleap": "#7F7F7F",
    "bytetrack": "#E69F00",
    "luc3d": "#56B4E9",
    "luc3d_paf_l1": "#009E73",
    "luc3d_paf_l1_flip": "#D55E00",
    "luc3d_paf_l1_tracklet": "#CC79A7",
    "luc3d_paf_l1_phantom": "#F0E442",
    "luc3d_paf_full": "#0072B2",
}


def set_style():
    sns.set_theme(context="paper", style="ticks", font="DejaVu Sans")
    mpl.rcParams.update({
        "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
        "savefig.facecolor": "white", "axes.facecolor": "white",
        "axes.edgecolor": "0.15", "axes.linewidth": 0.8,
        "axes.titlesize": 8.5, "axes.titleweight": "regular", "axes.titlepad": 4,
        "axes.labelsize": 7.5, "axes.labelpad": 2.5,
        "axes.spines.top": False, "axes.spines.right": False,
        "xtick.labelsize": 6.8, "ytick.labelsize": 6.8,
        "xtick.color": "0.15", "ytick.color": "0.15",
        "xtick.major.size": 2.6, "ytick.major.size": 2.6,
        "xtick.major.width": 0.7, "ytick.major.width": 0.7,
        "xtick.direction": "out", "ytick.direction": "out",
        "legend.fontsize": 6.2, "legend.frameon": False,
        "legend.handlelength": 1.1, "legend.handletextpad": 0.4,
        "legend.borderaxespad": 0.2,
        "lines.linewidth": 1.3, "lines.markersize": 3.6,
        "axes.grid": False, "pdf.fonttype": 42, "ps.fonttype": 42, "font.size": 7.5,
    })


def add_panel_label(ax, text):
    ax.text(-0.22, 1.08, text, transform=ax.transAxes,
            fontsize=9.5, fontweight="bold", ha="left", va="bottom")


def fig_legend(fig):
    handles = [mpl.lines.Line2D([], [], color=PALETTE[t], marker="o",
                                markersize=4, markeredgecolor="white",
                                markeredgewidth=0.5, label=TRACKER_LABEL[t],
                                linewidth=1.6) for t in TRACKERS]
    fig.legend(handles=handles, loc="upper center", ncol=4,
               bbox_to_anchor=(0.5, 1.0), frameon=False,
               fontsize=7.5, columnspacing=1.5, handletextpad=0.5)


def panel_overall_bars(ax, df: pd.DataFrame):
    metrics = ["idf1", "idp", "idr"]
    labels = ["IDF1", "IDP\n(precision)", "IDR\n(accuracy)"]
    means = df.groupby("tracker")[metrics].mean()
    x = np.arange(len(metrics))
    w = 0.20
    for i, t in enumerate(TRACKERS):
        if t not in means.index: continue
        vals = means.loc[t, metrics].values
        ax.bar(x + (i - 1.5) * w, vals, w, color=PALETTE[t],
               edgecolor="white", linewidth=0.5, label=TRACKER_LABEL[t])
        if t == "luc3d_paf":
            for xi, v in zip(x, vals):
                ax.text(xi + (i - 1.5) * w, v + 0.02, f"{v:.2f}",
                        ha="center", va="bottom", fontsize=5.5, color=PALETTE[t])
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.05); ax.set_yticks(np.arange(0, 1.01, 0.2))
    ax.set_ylabel("score"); ax.set_title("Identity tracking (higher = better)")


SHORT_LABELS = {
    "sleap": "SLEAP",
    "bytetrack": "ByteTrack",
    "luc3d": "LUC3D",
    "luc3d_paf_l1": "+PAF L1",
    "luc3d_paf_l1_flip": "+L1 flip",
    "luc3d_paf_l1_tracklet": "+L1 tracklet",
    "luc3d_paf_l1_phantom": "+L1 phantom",
    "luc3d_paf_full": "+PAF L1+2+3",
}


def panel_switches_distribution(ax, df: pd.DataFrame):
    """Horizontal boxplot of per-camera-session ID switch counts.
    No fliers, log x-axis, compact tick labels. One row per tracker.
    """
    data = [df[df["tracker"] == t]["num_switches"].values for t in TRACKERS]
    means = [df[df["tracker"] == t]["num_switches"].mean() for t in TRACKERS]
    positions = np.arange(len(TRACKERS))
    palette_local = [PALETTE[t] for t in TRACKERS]

    bp = ax.boxplot(
        [d + 0.5 for d in data],  # shift for log scale
        positions=positions, widths=0.6, patch_artist=True,
        vert=False, showfliers=False, showmeans=True,
        medianprops=dict(color="black", linewidth=1.0),
        meanprops=dict(marker="D", markerfacecolor="white",
                       markeredgecolor="black", markersize=3.5),
        whiskerprops=dict(color="0.2", linewidth=0.7),
        capprops=dict(color="0.2", linewidth=0.7),
        boxprops=dict(linewidth=0.7),
    )
    for patch, c in zip(bp["boxes"], palette_local):
        patch.set_facecolor(c); patch.set_alpha(0.82); patch.set_edgecolor("0.2")

    ax.set_xscale("log")
    ax.set_yticks(positions)
    ax.set_yticklabels([SHORT_LABELS[t] for t in TRACKERS], fontsize=7)
    ax.set_xlabel("switches / session (log)")
    ax.set_title("Per-session distribution")
    # Annotate mean on each row.
    for i, t in enumerate(TRACKERS):
        ax.text(means[i] + 0.5, i, f" μ={means[i]:.1f}", va="center", fontsize=6.0, color="0.15")
    ax.invert_yaxis()  # SLEAP on top (matches legend order)
    ax.set_xlim(0.4, None)


def panel_switches_total(ax, df: pd.DataFrame):
    """Total ID switches across all 444 camera-sessions per tracker.
    Horizontal bars so the labels are readable; sorted ascending so the
    winning low-switch trackers float to the top.
    """
    totals = {t: int(df[df["tracker"] == t]["num_switches"].sum()) for t in TRACKERS}
    order = sorted(TRACKERS, key=lambda t: totals[t])  # lowest first
    vals = [totals[t] for t in order]
    colors = [PALETTE[t] for t in order]
    labels = [SHORT_LABELS[t] for t in order]
    positions = np.arange(len(order))

    ax.barh(positions, vals, color=colors, edgecolor="white", linewidth=0.5, height=0.65)
    ax.set_yticks(positions)
    ax.set_yticklabels(labels, fontsize=7)
    ax.invert_yaxis()  # lowest at top
    ax.set_xlabel("total switches across full bench")
    ax.set_title(f"Total ID switches (sum over {df['session'].nunique()*6:,} camera-sessions)")
    xmax = max(vals) if max(vals) > 0 else 1
    for i, v in enumerate(vals):
        ax.text(v + xmax * 0.012, i, f"{v:,}", va="center", fontsize=6.5, color="0.15")
    ax.set_xlim(0, xmax * 1.16)


def panel_idf1_distribution(ax, df: pd.DataFrame):
    data = [df[df["tracker"] == t]["idf1"].values for t in TRACKERS if t in df["tracker"].values]
    labels = [SHORT_LABELS[t] for t in TRACKERS if t in df["tracker"].values]
    palette_local = [PALETTE[t] for t in TRACKERS if t in df["tracker"].values]
    positions = np.arange(len(data))
    bp = ax.boxplot(
        data, positions=positions, widths=0.55, patch_artist=True,
        showmeans=True, medianprops=dict(color="black", linewidth=1.0),
        meanprops=dict(marker="D", markerfacecolor="white", markeredgecolor="black", markersize=3.4),
        flierprops=dict(marker="o", markerfacecolor="0.5", markeredgecolor="none", markersize=2.2, alpha=0.6),
        whiskerprops=dict(color="0.2", linewidth=0.7),
        capprops=dict(color="0.2", linewidth=0.7),
        boxprops=dict(linewidth=0.7),
    )
    for patch, c in zip(bp["boxes"], palette_local):
        patch.set_facecolor(c); patch.set_alpha(0.75); patch.set_edgecolor("0.2")
    ax.set_xticks(positions); ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=6.2)
    ax.set_ylim(-0.02, 1.05); ax.set_yticks(np.arange(0, 1.01, 0.2))
    ax.set_ylabel("IDF1 (per camera-session)")
    ax.set_title("Distribution of IDF1")


def panel_by_group(ax, df: pd.DataFrame, group_col: str, metric: str,
                   ylabel: str, title: str):
    grouped = df.groupby(["tracker", group_col])[metric].mean().reset_index()
    for t in TRACKERS:
        sub = grouped[grouped["tracker"] == t].sort_values(group_col)
        if sub.empty: continue
        ax.plot(sub[group_col], sub[metric], marker="o",
                label=TRACKER_LABEL[t], color=PALETTE[t],
                markeredgecolor="white", markeredgewidth=0.5)
    ax.set_xlabel(group_col); ax.set_ylabel(ylabel); ax.set_title(title)
    ax.set_xticks(sorted(grouped[group_col].unique()))


def _survival_curve(values: np.ndarray, n_steps: int = 201):
    vals = np.asarray(values, dtype=float); vals = vals[np.isfinite(vals)]
    taus = np.linspace(0.0, 1.0, n_steps)
    surv = np.array([(vals >= t).mean() if len(vals) else 0.0 for t in taus])
    return taus, surv


def _auc(taus, surv):
    return float(np.trapezoid(surv, taus))


def panel_survival(ax, df: pd.DataFrame, metric: str, title: str, xlabel: str = None):
    for t in TRACKERS:
        sub = df[df["tracker"] == t]
        if sub.empty: continue
        vals = sub[metric].values
        taus, surv = _survival_curve(vals)
        auc = _auc(taus, surv)
        ax.plot(taus, surv, color=PALETTE[t], linewidth=1.5,
                label=f"{TRACKER_LABEL[t]} (AUC={auc:.3f})")
        ax.fill_between(taus, 0, surv, color=PALETTE[t], alpha=0.07)
    ax.set_xlim(0, 1.0); ax.set_ylim(0, 1.02)
    ax.set_xticks(np.arange(0, 1.01, 0.2)); ax.set_yticks(np.arange(0, 1.01, 0.2))
    ax.set_xlabel(xlabel or f"{metric.upper()} threshold τ")
    ax.set_ylabel(f"frac. sessions with {metric.upper()} ≥ τ"); ax.set_title(title)
    ax.legend(loc="lower left", frameon=False, fontsize=5.8)


def panel_switches_survival(ax, df: pd.DataFrame, kmax: int = 60):
    ks = np.arange(0, kmax + 1)
    for t in TRACKERS:
        sub = df[df["tracker"] == t]
        if sub.empty: continue
        vals = sub["num_switches"].values
        surv = np.array([(vals <= k).mean() for k in ks])
        auc_norm = float(np.trapezoid(surv, ks) / kmax)
        ax.plot(ks, surv, color=PALETTE[t], linewidth=1.5,
                label=f"{TRACKER_LABEL[t]} (AUC={auc_norm:.3f})")
        ax.fill_between(ks, 0, surv, color=PALETTE[t], alpha=0.07)
    ax.set_xlim(0, kmax); ax.set_ylim(0, 1.02); ax.set_yticks(np.arange(0, 1.01, 0.2))
    ax.set_xlabel("max switches K")
    ax.set_ylabel("frac. sessions with switches ≤ K")
    ax.set_title("Switch survival (higher = better)")
    ax.legend(loc="lower right", frameon=False, fontsize=5.8)


def panel_gap_sweep(ax):
    """Total switches vs tracklet_gap for baseline LUC3D + tracklet-vote.
    U-curve with minimum near gap=30. Secondary y-axis shows IDP.
    """
    from pathlib import Path
    base_dir = Path("/root/vast/eric/luc3d-bench/outputs/PAF_3d_kalman")
    df_ref = pd.read_csv(base_dir / "_eval_baseline.csv")
    none_sw = int(df_ref[df_ref["tracker"]=="luc3d"]["num_switches"].sum())
    none_idp = df_ref[df_ref["tracker"]=="luc3d"]["idp"].mean()
    rows = []
    for g in [1, 2, 3, 5, 10, 15, 20, 30, 60, 120, 300, 600, 1200]:
        p = base_dir / (f"_eval_baseline_tracklet_g{g}.csv" if g != 30 else "_eval_baseline_tracklet.csv")
        if not p.exists(): continue
        sub = pd.read_csv(p); sub = sub[sub["tracker"]=="luc3d"]
        rows.append((g, int(sub["num_switches"].sum()), sub["idp"].mean()))
    gaps = [r[0] for r in rows]
    sw   = [r[1] for r in rows]
    idp  = [r[2] for r in rows]

    color_sw  = PALETTE["luc3d_paf_l1_tracklet"]
    color_idp = PALETTE["luc3d_paf_l1"]
    ax.plot(gaps, sw, marker="o", markersize=4, linewidth=1.4,
            color=color_sw, label="total switches")
    ax.axhline(none_sw, ls="--", linewidth=0.8, color="0.5", alpha=0.6)
    ax.text(gaps[-1], none_sw, f"  no postproc = {none_sw:,}",
            va="center", ha="right", fontsize=6.0, color="0.4")
    ax.set_xscale("log")
    ax.set_xlabel("tracklet_gap (frames; ≈ 1/30 s each)")
    ax.set_ylabel("total switches", color=color_sw)
    ax.tick_params(axis="y", labelcolor=color_sw)
    ax.set_title("Tracklet gap sweep — baseline LUC3D")
    imin = sw.index(min(sw))
    ax.scatter([gaps[imin]], [sw[imin]], s=80, facecolor="none",
               edgecolor="black", linewidth=1.2, zorder=5)
    ax.annotate(f"min @ gap={gaps[imin]}: {sw[imin]:,}",
                xy=(gaps[imin], sw[imin]),
                xytext=(8, 16), textcoords="offset points",
                fontsize=6.5, color="0.15",
                arrowprops=dict(arrowstyle="-", linewidth=0.5, color="0.4"))

    ax2 = ax.twinx()
    ax2.plot(gaps, idp, marker="s", markersize=2.5, linewidth=1.0,
             color=color_idp, alpha=0.85, linestyle=":")
    ax2.set_ylabel("precision (IDP)", color=color_idp)
    ax2.tick_params(axis="y", labelcolor=color_idp)


def panel_scatter(ax, df: pd.DataFrame, ref_tracker: str = "luc3d",
                  y_tracker: str = "luc3d_paf",
                  metric: str = "idf1", metric_label: str | None = None):
    metric_label = metric_label or metric.upper()
    pivot = df.groupby(["session", "tracker"])[metric].mean().unstack("tracker")
    if y_tracker not in pivot.columns or ref_tracker not in pivot.columns:
        ax.text(0.5, 0.5, "(data missing)", transform=ax.transAxes, ha="center")
        return
    pivot = pivot.dropna(subset=[y_tracker, ref_tracker])
    ax.plot([0, 1], [0, 1], ls="--", color="0.6", lw=0.8, zorder=1)
    ax.scatter(pivot[ref_tracker], pivot[y_tracker],
               s=16, color=PALETTE[y_tracker], alpha=0.75,
               edgecolor="white", linewidth=0.4, zorder=2)
    ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks(np.arange(0, 1.01, 0.25)); ax.set_yticks(np.arange(0, 1.01, 0.25))
    ax.set_xlabel(f"{TRACKER_LABEL[ref_tracker]} {metric_label}")
    ax.set_ylabel(f"{TRACKER_LABEL[y_tracker]} {metric_label}")
    n_above = int((pivot[y_tracker] > pivot[ref_tracker]).sum())
    n_total = len(pivot)
    ax.set_title(f"{TRACKER_LABEL[y_tracker]} vs {TRACKER_LABEL[ref_tracker]}\n"
                 f"({n_above}/{n_total} favor PAF+tracklet)", fontsize=7.5)


def make_figure(df: pd.DataFrame, out_stem: Path):
    fig = plt.figure(figsize=(8.2, 8.2))
    gs = fig.add_gridspec(nrows=3, ncols=3, hspace=0.65, wspace=0.50,
                          left=0.085, right=0.985, top=0.92, bottom=0.07)
    axes = [[fig.add_subplot(gs[r, c]) for c in range(3)] for r in range(3)]
    ax_a, ax_b, ax_c = axes[0]
    ax_d, ax_e, ax_f = axes[1]
    ax_g, ax_h, ax_i = axes[2]

    # Row 1 — switches focus
    panel_switches_distribution(ax_a, df)
    panel_switches_total(ax_b, df)
    panel_by_group(ax_c, df, "difficulty", "num_switches", "switches",
                   "Switches by difficulty")

    # Row 2 — identity quality
    panel_by_group(ax_d, df, "difficulty", "idf1", "IDF1", "IDF1 by difficulty")
    ax_d.set_ylim(0, 1.0); ax_d.set_yticks(np.arange(0, 1.01, 0.2))
    panel_by_group(ax_e, df, "difficulty", "idp", "IDP (precision)",
                   "Precision by difficulty")
    ax_e.set_ylim(0, 1.05); ax_e.set_yticks(np.arange(0, 1.01, 0.2))
    panel_gap_sweep(ax_f)

    # Row 3 — AUC + scatter
    panel_survival(ax_g, df, "idr", "Accuracy AUC", xlabel="accuracy (IDR) τ")
    panel_survival(ax_h, df, "idp", "Precision AUC", xlabel="precision (IDP) τ")
    panel_scatter(ax_i, df, ref_tracker="sleap", y_tracker="luc3d_paf_l1_tracklet",
                  metric="idr", metric_label="accuracy (IDR)")

    for row in axes:
        for ax in row:
            sns.despine(ax=ax, trim=False, offset=2)
    for lab, ax in zip("abcdefghi", [a for row in axes for a in row]):
        add_panel_label(ax, lab)

    fig_legend(fig)
    fig.savefig(out_stem.with_suffix(".png"))
    fig.savefig(out_stem.with_suffix(".pdf"))
    plt.close(fig)


def write_auc_summary(df: pd.DataFrame, out_path: Path):
    rows = []
    for t in TRACKERS:
        sub = df[df["tracker"] == t]
        if sub.empty: continue
        row = {"tracker": t}
        for metric in ["idf1", "idr", "idp"]:
            taus, surv = _survival_curve(sub[metric].values)
            row[f"{metric}_auc"] = _auc(taus, surv)
            row[f"{metric}_mean"] = float(sub[metric].mean())
        ks = np.arange(0, 61)
        surv_k = np.array([(sub["num_switches"].values <= k).mean() for k in ks])
        row["switches_auc_norm"] = float(np.trapezoid(surv_k, ks) / 60)
        row["switches_mean"] = float(sub["num_switches"].mean())
        rows.append(row)
    pd.DataFrame(rows).to_csv(out_path, sep="\t", index=False, float_format="%.4f")


def write_by_tracker_summary(df: pd.DataFrame, out_path: Path):
    """One row per tracker with all the headline metrics."""
    agg = df.groupby("tracker").agg(
        idf1=("idf1", "mean"), idp=("idp", "mean"), idr=("idr", "mean"),
        precision=("precision", "mean"), recall=("recall", "mean"),
        num_switches=("num_switches", "mean"),
        num_fragmentations=("num_fragmentations", "mean"),
        mostly_tracked=("mostly_tracked", "mean"),
        partially_tracked=("partially_tracked", "mean"),
        mostly_lost=("mostly_lost", "mean"),
        n=("idf1", "count"),
    ).reset_index()
    agg.to_csv(out_path, index=False)


def write_by_difficulty(df: pd.DataFrame, out_path: Path):
    grp = df.groupby(["tracker", "difficulty"]).agg(
        idf1=("idf1", "mean"),
        num_switches=("num_switches", "mean"),
        idp=("idp", "mean"),
        idr=("idr", "mean"),
        n=("idf1", "count"),
    ).reset_index()
    grp.to_csv(out_path, index=False)


def main():
    global TRACKERS
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(SRC)
    # Filter TRACKERS to only those present in the data, preserving order.
    present = set(df["tracker"].unique())
    TRACKERS = [t for t in TRACKERS if t in present]
    set_style()
    make_figure(df, PLOTS_DIR / "figure_main")
    write_auc_summary(df, METRICS_DIR / "auc_summary.tsv")
    write_by_tracker_summary(df, METRICS_DIR / "by_tracker.csv")
    write_by_difficulty(df, METRICS_DIR / "by_difficulty.csv")
    n_sessions = df["session"].nunique()
    n_trackers = df["tracker"].nunique()
    print(f"wrote {PLOTS_DIR / 'figure_main.png'} "
          f"({n_sessions} sessions, {n_trackers} trackers in data)")


if __name__ == "__main__":
    main()
