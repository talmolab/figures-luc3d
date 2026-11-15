"""Generate outputs/PAF_3d_kalman/final_report.md from the merged 5-tracker CSV.

Run AFTER scripts/make_plots_paf.py so the CSVs in outputs/PAF_3d_kalman/metrics/
exist. Produces a comprehensive markdown report.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
PAF_DIR = REPO / "outputs" / "PAF_3d_kalman"
METRICS_DIR = PAF_DIR / "metrics"

TRACKER_ORDER = ["sleap", "bytetrack", "luc3d", "luc3d_paf_l1", "luc3d_paf_full"]
TRACKER_LABEL = {
    "sleap": "SLEAP",
    "bytetrack": "ByteTrack",
    "luc3d": "LUC3D (baseline)",
    "luc3d_paf_l1": "LUC3D + PAF (L1 only)",
    "luc3d_paf_full": "LUC3D + PAF (L1+2+3, tight)",
}


def _survival_auc(values, n_steps=201):
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    taus = np.linspace(0, 1, n_steps)
    surv = np.array([(vals >= t).mean() if len(vals) else 0 for t in taus])
    return float(np.trapezoid(surv, taus))


def _switch_auc(values, kmax=60):
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    ks = np.arange(0, kmax + 1)
    surv = np.array([(vals <= k).mean() for k in ks])
    return float(np.trapezoid(surv, ks) / kmax)


def headline_table(df):
    rows = []
    for t in TRACKER_ORDER:
        sub = df[df["tracker"] == t]
        if sub.empty: continue
        rows.append({
            "tracker": TRACKER_LABEL[t],
            "n": len(sub),
            "idf1_mean": sub["idf1"].mean(),
            "idf1_auc": _survival_auc(sub["idf1"].values),
            "idp_mean": sub["idp"].mean(),
            "idr_mean": sub["idr"].mean(),
            "switches_mean": sub["num_switches"].mean(),
            "switches_auc": _switch_auc(sub["num_switches"].values),
            "fragmentations_mean": sub["num_fragmentations"].mean(),
        })
    return pd.DataFrame(rows)


def by_difficulty_table(df, metric):
    g = df.groupby(["tracker", "difficulty"])[metric].mean().unstack("tracker")
    if "luc3d" in g.columns:
        order = [t for t in TRACKER_ORDER if t in g.columns]
        g = g[order]
        g.columns = [TRACKER_LABEL[t] for t in order]
    return g


def pairwise_wins(df, ref="luc3d", challenger="luc3d_paf_full"):
    pivot = df.groupby(["session", "tracker"])["idf1"].mean().unstack("tracker")
    if ref not in pivot.columns or challenger not in pivot.columns:
        return None
    p = pivot.dropna(subset=[ref, challenger])
    wins = int((p[challenger] > p[ref]).sum())
    losses = int((p[challenger] < p[ref]).sum())
    ties = int((p[challenger] == p[ref]).sum())
    return {"sessions": len(p), "wins": wins, "losses": losses, "ties": ties}


def main():
    csv = METRICS_DIR / "per_camera_session_metrics.csv"
    if not csv.exists():
        print(f"missing {csv}; run scripts/run_paf_evals.sh first")
        return
    df = pd.read_csv(csv)
    headline = headline_table(df)

    lines = []
    lines.append("# Temporal 3D PAF for LUC3D — Overnight Benchmark Results\n")
    lines.append(f"**Date:** 2026-05-19 (overnight)\n")
    lines.append(f"**Sessions evaluated:** {df['session'].nunique()}")
    lines.append(f"**Per-camera-session rows:** {len(df)}")
    lines.append(f"**Trackers benchmarked:** {sorted(df['tracker'].unique().tolist())}\n")

    lines.append("## Headline (one row per tracker)\n")
    lines.append(headline.to_markdown(index=False, floatfmt=".4f"))
    lines.append("")

    # AUC summary
    lines.append("## AUC summary (higher = better, except switches_mean)\n")
    lines.append("**IDF1 AUC** = area under \"fraction of camera-sessions with IDF1 ≥ τ\" — equals the mean IDF1.\n")
    lines.append("**Switches AUC** = fraction-of-sessions-with-≤K-switches curve over K∈[0,60], normalized to [0,1].\n")
    lines.append(headline[["tracker", "idf1_auc", "switches_auc", "idf1_mean", "idp_mean", "idr_mean", "switches_mean"]]
                 .to_markdown(index=False, floatfmt=".4f"))
    lines.append("")

    # By difficulty — switches
    lines.append("## ID switches by difficulty (lower = better)\n")
    by_diff_sw = by_difficulty_table(df, "num_switches")
    lines.append(by_diff_sw.to_markdown(floatfmt=".2f"))
    lines.append("")

    # By difficulty — IDF1
    lines.append("## IDF1 by difficulty (higher = better)\n")
    by_diff_idf1 = by_difficulty_table(df, "idf1")
    lines.append(by_diff_idf1.to_markdown(floatfmt=".4f"))
    lines.append("")

    # Pairwise wins
    lines.append("## Pairwise per-session IDF1 — LUC3D+PAF vs baselines\n")
    for challenger in ["luc3d_paf_l1", "luc3d_paf_full"]:
        for ref in ["luc3d", "sleap"]:
            res = pairwise_wins(df, ref=ref, challenger=challenger)
            if res is None: continue
            lines.append(f"- {TRACKER_LABEL[challenger]} vs {TRACKER_LABEL[ref]}: "
                         f"**{res['wins']}** wins / {res['losses']} losses / {res['ties']} ties "
                         f"out of {res['sessions']} sessions")
    lines.append("")

    # Save the auc tables
    headline.to_csv(METRICS_DIR / "headline.csv", index=False)

    # Methods + ablation notes
    lines.append("---\n")
    lines.append("## What we built (overnight)\n")
    lines.append("### `lucid/kalman.js` (NEW)")
    lines.append("`KalmanIdentityTracker`: per-identity 6-D state (3D position + 3D velocity), "
                 "constant-velocity model. Decomposes into 3 independent 1-D KFs (one per spatial "
                 "axis) so the implementation is closed-form with no general matrix inverse.")
    lines.append("`mahalanobisSq(id, centroid, frameIdx)` returns the squared Mahalanobis distance "
                 "of an observed centroid under the KF-predicted Gaussian — the **backward flow "
                 "integral** along the t-1 → t trajectory.\n")
    lines.append("### `lucid/tracker.js` (MODIFIED — Landing 1)")
    lines.append("Added \"Signal 5\" to `reorderGroupsByPrevTargets`: "
                 "`kfWeight × exp(-maha²/2)`. This is the analog of SLEAP's spatial PAF line "
                 "integral, lifted from `(pixel × part-pair)` into `(time × 3D space × identity)`. "
                 "Soft, integrative — replaces a hard distance threshold with a Gaussian "
                 "likelihood score that saturates near 1 for close matches and falls off sharply "
                 "for outliers.\n")
    lines.append("### `scripts/luc3d_track_all.mjs` (MODIFIED — Landings 2 + 3)")
    lines.append("- **Landing 2 — hypothesis enumeration:** after the standard match, if any "
                 "matched group's KF Mahalanobis² exceeds the gate AND an alternative permutation "
                 "has total cost ≤ `pafSwapMarginInv × currentCost`, commit the swap. For n=2 "
                 "this is one alternative; for n=3, six alternatives; for n=4, twenty-four. "
                 "Tractable for SLAP-2M (n ≤ 4).")
    lines.append("- **Landing 3 — t → t+1 lookahead:** maintain a one-frame output buffer. On "
                 "frame t, retroactively re-score frame t-1's assignment using frame t's "
                 "evidence (compute total Δ centroid² between t-1 and t under each permutation). "
                 "Flip only when `currentCost − bestCost ≥ pafFlipMinAbsDelta` AND `bestCost ≤ "
                 "pafFlipMaxRatio × currentCost`. Deferred KF update keeps the forward pass "
                 "consistent. This is the **forward flow integral** leg of the PAF analog.\n")

    lines.append("---\n")
    lines.append("## Honest read of the numbers\n")
    # Programmatic interpretation
    base_row = headline[headline["tracker"] == TRACKER_LABEL["luc3d"]].iloc[0] if not headline[headline["tracker"] == TRACKER_LABEL["luc3d"]].empty else None
    l1_row   = headline[headline["tracker"] == TRACKER_LABEL["luc3d_paf_l1"]].iloc[0] if not headline[headline["tracker"] == TRACKER_LABEL["luc3d_paf_l1"]].empty else None
    full_row = headline[headline["tracker"] == TRACKER_LABEL["luc3d_paf_full"]].iloc[0] if not headline[headline["tracker"] == TRACKER_LABEL["luc3d_paf_full"]].empty else None
    sleap_row = headline[headline["tracker"] == TRACKER_LABEL["sleap"]].iloc[0] if not headline[headline["tracker"] == TRACKER_LABEL["sleap"]].empty else None
    if base_row is not None and l1_row is not None and full_row is not None and sleap_row is not None:
        d_l1 = l1_row["idf1_mean"] - base_row["idf1_mean"]
        d_full = full_row["idf1_mean"] - base_row["idf1_mean"]
        sw_l1 = l1_row["switches_mean"] - base_row["switches_mean"]
        sw_full = full_row["switches_mean"] - base_row["switches_mean"]
        d_full_vs_sleap = full_row["idf1_mean"] - sleap_row["idf1_mean"]
        sw_full_vs_sleap = full_row["switches_mean"] - sleap_row["switches_mean"]
        lines.append(f"- **PAF L1 vs baseline LUC3D:** IDF1 Δ = {d_l1:+.4f}, switches Δ = {sw_l1:+.2f}")
        lines.append(f"- **PAF L1+2+3 vs baseline LUC3D:** IDF1 Δ = {d_full:+.4f}, switches Δ = {sw_full:+.2f}")
        lines.append(f"- **PAF L1+2+3 vs SLEAP:** IDF1 Δ = {d_full_vs_sleap:+.4f}, switches Δ = {sw_full_vs_sleap:+.2f}")
        lines.append("")

    lines.append("## Figures + tables\n")
    lines.append("- `plots/figure_main.png` — 5-tracker comparison: IDF1/IDP/IDR bars, mean "
                 "switches, IDF1 distribution, by-difficulty curves, AUC survival curves, "
                 "LUC3D+PAF vs LUC3D baseline scatter")
    lines.append("- `metrics/per_camera_session_metrics.csv` — full per-camera-session metrics")
    lines.append("- `metrics/headline.csv` — one-row-per-tracker summary")
    lines.append("- `metrics/by_tracker.csv`, `by_difficulty.csv`, `auc_summary.tsv`")
    lines.append("")

    out = PAF_DIR / "final_report.md"
    out.write_text("\n".join(lines))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
