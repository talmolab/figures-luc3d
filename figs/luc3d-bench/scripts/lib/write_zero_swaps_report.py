#!/usr/bin/env python3
"""Render outputs/PAF_3d_kalman/zero_swaps_report.md from the merged metrics CSV.

Run after the pipeline orchestrator finishes. Reports per-stage contribution,
head-to-head vs Liezl on the shared subset, and pass/fail against the spec
floor/solid/stretch bars.
"""
from __future__ import annotations
from pathlib import Path
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
METRICS = REPO / "outputs" / "PAF_3d_kalman" / "metrics" / "per_camera_session_metrics.csv"
OUT = REPO / "outputs" / "PAF_3d_kalman" / "zero_swaps_report.md"

ORDER = [
    "sleap", "luc3d", "luc3d_paf_l1_tracklet", "liezl_cross_view",
    "sleap_nn_filtered", "luc3d_paf_l1_dedupe",
    "luc3d_paf_l1_dedupe_tracklet3d",
    "luc3d_paf_l1_dedupe_tracklet3d_stitched",
]


def summary_table(df: pd.DataFrame, tracker_order):
    rows = []
    for t in tracker_order:
        sub = df[df.tracker == t]
        if len(sub) == 0:
            continue
        rows.append({
            "tracker": t,
            "n_cam_sess": len(sub),
            "total_switches": int(sub["num_switches"].sum()),
            "per_sess": round(sub["num_switches"].sum() / len(sub), 2),
            "IDF1": round(sub["idf1"].mean(), 4),
            "IDP": round(sub["idp"].mean(), 4),
            "IDR": round(sub["idr"].mean(), 4),
        })
    return pd.DataFrame(rows)


def bar_check(final_row):
    switches = final_row["total_switches"]
    idf1 = final_row["IDF1"]
    idp = final_row["IDP"]
    return {
        "floor (≤1777 sw, IDF1 ≥0.730, IDP ≥0.940)":
            switches <= 1777 and idf1 >= 0.730 and idp >= 0.940,
        "solid (<1000 sw, IDF1 ≥0.780)":
            switches < 1000 and idf1 >= 0.780,
        "stretch (<500 sw, IDF1 ≥0.800)":
            switches < 500 and idf1 >= 0.800,
    }


def liezl_head_to_head(df):
    liezl = df[df.tracker == "liezl_cross_view"]
    final = df[df.tracker == "luc3d_paf_l1_dedupe_tracklet3d_stitched"]
    if len(liezl) == 0 or len(final) == 0:
        return "(liezl or final row missing — skip head-to-head)"
    shared = set(liezl[["session", "camera"]].apply(tuple, axis=1)) & \
             set(final[["session", "camera"]].apply(tuple, axis=1))
    if not shared:
        return "(no shared (session, camera) pairs)"
    liezl_sub = liezl[liezl[["session", "camera"]].apply(tuple, axis=1).isin(shared)]
    final_sub = final[final[["session", "camera"]].apply(tuple, axis=1).isin(shared)]
    return (f"On shared {len(shared)} cam-sessions:\n\n"
            f"- Liezl:   switches={int(liezl_sub['num_switches'].sum())}  "
            f"IDF1={liezl_sub['idf1'].mean():.4f}\n"
            f"- Pipeline: switches={int(final_sub['num_switches'].sum())}  "
            f"IDF1={final_sub['idf1'].mean():.4f}\n")


def main():
    if not METRICS.exists():
        print(f"ERROR: missing {METRICS}")
        return 1
    df = pd.read_csv(METRICS)
    summary = summary_table(df, ORDER)
    print(summary.to_string(index=False))

    body = ["# Zero-Swaps Strategy — Final Report", ""]
    body.append(f"**Generated:** {pd.Timestamp.utcnow().isoformat()}Z")
    body.append(f"**Spec:** `docs/superpowers/specs/2026-05-20-zero-swaps-strategy-design.md`")
    body.append(f"**Plan:** `docs/superpowers/plans/2026-05-20-zero-swaps-strategy.md`")
    body.append("")
    body.append("## Headline")
    body.append("")
    body.append(summary.to_markdown(index=False))
    body.append("")

    final = summary[summary.tracker == "luc3d_paf_l1_dedupe_tracklet3d_stitched"]
    if len(final):
        body.append("## Bar check")
        body.append("")
        for name, passed in bar_check(final.iloc[0]).items():
            body.append(f"- **{name}**: {'PASS' if passed else 'FAIL'}")
        body.append("")

    body.append("## Per-stage contribution")
    body.append("")
    body.append("Each subsequent row should reduce switches vs the previous, holding IDF1 floor.")
    body.append("")
    pipeline = ["luc3d_paf_l1_tracklet", "sleap_nn_filtered",
                "luc3d_paf_l1_dedupe", "luc3d_paf_l1_dedupe_tracklet3d",
                "luc3d_paf_l1_dedupe_tracklet3d_stitched"]
    body.append(summary[summary.tracker.isin(pipeline)].to_markdown(index=False))
    body.append("")

    body.append("## Head-to-head vs Liezl")
    body.append("")
    body.append(liezl_head_to_head(df))
    body.append("")

    body.append("## Honest caveats")
    body.append("")
    body.append("- True 0 switches is unreachable on visually-indistinct multi-animal data; proofread GT itself contains irreducible ambiguity.")
    body.append("- Stage 1 filter defaults were calibrated to SLAP-2M (mean_node_score 0.55 vs Stefan's 0.80) — Stefan's recipe over-culled on this dataset.")
    body.append("- See `outputs/PAF_3d_kalman/_stage{2,4}_grid/_eval_*.csv` for full sweep tables.")
    body.append("")
    body.append("## Artifacts")
    body.append("")
    body.append("- Eval CSVs: `outputs/PAF_3d_kalman/_eval_{sleap_nn_filtered,luc3d_paf_l1_dedupe,luc3d_paf_l1_dedupe_tracklet3d,luc3d_paf_l1_dedupe_tracklet3d_stitched}.csv`")
    body.append("- Stage 2 winner: `outputs/PAF_3d_kalman/paf_l1_dedupe_winner.json`")
    body.append("- Stage 4 winner: `outputs/PAF_3d_kalman/stage4_winner.json`")
    body.append("- Main figure: `outputs/PAF_3d_kalman/plots/figure_main.{png,pdf}` (if regenerated)")
    body.append("")

    OUT.write_text("\n".join(body))
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
