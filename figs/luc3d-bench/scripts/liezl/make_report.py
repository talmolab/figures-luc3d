"""Generate the Track B report comparing liezl_cross_view to LUC3D + tracklet.

Produces `outputs/PAF_3d_kalman/track_b_report.md` and prints to stdout.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

REPO = Path("/root/vast/eric/luc3d-bench")
PAF = REPO / "outputs/PAF_3d_kalman"


def fmt(x, prec=4):
    if isinstance(x, float):
        return f"{x:.{prec}f}"
    return str(x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--liezl-csv", default=str(PAF / "_eval_liezl.csv"))
    ap.add_argument("--baseline-csv", default=str(PAF / "metrics/per_camera_session_metrics.csv"))
    ap.add_argument("--out", default=str(PAF / "track_b_report.md"))
    args = ap.parse_args()

    liezl = pd.read_csv(args.liezl_csv)
    base = pd.read_csv(args.baseline_csv)

    # Ensure tracker labels are unique. Keep only the trackers we care about from base
    # to make rows comparable.
    if "liezl_cross_view" not in liezl["tracker"].unique():
        liezl = liezl.copy()
        liezl.loc[liezl["tracker"] == "luc3d", "tracker"] = "liezl_cross_view"
    # Filter base to drop any preexisting liezl_cross_view from a previous run.
    base = base[base["tracker"] != "liezl_cross_view"]

    merged = pd.concat(
        [base, liezl[liezl["tracker"] == "liezl_cross_view"]],
        ignore_index=True,
    )

    keep_trackers = [
        "luc3d_paf_l1_tracklet",
        "luc3d_paf_l1",
        "luc3d",
        "liezl_cross_view",
        "sleap",
        "bytetrack",
    ]
    merged = merged[merged["tracker"].isin(keep_trackers)]

    # Overall numbers
    overall = (
        merged.groupby("tracker")
        .agg(
            idf1=("idf1", "mean"),
            idp=("idp", "mean"),
            idr=("idr", "mean"),
            switches=("num_switches", "sum"),
            frags=("num_fragmentations", "sum"),
            mota=("mota", "mean"),
            n_camsessions=("session", "count"),
        )
        .reindex(keep_trackers)
        .reset_index()
    )

    # Per difficulty
    per_diff = (
        merged.groupby(["tracker", "difficulty"])
        .agg(
            idf1=("idf1", "mean"),
            switches=("num_switches", "sum"),
            n_camsessions=("session", "count"),
        )
        .reset_index()
    )

    lines: list[str] = []
    lines.append("# Track B Report — Liezl `CrossViewTracker` benchmark on SLAP-2M")
    lines.append("")
    lines.append("- **Tracker:** `sleap_3d.tracker.CrossViewTracker` (`talmolab/sleap-3d`).")
    lines.append("- **Inputs:** raw `.predictions.slp` (non-proofread) per camera; 6 bench views.")
    lines.append("- **Hyperparameters used:** corr2d=1.0, corr3d=1.0, vel_thresh=10.0, dist_thresh=50.0, time_penalty=0.1.")
    lines.append(f"- **Sessions evaluated:** {liezl['session'].nunique()} of 74 — see `_eval_liezl.csv`.")
    lines.append("")
    lines.append("## Setup notes")
    lines.append("- The repo URL in the plan (`talmolab/sleap-3d`) is 404 on public github. The package was installed from a local clone at `/root/vast/eric/sleap-3d` via `pip install -e`.")
    lines.append("- The `CrossViewTracker.track(...)` API is `(labels, session, start_frame_idx, end_frame_idx, ...)` — not the kwargs in the plan. Updated accordingly.")
    lines.append("- The tracker mutates `Instance.track` in place and assigns `sio.Track(name=\"cross-view\")` for every new cross-view target — so per-target identities must be disambiguated by `id(Track)`, not by `Track.name`.")
    lines.append("- For each PredictedInstance we cache `(camera, raw_track_idx)` BEFORE running the tracker — that mapping is used to emit `\"{cam}:{raw_track_idx}\"` keys in the LUC3D JSON schema.")
    lines.append("")
    lines.append("## Headline numbers")
    lines.append("")
    lines.append("| Tracker | Mean IDF1 | Mean IDP | Mean IDR | Total switches | Total frags | Mean MOTA | (cam·session) rows |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for _, r in overall.iterrows():
        lines.append(
            f"| `{r['tracker']}` | {fmt(r['idf1'])} | {fmt(r['idp'])} | {fmt(r['idr'])} | "
            f"{int(r['switches'])} | {int(r['frags'])} | {fmt(r['mota'])} | {int(r['n_camsessions'])} |"
        )
    lines.append("")
    lines.append("## Per-difficulty breakdown (IDF1 mean / switches sum)")
    lines.append("")
    difficulties = sorted(per_diff["difficulty"].unique())
    header = "| Tracker | " + " | ".join(f"D{int(d)}" for d in difficulties) + " |"
    sep = "|---|" + "|".join(["---"] * len(difficulties)) + "|"
    lines.append(header)
    lines.append(sep)
    for t in keep_trackers:
        cells = []
        for d in difficulties:
            sub = per_diff[(per_diff["tracker"] == t) & (per_diff["difficulty"] == d)]
            if len(sub) == 0:
                cells.append("—")
            else:
                row = sub.iloc[0]
                cells.append(f"{row['idf1']:.3f}/{int(row['switches'])}")
        lines.append(f"| `{t}` | " + " | ".join(cells) + " |")

    # Read
    lines.append("")
    lines.append("## Honest read")
    lines.append("")
    liezl_row = overall[overall["tracker"] == "liezl_cross_view"]
    tracklet_row = overall[overall["tracker"] == "luc3d_paf_l1_tracklet"]
    if len(liezl_row) and len(tracklet_row):
        liezl_idf1 = liezl_row["idf1"].iloc[0]
        tracklet_idf1 = tracklet_row["idf1"].iloc[0]
        liezl_sw = int(liezl_row["switches"].iloc[0])
        tracklet_sw = int(tracklet_row["switches"].iloc[0])
        diff_idf1 = liezl_idf1 - tracklet_idf1
        diff_sw = liezl_sw - tracklet_sw
        sign = "+" if diff_idf1 >= 0 else ""
        winner_idf1 = "liezl_cross_view" if diff_idf1 > 0 else "luc3d_paf_l1_tracklet"
        winner_sw = "liezl_cross_view" if diff_sw < 0 else "luc3d_paf_l1_tracklet"
        lines.append(f"- IDF1: `liezl_cross_view`={liezl_idf1:.4f} vs `luc3d_paf_l1_tracklet`={tracklet_idf1:.4f}  ({sign}{diff_idf1:.4f}; winner: **{winner_idf1}**)")
        lines.append(f"- Total switches: `liezl_cross_view`={liezl_sw} vs `luc3d_paf_l1_tracklet`={tracklet_sw}  ({diff_sw:+d}; winner: **{winner_sw}**)")

    p = Path(args.out)
    p.write_text("\n".join(lines) + "\n")
    print(f"wrote {p}")
    print()
    print("\n".join(lines))


if __name__ == "__main__":
    main()
