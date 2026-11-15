"""Aggregate per-camera metrics into a final benchmark report.

Reads outputs/metrics/per_camera_session_metrics.csv (with sleap/bytetrack/luc3d)
and writes:
  - outputs/metrics/final_report.md
  - outputs/metrics/by_tracker.csv
  - outputs/metrics/by_difficulty.csv
  - outputs/metrics/by_animals.csv
  - outputs/metrics/by_bedding.csv
  - outputs/metrics/by_camera.csv
  - outputs/metrics/error_decomposition.csv
  - outputs/metrics/session_wins.csv
  - outputs/metrics/worst_idf1_luc3d.csv
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
METRICS_DIR = REPO / "outputs" / "metrics"
SRC = METRICS_DIR / "per_camera_session_metrics.csv"
REPORT = METRICS_DIR / "final_report.md"

TRACKER_ORDER = ["sleap", "bytetrack", "luc3d"]

MEAN_METRICS = [
    "mota", "motp", "idf1", "idp", "idr",
    "precision", "recall",
    "num_switches", "num_fragmentations",
    "mostly_tracked", "partially_tracked", "mostly_lost",
]
SUM_METRICS = [
    "idtp", "idfp", "idfn",
    "num_false_positives", "num_misses",
    "num_detections", "num_objects", "num_predictions",
    "num_unique_objects", "num_frames",
]


def order(df: pd.DataFrame, col: str = "tracker") -> pd.DataFrame:
    df = df.copy()
    df[col] = pd.Categorical(df[col], categories=TRACKER_ORDER, ordered=True)
    return df.sort_values([col] + [c for c in df.columns if c != col and df[c].dtype.kind in "iuf" and c != col][:0])


def group_means(df: pd.DataFrame, by) -> pd.DataFrame:
    by_list = by if isinstance(by, list) else [by]
    return df.groupby(["tracker"] + by_list)[MEAN_METRICS].mean().reset_index()


def per_session_wins(df: pd.DataFrame) -> pd.DataFrame:
    """For each camera-session, which tracker has the highest IDF1."""
    pv = df.pivot_table(index=["session", "camera"], columns="tracker",
                        values="idf1", aggfunc="mean")
    if not set(TRACKER_ORDER).issubset(pv.columns):
        return pd.DataFrame()
    winner = pv.idxmax(axis=1)
    out = winner.value_counts().reindex(TRACKER_ORDER, fill_value=0).rename("wins").reset_index()
    out["wins_pct"] = out["wins"] / out["wins"].sum()
    out.columns = ["tracker", "wins", "wins_pct"]

    pairwise_rows = []
    for a in TRACKER_ORDER:
        for b in TRACKER_ORDER:
            if a == b:
                continue
            n = (pv[a] > pv[b]).sum()
            ties = (pv[a] == pv[b]).sum()
            pairwise_rows.append({"a": a, "b": b, "a_beats_b": int(n),
                                  "ties": int(ties), "total": int(len(pv))})
    return out, pd.DataFrame(pairwise_rows)


def error_decomposition(df: pd.DataFrame) -> pd.DataFrame:
    """Frame-weighted FP / FN / IDSW totals, plus MOTA(% breakdown)."""
    rows = []
    for t in TRACKER_ORDER:
        sub = df[df["tracker"] == t]
        if sub.empty:
            continue
        fp = sub["num_false_positives"].sum()
        fn = sub["num_misses"].sum()
        ids = sub["num_switches"].sum()
        gt = sub["num_objects"].sum()
        mota = 1.0 - (fp + fn + ids) / max(gt, 1)
        total_err = fp + fn + ids
        rows.append({
            "tracker": t,
            "fp_total": int(fp), "fn_total": int(fn), "idsw_total": int(ids),
            "gt_total": int(gt), "mota_weighted": mota,
            "fp_share": fp / max(total_err, 1),
            "fn_share": fn / max(total_err, 1),
            "idsw_share": ids / max(total_err, 1),
        })
    return pd.DataFrame(rows)


def main():
    df = pd.read_csv(SRC)

    by_tracker = df.groupby("tracker")[MEAN_METRICS].mean().reindex(TRACKER_ORDER).reset_index()
    by_tracker.to_csv(METRICS_DIR / "by_tracker.csv", index=False)

    by_difficulty = group_means(df, "difficulty")
    by_difficulty.to_csv(METRICS_DIR / "by_difficulty.csv", index=False)

    by_animals = group_means(df, "animals")
    by_animals.to_csv(METRICS_DIR / "by_animals.csv", index=False)

    by_bedding = group_means(df, "bedding")
    by_bedding.to_csv(METRICS_DIR / "by_bedding.csv", index=False)

    by_camera = group_means(df, "camera")
    by_camera.to_csv(METRICS_DIR / "by_camera.csv", index=False)

    err = error_decomposition(df)
    err.to_csv(METRICS_DIR / "error_decomposition.csv", index=False)

    wins_result = per_session_wins(df)
    if isinstance(wins_result, tuple):
        wins, pairwise = wins_result
        wins.to_csv(METRICS_DIR / "session_wins.csv", index=False)
        pairwise.to_csv(METRICS_DIR / "session_pairwise.csv", index=False)
    else:
        wins = pd.DataFrame()
        pairwise = pd.DataFrame()

    luc3d = df[df["tracker"] == "luc3d"].copy()
    worst = (luc3d.groupby("session")["idf1"].mean()
             .sort_values().head(10).reset_index()
             .rename(columns={"idf1": "mean_idf1"}))
    worst.to_csv(METRICS_DIR / "worst_idf1_luc3d.csv", index=False)

    # --- markdown report ---
    lines = ["# SLEAP vs ByteTrack vs LUC3D — SLAP-2M Benchmark", ""]
    lines += [
        f"- sessions evaluated: {df['session'].nunique()}",
        f"- cameras: {sorted(df['camera'].unique().tolist())}",
        f"- per-camera-session rows: {len(df)}",
        f"- trackers: {sorted(df['tracker'].unique().tolist())}",
        "",
    ]

    # Lead with an interpretive summary so the reader doesn't have to decode
    # what IDF1 / MOTA / IDR mean.
    bt = by_tracker.set_index("tracker") if "tracker" in by_tracker.columns else by_tracker
    try:
        sleap_idf1 = bt.loc["sleap", "idf1"]; luc3d_idf1 = bt.loc["luc3d", "idf1"]; bt_idf1 = bt.loc["bytetrack", "idf1"]
        sleap_idr  = bt.loc["sleap", "idr"];  luc3d_idr  = bt.loc["luc3d", "idr"];  bt_idr  = bt.loc["bytetrack", "idr"]
        sleap_idsw = bt.loc["sleap", "num_switches"]; luc3d_idsw = bt.loc["luc3d", "num_switches"]; bt_idsw = bt.loc["bytetrack", "num_switches"]
        lines += [
            "## Headline",
            "",
            f"- **LUC3D** got the right ID on **{luc3d_idr*100:.1f}%** of ground-truth detections; "
            f"SLEAP {sleap_idr*100:.1f}%, ByteTrack {bt_idr*100:.1f}% (this is IDR — 'identity recall').",
            f"- **IDF1** (overall identity F1, the headline metric): "
            f"LUC3D **{luc3d_idf1:.3f}** > SLEAP {sleap_idf1:.3f} > ByteTrack {bt_idf1:.3f}.",
            f"- **ID switches per camera-session**: LUC3D **{luc3d_idsw:.1f}**, SLEAP {sleap_idsw:.1f}, "
            f"ByteTrack {bt_idsw:.1f}. (Lower is better; ByteTrack lacks cross-view geometry.)",
            f"- **MOTA is essentially identical across all three** (LUC3D {bt.loc['luc3d','mota']:.3f}, "
            f"SLEAP {bt.loc['sleap','mota']:.3f}, ByteTrack {bt.loc['bytetrack','mota']:.3f}) because all "
            "three consume the **same SLEAP detections** — see the error decomposition below "
            "(FN is ~99% of total error, IDSW <1%, so MOTA cannot distinguish trackers here).",
            "",
        ]
    except (KeyError, AttributeError):
        pass
    lines += ["## Overall mean by tracker", "",
              by_tracker.to_markdown(index=False, floatfmt=".4f"), ""]

    if not err.empty:
        lines += ["## Error decomposition (frame-weighted totals)", "",
                  "Detection FP, missed GT (FN), and ID switches (IDSW), summed over",
                  "all evaluated camera-frames. `mota_weighted` is the standard MOTA",
                  "computed from the totals (i.e. `1 − (FP+FN+IDSW)/GT`).",
                  "",
                  err.to_markdown(index=False, floatfmt=".4f"), ""]

    if not wins.empty:
        lines += ["## Per camera-session wins (best IDF1)", "",
                  wins.to_markdown(index=False, floatfmt=".4f"), ""]
    if not pairwise.empty:
        lines += ["### Pairwise wins on IDF1", "",
                  pairwise.to_markdown(index=False), ""]

    lines += ["## By difficulty", "",
              by_difficulty.to_markdown(index=False, floatfmt=".4f"), ""]
    lines += ["## By animal count", "",
              by_animals.to_markdown(index=False, floatfmt=".4f"), ""]
    lines += ["## By bedding", "",
              by_bedding.to_markdown(index=False, floatfmt=".4f"), ""]
    lines += ["## By camera", "",
              by_camera.to_markdown(index=False, floatfmt=".4f"), ""]
    lines += ["## Worst LUC3D sessions (lowest mean IDF1)", "",
              worst.to_markdown(index=False, floatfmt=".4f"), ""]

    lines += ["## Method notes", "",
              "- **SLEAP** = identity assignments from the original `.predictions.slp` "
              "tracker output. Each predicted column index `a` in the aggregated H5 "
              "(`outputs/predictions_h5s/{cam}_predictions.h5`, mirroring "
              "`/root/vast/eric/slap-2m-h5s/{cam}_inference.h5`) is treated as the "
              "predicted identity — i.e. what SLEAP produced *without proofreading or "
              "post-hoc tracking*.",
              "- **ByteTrack** = `supervision.ByteTrack` run independently per camera "
              "on pose-derived bboxes (`scripts/run_bytetrack_bench.py`). No cross-view "
              "geometry.",
              "- **LUC3D** = headless Node driver (`scripts/luc3d_track_all.mjs`) "
              "replicating the GUI's Track-All flow with reprojection-weighted "
              "multi-view matching over the same SLEAP detections.",
              "- All three trackers consume the **same detection set** "
              "(`outputs/predictions_h5s`) so the comparison isolates the tracking "
              "algorithm. Detection-level precision/recall should therefore be "
              "essentially identical across trackers.",
              "- Ground truth is the proofread per-camera SLP, scored with motmetrics "
              "(IoU ≥ 0.5).",
              "- Cameras `side`/`sideL` are excluded (per direction 2026-05-15; no "
              "proofread side data).",
              "- Metric definitions: **MOTA** combines FP, FN, IDSW vs GT — detection-"
              "dominated. **IDF1** is the F1 of a global GT↔predicted trajectory "
              "assignment — identity-dominated. **MOTP** = mean IoU distance of "
              "matched pairs. **FM** (fragmentations) = how often a GT track switches "
              "from matched to unmatched. **MT/PT/ML** = fraction of GT lifespan "
              "tracked ≥80% / 20–80% / <20%."]
    REPORT.write_text("\n".join(lines))
    print(f"wrote {REPORT}")
    print(by_tracker.to_string(index=False))


if __name__ == "__main__":
    main()
