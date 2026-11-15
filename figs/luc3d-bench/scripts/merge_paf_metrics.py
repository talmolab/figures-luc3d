"""Merge baseline 4-tracker metrics with PAF tracker metrics into a single CSV.

Inputs:
  outputs/metrics/per_camera_session_metrics.csv         — sleap, bytetrack, luc3d (baseline)
  outputs/PAF_3d_kalman/metrics/_paf_only.csv           — luc3d-renamed-to-luc3d_paf

Output:
  outputs/PAF_3d_kalman/metrics/per_camera_session_metrics.csv  — 4 trackers combined

After this, scripts/make_plots_paf.py can be run to generate figures.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
PAF_DIR = REPO / "outputs" / "PAF_3d_kalman"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", default=str(REPO / "outputs/metrics/per_camera_session_metrics.csv"))
    ap.add_argument("--paf-csv", required=True, help="evaluate.py output run on PAF results")
    ap.add_argument("--paf-label", default="luc3d_paf", help="rename luc3d→this in the PAF CSV")
    ap.add_argument("--out", default=str(PAF_DIR / "metrics/per_camera_session_metrics.csv"))
    args = ap.parse_args()

    base = pd.read_csv(args.baseline)
    paf = pd.read_csv(args.paf_csv)

    # In the PAF CSV, the rows we want are the luc3d rows; relabel to args.paf_label.
    paf = paf[paf["tracker"] == "luc3d"].copy()
    paf["tracker"] = args.paf_label

    merged = pd.concat([base, paf], ignore_index=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(args.out, index=False)
    print(f"wrote {args.out}: {len(merged)} rows ({merged['tracker'].unique().tolist()})")


if __name__ == "__main__":
    main()
