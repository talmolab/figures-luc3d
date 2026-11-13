"""Merge all per-tracker eval CSVs into a single per_camera_session_metrics CSV
covering 7 trackers:
  sleap, bytetrack, luc3d, luc3d_paf_l1, luc3d_paf_full, luc3d_paf_l1_flip,
  luc3d_paf_l1_tracklet, luc3d_paf_l1_phantom

Each PAF-variant CSV labels its tracker as "luc3d" — we rename to the variant
key. The sleap/bytetrack rows from one of them is enough for downstream plots.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
PAF_DIR = REPO / "outputs" / "PAF_3d_kalman"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", default=str(PAF_DIR / "_eval_baseline.csv"))
    ap.add_argument("--l1", default=str(PAF_DIR / "_eval_l1.csv"))
    ap.add_argument("--full", default=str(PAF_DIR / "_eval_tight.csv"))
    ap.add_argument("--flip", default=str(PAF_DIR / "_eval_flip.csv"))
    ap.add_argument("--tracklet", default=str(PAF_DIR / "_eval_tracklet.csv"))
    ap.add_argument("--phantom", default=str(PAF_DIR / "_eval_phantom.csv"))
    ap.add_argument("--liezl", default=str(PAF_DIR / "_eval_liezl.csv"))
    ap.add_argument("--tracklet-3d", dest="tracklet_3d",
                    default=str(PAF_DIR / "_eval_tracklet_3d_winner.csv"),
                    help="winning 3D-anchored tracklet vote eval CSV (Track A)")
    ap.add_argument("--out", default=str(PAF_DIR / "metrics/per_camera_session_metrics.csv"))
    args = ap.parse_args()

    files_and_labels = [
        (args.baseline, None),                  # keeps "luc3d", "sleap", "bytetrack"
        (args.l1, "luc3d_paf_l1"),
        (args.full, "luc3d_paf_full"),
        (args.flip, "luc3d_paf_l1_flip"),
        (args.tracklet, "luc3d_paf_l1_tracklet"),
        (args.phantom, "luc3d_paf_l1_phantom"),
        (args.liezl, "liezl_cross_view"),
        (args.tracklet_3d, "luc3d_paf_l1_tracklet_3d"),
    ]

    pieces = []
    for path, relabel in files_and_labels:
        if not Path(path).exists():
            print(f"  skip {path}: not found")
            continue
        df = pd.read_csv(path)
        if relabel is None:
            pieces.append(df)
        else:
            # Source tracker name is usually "luc3d" but a CSV may already be relabeled
            # to the target name. Match on either.
            sub = df[df["tracker"].isin(["luc3d", relabel])].copy()
            sub["tracker"] = relabel
            pieces.append(sub)
            print(f"  {path} → {relabel}: {len(sub)} rows")

    merged = pd.concat(pieces, ignore_index=True)
    out_path = Path(args.out); out_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out_path, index=False)
    print(f"\nwrote {out_path}: {len(merged)} rows, trackers = {sorted(merged['tracker'].unique())}")


if __name__ == "__main__":
    main()
