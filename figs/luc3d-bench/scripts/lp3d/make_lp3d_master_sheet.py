"""Write outputs/lp3d_master_sheet.tsv: a copy of predictions_master_sheet.tsv
that the LUC3D bench reads, but points the per-camera predictions_h5 to lp3d_predictions_h5s/.

We don't actually change the *_inference_h5 columns inside the sheet (the LUC3D driver doesn't
read those — it reads --pred-h5-dir). Instead we just keep the master sheet as-is and the
bench is run with --pred-h5-dir pointing to lp3d_predictions_h5s/.

So this script is a no-op verifier: it just copies the master sheet and prints which sessions
have LP3D coverage.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
SRC_MASTER = REPO / "outputs" / "predictions_master_sheet.tsv"
LP3D_H5_DIR = REPO / "outputs" / "lp3d_predictions_h5s"
DST_MASTER = REPO / "outputs" / "lp3d_master_sheet.tsv"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", type=Path, default=SRC_MASTER)
    ap.add_argument("--lp3d-dir", type=Path, default=LP3D_H5_DIR)
    ap.add_argument("--dst", type=Path, default=DST_MASTER)
    args = ap.parse_args()

    df = pd.read_csv(args.source, sep="\t", index_col=0)

    coverage = {}
    cams = ["back", "backL", "mid", "midL", "top", "topL"]
    for cam in cams:
        h5 = args.lp3d_dir / f"{cam}_predictions.h5"
        if not h5.exists():
            coverage[cam] = []
            continue
        with h5py.File(h5, "r") as f:
            arr = f["tracks"][:]
            has_data = np.isfinite(arr).any(axis=(1, 2, 3, 4))
            coverage[cam] = [int(i) for i, v in enumerate(has_data) if v]

    # write copy
    args.dst.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.dst, sep="\t")
    print(f"wrote {args.dst}")
    # summary
    print(f"\nLP3D coverage by camera:")
    for cam, idxs in coverage.items():
        if idxs:
            sessions = df.loc[idxs, "session"].astype(str).tolist()
            print(f"  {cam}: {len(idxs)} sessions → {sessions[:5]}{'...' if len(sessions)>5 else ''}")
        else:
            print(f"  {cam}: NO COVERAGE")


if __name__ == "__main__":
    main()
