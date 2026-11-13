"""Run `scripts/evaluate.py` on the Liezl CrossViewTracker outputs, then
relabel the tracker column to `liezl_cross_view` and write a merged-friendly CSV.

Outputs:
  outputs/PAF_3d_kalman/_eval_liezl.csv  (with tracker='liezl_cross_view')
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd

REPO = Path("/root/vast/eric/luc3d-bench")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument(
        "--luc3d-dir",
        default=str(REPO / "outputs/PAF_3d_kalman/liezl_cross_view_results"),
    )
    ap.add_argument(
        "--out-csv",
        default=str(REPO / "outputs/PAF_3d_kalman/_eval_liezl.csv"),
    )
    args = ap.parse_args()

    cmd = [
        sys.executable,
        str(REPO / "scripts/evaluate.py"),
        "--luc3d-dir", args.luc3d_dir,
        "--out-csv", args.out_csv,
        "--workers", str(args.workers),
    ]
    print("Running:", " ".join(cmd))
    rc = subprocess.run(cmd, cwd=str(REPO))
    rc.check_returncode()

    # Relabel "luc3d" -> "liezl_cross_view"
    p = Path(args.out_csv)
    df = pd.read_csv(p)
    df.loc[df["tracker"] == "luc3d", "tracker"] = "liezl_cross_view"
    df.to_csv(p, index=False)
    print(f"relabel done in {p}")
    print(df["tracker"].value_counts())


if __name__ == "__main__":
    main()
