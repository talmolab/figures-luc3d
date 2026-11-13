"""Clone outputs/predictions_master_sheet.tsv with the *_raw_pred_slp columns
repointed to the FILTER-ONLY detections at outputs/detections_only/<session>/<cam>.slp.

Everything else (proofread H5s, calibration TOML, difficulty, bedding,
num_animals) is preserved so Liezl / LUC3D / eval can swap in this sheet by flag.
The positional index is reset to 0..N-1 so the row position matches the H5
session axis (handoff §11).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
CAMS = ["back", "backL", "mid", "midL", "top", "topL"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src",
                    default=str(REPO / "outputs" / "predictions_master_sheet.tsv"))
    ap.add_argument("--detections-dir",
                    default=str(REPO / "outputs" / "detections_only"))
    ap.add_argument("--dst",
                    default=str(REPO / "outputs" / "detections_only_master_sheet.tsv"))
    a = ap.parse_args()
    df = pd.read_csv(a.src, sep="\t")
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])
    df = df.reset_index(drop=True)
    det_dir = Path(a.detections_dir)
    missing = 0
    for cam in CAMS:
        col = f"{cam}_raw_pred_slp"
        if col not in df.columns:
            continue
        paths = [str(det_dir / str(s) / f"{cam}.slp") for s in df["session"]]
        df[col] = paths
        missing += sum(0 if Path(p).exists() else 1 for p in paths)
    df.to_csv(a.dst, sep="\t")
    print(f"wrote {a.dst} with {len(df)} sessions (index reset to 0..{len(df)-1})")
    if missing:
        print(f"  WARNING: {missing} repointed detection .slp paths do not exist yet")


if __name__ == "__main__":
    main()
