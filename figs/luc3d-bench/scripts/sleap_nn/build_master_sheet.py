"""Clone outputs/predictions_master_sheet.tsv with the *_raw_pred_slp columns
repointed to outputs/sleap_nn_retracked/<session>/<cam>.slp.

The rest of the schema (proofread H5s, calibration TOML, num_animals) is
preserved untouched so downstream LUC3D scripts can swap in this sheet by
flag without other code changes.
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
    ap.add_argument("--retrack-dir",
                    default=str(REPO / "outputs" / "sleap_nn_retracked"))
    ap.add_argument("--dst",
                    default=str(REPO / "outputs" / "sleap_nn_master_sheet.tsv"))
    a = ap.parse_args()
    df = pd.read_csv(a.src, sep="\t")
    # Drop any "Unnamed: 0" index column carried over from a prior subset; we
    # re-emit positional indices so the row position matches the H5 session
    # axis 0..N-1 (otherwise luc3d_track_all gets stale session-idx values).
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])
    df = df.reset_index(drop=True)
    retrack_dir = Path(a.retrack_dir)
    for cam in CAMS:
        col = f"{cam}_raw_pred_slp"
        if col not in df.columns:
            continue
        df[col] = [str(retrack_dir / str(s) / f"{cam}.slp")
                   for s in df["session"]]
    df.to_csv(a.dst, sep="\t")
    print(f"wrote {a.dst} with {len(df)} sessions (index reset to 0..{len(df)-1})")


if __name__ == "__main__":
    main()
