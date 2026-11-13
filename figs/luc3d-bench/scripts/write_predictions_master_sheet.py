"""Produce predictions_master_sheet.tsv from the original master_sheet.tsv.

For each row, adds columns:
  - `{cam}_raw_pred_slp`: absolute path to non-proofread `.predictions.slp`
  - `{cam}_proofread_h5`: absolute path to `.predictions.proofread.slp.analysis.h5`
  - `calibration_toml`:    absolute path to per-session calibration.toml
  - `num_animals`:         white_mice + agouti_mice + black_mice

The aggregated camera H5s are looked up positionally by row index.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from lib.slap2m_paths import BENCH_CAMERAS, calibration_toml, proofread_analysis_h5, raw_pred_slp

MASTER = Path("/root/vast/eric/slap-2m-h5s/master_sheet.tsv")
OUT = Path("/root/vast/eric/luc3d-bench/outputs/predictions_master_sheet.tsv")


def main():
    df = pd.read_csv(MASTER, sep="\t", index_col=0)
    df["num_animals"] = (df["white_mice"] + df["agouti_mice"] + df["black_mice"]).astype(int)
    df["calibration_toml"] = df["session_path"].apply(lambda s: str(calibration_toml(s)))
    for cam in BENCH_CAMERAS:
        df[f"{cam}_raw_pred_slp"] = df["session_path"].apply(lambda s, c=cam: (lambda p: str(p) if p else "")(raw_pred_slp(s, c)))
        df[f"{cam}_proofread_h5"] = df["session_path"].apply(lambda s, c=cam: (lambda p: str(p) if p else "")(proofread_analysis_h5(s, c)))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, sep="\t")
    print(f"wrote {OUT} ({len(df)} rows, {len(df.columns)} cols)")
    missing_raw = {c: (df[f"{c}_raw_pred_slp"] == "").sum() for c in BENCH_CAMERAS}
    missing_pf = {c: (df[f"{c}_proofread_h5"] == "").sum() for c in BENCH_CAMERAS}
    print("missing raw_pred_slp per cam:", missing_raw)
    print("missing proofread_h5 per cam:", missing_pf)


if __name__ == "__main__":
    main()
