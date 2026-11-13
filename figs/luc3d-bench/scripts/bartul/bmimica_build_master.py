#!/usr/bin/env python
"""Stage 3a: build the 50-session master sheet for Liezl + eval.
Points {serial}_raw_pred_slp -> retracked (filter+track) .slp, {serial}_proofread_h5 -> GT.
Paths are deterministic, so this can run before retrack finishes."""
import os, glob
import pandas as pd

ROOT = "/root/vast/eric/BMimica"
BM = "/root/vast/eric/luc3d-bench/outputs/bmimica"
SER = ["21241563", "21369048", "21372315", "21372316", "22085397"]


def runnable():
    out = []
    for s in sorted(glob.glob(ROOT + "/*")):
        sid = os.path.basename(s)
        if os.path.isdir(s) and sid[0].isdigit() and \
           glob.glob(s+"/calibration/*_calibration.toml") and glob.glob(s+"/*points3d*.h5") and \
           sum(1 for c in SER if glob.glob(f"{s}/{c}/*.slp")) == 5:
            out.append(sid)
    return out


def main():
    rows = []
    for sid in runnable():
        calib = glob.glob(f"{ROOT}/{sid}/calibration/*_calibration.toml")[0]
        row = {"session": sid, "calibration_toml": calib, "num_animals": 2,
               "difficulty": 0, "bedding": "na", "animals": 2}
        for c in SER:
            row[f"{c}_raw_pred_slp"] = f"{BM}/retracked/{sid}/{c}.slp"
            row[f"{c}_proofread_h5"] = f"{BM}/gt/{sid}/{c}/proofread.analysis.h5"
        rows.append(row)
    df = pd.DataFrame(rows)
    out = f"{BM}/bmimica_master.tsv"
    df.to_csv(out, sep="\t")
    print(f"wrote {out}: {len(df)} sessions, {len(df.columns)} cols")


if __name__ == "__main__":
    main()
