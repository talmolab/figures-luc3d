#!/usr/bin/env python
"""Build the shared benchmark inputs for the Bartul 2-animal smoke test:

  outputs/bartul_smoke/det_h5_full/{serial}_predictions.h5   (1, W, n_dets, 15, 2)  canonical slot order
  outputs/bartul_smoke/sleap_h5_full/{serial}_predictions.h5 (1, W, 2, 15, 2)       SLEAP-track slot (for SLEAP ref bar)
  outputs/bartul_smoke/gt/{serial}/proofread.analysis.h5                        (copied; parent dir = serial for evaluate.py)
  outputs/bartul_smoke/bartul_master.tsv                                        one-row master sheet

The canonical detection H5 uses the SAME canonical_order.sort_instances the Liezl
session builder uses, so slot indices line up across Liezl / LUC3D / evaluator.
"""
import os, glob, shutil, sys
import numpy as np
import h5py
import pandas as pd
import sleap_io as sio

sys.path.insert(0, "/root/vast/eric/luc3d-bench/scripts/lib")
from canonical_order import sort_instances, instance_xy

SESSION_DIR = "/root/vast/eric/bartul-data/20250827_141755"
SESSION_ID = "20250827_141755"
OUT = "/root/vast/eric/luc3d-bench/outputs/bartul_smoke"
GT_SRC = os.path.join(OUT, "gt")               # {serial}.gt.analysis.h5 from build_gt_reproj.py
SERIALS = ["21241563", "21369048", "21372315", "21372316", "22085397"]
W_START, W_LEN = 0, 180199
N_NODES = 15
N_DETS = 4   # canonical pool width (NaN-padded); 2 animals but allow spurious extras


def cam_slp(serial):
    return glob.glob(os.path.join(SESSION_DIR, serial, "*.slp"))[0]


def build_canonical_h5(serial):
    """Return (canonical (W,N_DETS,15,2), sleaptrack (W,2,15,2))."""
    labels = sio.load_slp(cam_slp(serial), open_videos=False)
    track_index = {t: i for i, t in enumerate(labels.tracks)}
    canon = np.full((W_LEN, N_DETS, N_NODES, 2), np.nan)
    sleap = np.full((W_LEN, 2, N_NODES, 2), np.nan)
    maxk = 0
    for lf in labels.labeled_frames:
        fi = lf.frame_idx
        if fi < W_START or fi >= W_START + W_LEN:
            continue
        j = fi - W_START
        insts = list(lf.instances)
        # canonical slot order (score desc, centroid tiebreak) — matches Liezl
        for slot, inst in enumerate(sort_instances(insts)):
            if slot >= N_DETS:
                break
            canon[j, slot] = instance_xy(inst)
        maxk = max(maxk, len(insts))
        # SLEAP-track slot (identity as SLEAP assigned it)
        for inst in insts:
            ti = track_index.get(inst.track, None)
            if ti is not None and ti < 2:
                sleap[j, ti] = instance_xy(inst)
    return canon, sleap, maxk


def main():
    det_dir = os.path.join(OUT, "det_h5"); os.makedirs(det_dir, exist_ok=True)
    sleap_dir = os.path.join(OUT, "sleap_h5"); os.makedirs(sleap_dir, exist_ok=True)
    calib = glob.glob(os.path.join(SESSION_DIR, "calibration", "*_calibration.toml"))[0]

    row = {"session": SESSION_ID, "calibration_toml": calib, "num_animals": 2,
           "difficulty": 0, "bedding": "na", "animals": 2}
    for s in SERIALS:
        canon, sleap, maxk = build_canonical_h5(s)
        occ = (~np.isnan(canon).any(-1)).any(-1)  # (W,N_DETS)
        print(f"{s}: max_inst/frame={maxk}  det-occupancy per slot={occ.mean(0).round(3)}")
        with h5py.File(os.path.join(det_dir, f"{s}_predictions.h5"), "w") as f:
            f.create_dataset("tracks", data=canon[None], compression="gzip", compression_opts=1)  # (1,W,N_DETS,15,2)
        with h5py.File(os.path.join(sleap_dir, f"{s}_predictions.h5"), "w") as f:
            f.create_dataset("tracks", data=sleap[None], compression="gzip", compression_opts=1)  # (1,W,2,15,2)

        # GT into per-serial subdir (evaluate.py uses gt_h5.parent.name as the camera name)
        gt_sub = os.path.join(OUT, "gt", s); os.makedirs(gt_sub, exist_ok=True)
        shutil.copy(os.path.join(GT_SRC, f"{s}.gt.analysis.h5"), os.path.join(gt_sub, "proofread.analysis.h5"))

        row[f"{s}_raw_pred_slp"] = cam_slp(s)
        row[f"{s}_proofread_h5"] = os.path.join(gt_sub, "proofread.analysis.h5")

    df = pd.DataFrame([row])
    master = os.path.join(OUT, "bartul_master.tsv")
    df.to_csv(master, sep="\t")   # index col 0 == session_idx 0
    print("wrote", master)
    print("columns:", list(df.columns))


if __name__ == "__main__":
    main()
