#!/usr/bin/env python
"""Evaluate every tracker on the Bartul session vs the reprojection GT.

Reuses scripts/evaluate.py:eval_camera (motmetrics IDF1 / raw switches) but with
the 5 Bartul serial cameras and per-tracker result dirs. Writes one combined
per-camera CSV that the figure script consumes.

Trackers (label -> spec):
  kind="luc3d"    : a LUC3D-schema JSON dir  {session}.json   (Liezl variants, LUC3D)
  kind="sleap"    : SLEAP-track-slot H5 dir  {cam}_predictions.h5
  kind="bytetrack": bytetrack results dir    {session}/{cam}.h5
"""
import os, sys, json
from pathlib import Path
import numpy as np
import pandas as pd

REPO = Path("/root/vast/eric/luc3d-bench")
sys.path.insert(0, str(REPO / "scripts"))
import evaluate as ev

SERIALS = ["21241563","21369048","21372315","21372316","22085397"]
ev.BENCH_CAMERAS = tuple(SERIALS)   # override module global used by eval loop

OUT = REPO / "outputs/bartul_smoke"
SESSION = "20250827_141755"
DET_DIR = OUT / "det_h5"            # canonical detection pool (Liezl/LUC3D/ByteTrack bboxes)
SLEAP_DIR = OUT / "sleap_h5"        # SLEAP-track-slot pool (SLEAP ref bar)
GT_DIR = OUT / "gt"                 # {serial}/proofread.analysis.h5
RES = OUT / "results"

# label -> (kind, path)
TRACKERS = {
    "Liezl (3D)":                 ("luc3d",    RES/"liezl_c1"),
    "Liezl — 3D-weight×3":        ("luc3d",    RES/"liezl_c3"),
    "Liezl — 3D-weight×6":        ("luc3d",    RES/"liezl_c6"),
    "Liezl ×6 + 3D swap-fix":     ("luc3d",    RES/"swapdp_c6"),
    "Liezl ×6 + appearance ReID": ("luc3d",    RES/"reid_c6"),
    "3D-MuPPET (SORT+xview)":     ("luc3d",    RES/"muppet"),
    "LUC3D (3D)":                 ("luc3d",    RES/"luc3d"),
    "ByteTrack":                  ("bytetrack",RES/"bytetrack"),
    "SLEAP per-camera 2D (ref)":  ("sleap",    SLEAP_DIR),
    "UDMT (off-pool, own det.)":  ("luc3d",    RES/"udmt"),
}


def eval_tracker(label, kind, path):
    rows = []
    for cam in SERIALS:
        gt_h5 = GT_DIR / cam / "proofread.analysis.h5"
        if kind == "luc3d":
            js = Path(path) / f"{SESSION}.json"
            if not js.exists():
                return None
            m = ev.eval_camera(DET_DIR/f"{cam}_predictions.h5", gt_h5, js, None,
                               session_idx=0, no_sleap=True)
            row = m["luc3d"]
        elif kind == "sleap":
            # any json works (sleap row is slot-based, json ignored for it); use a present one
            js = RES/"liezl_c1"/f"{SESSION}.json"
            m = ev.eval_camera(Path(path)/f"{cam}_predictions.h5", gt_h5, js, None,
                               session_idx=0, no_sleap=False)
            row = m["sleap"]
        elif kind == "bytetrack":
            bt = Path(path)/SESSION/f"{cam}.h5"
            if not bt.exists():
                return None
            js = RES/"liezl_c1"/f"{SESSION}.json"
            m = ev.eval_camera(DET_DIR/f"{cam}_predictions.h5", gt_h5, js, bt,
                               session_idx=0, no_sleap=True)
            row = m["bytetrack"]
        else:
            raise ValueError(kind)
        rows.append({"tracker": label, "camera": cam, "idf1": row["idf1"],
                     "idp": row["idp"], "idr": row["idr"], "mota": row["mota"],
                     "num_switches": row["num_switches"], "recall": row["recall"],
                     "num_frames": row["num_frames"]})
    return rows


def main():
    all_rows = []
    for label, (kind, path) in TRACKERS.items():
        r = eval_tracker(label, kind, path)
        if r is None:
            print(f"  SKIP (missing): {label}  [{path}]")
            continue
        all_rows.extend(r)
        df = pd.DataFrame(r)
        print(f"  {label:32s} IDF1={df.idf1.mean():.3f}  switches(sum)={int(df.num_switches.sum())}")
    out = pd.DataFrame(all_rows)
    out_csv = OUT / "bartul_eval_per_camera.csv"
    out.to_csv(out_csv, index=False)
    print("wrote", out_csv, f"({len(out)} rows)")


if __name__ == "__main__":
    main()
