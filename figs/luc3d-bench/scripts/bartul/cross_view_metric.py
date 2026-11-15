#!/usr/bin/env python
"""Cross-view identity-consistency metric for the Bartul benchmark.

The existing metric scores each camera INDEPENDENTLY (per-camera 2D MOT with a
per-camera optimal id remap), so it only measures WITHIN-camera temporal
stability — it cannot see whether a tracker keeps ONE identity for an animal
ACROSS cameras.

Here we pool all 5 cameras into a SINGLE MOT sequence:
  - GT object id = the GLOBAL animal id (0/1); the same animal is one object in
    every camera (GT comes from the single 3D reconstruction).
  - prediction id = the tracker's identity IN ITS NATIVE FORM:
      * multi-view trackers (Liezl, LUC3D) emit a GLOBAL id  -> shared across cams
      * per-camera trackers (SLEAP, ByteTrack) emit CAMERA-SCOPED ids -> we tag
        them (camera, local_id), so the same animal necessarily gets a different
        predicted id in each view.
IDF1 over this pooled accumulator finds ONE global id mapping over the whole
multi-camera sequence, so a tracker whose identity fragments across views is
penalised. We report per-camera IDF1 (within-view) next to pooled IDF1
(cross-view) — the GAP is the cross-view inconsistency.
"""
import sys, json
from pathlib import Path
import numpy as np

REPO = Path("/root/vast/eric/luc3d-bench")
sys.path.insert(0, str(REPO / "scripts"))
import evaluate as ev
import motmetrics as mm

OUT = REPO / "outputs/bartul_smoke"
SESSION = "20250827_141755"
SER = ["21241563","21369048","21372315","21372316","22085397"]
DET = OUT/"det_h5"; SLEAP = OUT/"sleap_h5"; GT = OUT/"gt"; RES = OUT/"results"
W = 9000

# label -> (kind, path). kind: luc3d(global id) | sleap(cam-scoped) | bytetrack(cam-scoped)
TRACKERS = {
    "Liezl — 3D-weight×6":       ("luc3d",    RES/"liezl_c6"),
    "Liezl (3D)":                ("luc3d",    RES/"liezl_c1"),
    "3D-MuPPET (SORT+xview)":    ("luc3d",    RES/"muppet"),
    "LUC3D (3D)":                ("luc3d",    RES/"luc3d"),
    "SLEAP per-camera 2D (ref)": ("sleap",    SLEAP),
    "ByteTrack":                 ("bytetrack",RES/"bytetrack"),
}
import h5py


def per_cam_and_pooled(kind, path):
    """Return (per_camera_idf1_mean, pooled_idf1, pooled_switches)."""
    percam = mm.MOTAccumulator(auto_id=False)  # not used for per-cam value; we compute per-cam separately
    pooled = mm.MOTAccumulator(auto_id=False)
    percam_accs = []
    if kind == "luc3d":
        luc = json.load(open(Path(path)/f"{SESSION}.json"))
    for ci, cam in enumerate(SER):
        gt = np.transpose(h5py.File(GT/cam/"proofread.analysis.h5")["tracks"][:], (3,0,2,1))[:W]
        pc = mm.MOTAccumulator(auto_id=False)
        if kind == "luc3d":
            det = h5py.File(DET/f"{cam}_predictions.h5")["tracks"][0][:W]
            ids = ev.luc3d_assignments_for_cam(luc, cam, W, det.shape[1])  # (W, n_dets) global id or -1
        elif kind == "sleap":
            det = h5py.File(SLEAP/f"{cam}_predictions.h5")["tracks"][0][:W]  # slot=SLEAP track
        elif kind == "bytetrack":
            bt = h5py.File(Path(path)/SESSION/f"{cam}.h5")["tracks"][:W]      # (W, A, 6)
        nf = min(W, gt.shape[0])
        for fi in range(nf):
            gtb, gti = [], []
            for t in range(gt.shape[1]):
                b = ev.bbox_from_kpts(gt[fi, t])
                if b is not None:
                    gtb.append(b); gti.append(int(t))          # GLOBAL animal id
            prb, pid_local, pid_global = [], [], []
            if kind in ("luc3d","sleap"):
                for a in range(det.shape[1]):
                    b = ev.bbox_from_kpts(det[fi, a])
                    if b is None: continue
                    if kind == "luc3d":
                        g = int(ids[fi, a])
                        if g < 0: continue
                        prb.append(b); pid_local.append(g); pid_global.append(g)      # global id shared across cams
                    else:  # sleap: slot a is the SLEAP track; camera-scoped
                        prb.append(b); pid_local.append(a); pid_global.append(ci*10 + a)
            else:  # bytetrack
                for a in range(bt.shape[1]):
                    row = bt[fi, a]
                    if not np.isfinite(row[0]): continue
                    prb.append(row[1:5]); pid_local.append(int(row[0])); pid_global.append(ci*100000 + int(row[0]))
            gtn = np.array(gtb) if gtb else np.empty((0,4))
            prn = np.array(prb) if prb else np.empty((0,4))
            d = mm.distances.iou_matrix(gtn, prn, max_iou=0.5)
            pc.update(gti, pid_local, d, frameid=fi)                       # per-camera (local ids)
            pooled.update(gti, pid_global, d, frameid=ci*1000000 + fi)      # pooled (global/cam-scoped ids)
        s = mm.metrics.create().compute(pc, metrics=["idf1"], name="c")
        percam_accs.append(s["idf1"]["c"])
    sp = mm.metrics.create().compute(pooled, metrics=["idf1","num_switches","idp","idr"], name="p")
    return float(np.mean(percam_accs)), float(sp["idf1"]["p"]), int(sp["num_switches"]["p"])


def main():
    rows = []
    for label,(kind,path) in TRACKERS.items():
        pcam, pool_idf1, pool_sw = per_cam_and_pooled(kind, path)
        rows.append({"tracker":label,"kind":kind,"per_camera_idf1":pcam,
                     "crossview_idf1":pool_idf1,"crossview_switches":pool_sw})
        print(f"  {label:28s} per-cam IDF1={pcam:.3f}   CROSS-VIEW IDF1={pool_idf1:.3f}  xview-sw={pool_sw}")
    import pandas as pd
    df = pd.DataFrame(rows)
    df.to_csv(OUT/"bartul_crossview_metric.csv", index=False)
    print("wrote", OUT/"bartul_crossview_metric.csv")


if __name__ == "__main__":
    main()
