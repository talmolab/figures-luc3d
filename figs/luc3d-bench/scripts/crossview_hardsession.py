#!/usr/bin/env python
"""Cross-view identity-consistency metric for the HARDEST SLAP-2M session.

Ports scripts/bartul/cross_view_metric.py to the 4-animal / 6-camera hard
session 10072022145420. The per-camera 2D metric only measures WITHIN-camera
temporal stability; it cannot see whether a tracker keeps ONE identity for an
animal ACROSS the 6 cameras (which is what a multi-view tracker is FOR).

Fix: pool all 6 cameras into a SINGLE MOT sequence.
  - GT object id = the GLOBAL animal id (0..3). Proofread identities in SLAP-2M
    come from the single 3D reconstruction, so track t is the SAME animal in
    every camera.
  - prediction id = the tracker's identity IN ITS NATIVE FORM:
      * multi-view (Liezl)               -> a GLOBAL id, shared across cameras
      * per-camera (SLEAP / ByteTrack / UDMT) -> CAMERA-SCOPED (camera,local),
        so the same animal necessarily gets a different predicted id per view.
IDF1 over this pooled accumulator finds ONE global id map over the whole
6-camera sequence, so a tracker whose identity fragments across views is
penalised. We report per-camera IDF1 (within-view) next to pooled IDF1
(cross-view); the GAP is the cross-view inconsistency.

Only the FOUR trackers the user asked for: SLEAP per-cam 2D (ref), ByteTrack,
UDMT (off-pool, per-camera), Liezl FILTER+TRACK + 3D-weight x6.

Output: outputs/subgroup_7_7_2026/crossview_hardsession_metric.csv
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd
import h5py

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "scripts" / "lib"))
import evaluate as ev            # noqa: E402  (installs np.asfarray shim + motmetrics)
import motmetrics as mm          # noqa: E402

SESSION = "10072022145420"
CAMS = ["back", "backL", "mid", "midL", "top", "topL"]
IOU_THRESH = 0.5

PAF = REPO / "outputs" / "PAF_3d_kalman"
MASTER = REPO / "outputs" / "sleap_nn_master_sheet.tsv"
KEEPTRACK_H5 = REPO / "outputs" / "keeptrack_h5s"            # Liezl 3d6 detection pool
SLEAP_H5 = REPO / "outputs" / "sleap_nn_predictions_h5s"     # stable single-view SLEAP tracks
LIEZL_JSON = PAF / "_liezlsweep" / "G_keeptrack_3d6" / f"{SESSION}.json"
MUPPET_JSON = REPO / "outputs" / "subgroup_7_7_2026" / "muppet_hardsession" / f"{SESSION}.json"
BYTE_DIR = PAF / "detok_bytetrack_results" / SESSION
UDMT_RUNS = Path("/root/vast/eric/UDMT_runs")
UDMT_RESIZE = 0.8


def gt_for_cam(gt_path: str):
    tr, occ = ev.load_gt(Path(gt_path))   # (F, T, N, 2), (F, T)
    return tr, occ


def udmt_boxes(cam: str) -> np.ndarray:
    """(A, F, 4) boxes [x1,y1,x2,y2] in ORIGINAL pixels for one camera."""
    proj = UDMT_RUNS / f"{SESSION}_{cam}-2026-06-30"
    res = proj / "tracking-results"
    vids = [d for d in res.iterdir() if d.is_dir()]
    vid = vids[0].name
    latest_root = res / vid
    subs = [d for d in latest_root.iterdir() if d.is_dir()]
    latest = max(subs, key=lambda d: d.stat().st_mtime)
    files = sorted(latest.glob(f"{vid}_*_new.txt"), key=lambda p: int(p.stem.split("_")[-2]))
    arrs = []
    for f in files:
        a = np.loadtxt(f, delimiter=",") if f.stat().st_size else np.empty((0, 4))
        if a.ndim == 1:
            a = a.reshape(-1, 4)
        arrs.append(a)
    F = max(a.shape[0] for a in arrs)
    out = np.full((len(arrs), F, 4), np.nan)
    for i, a in enumerate(arrs):
        n = a.shape[0]
        xywh = a[:, :4] / UDMT_RESIZE
        out[i, :n, 0] = xywh[:, 0]
        out[i, :n, 1] = xywh[:, 1]
        out[i, :n, 2] = xywh[:, 0] + xywh[:, 2]
        out[i, :n, 3] = xywh[:, 1] + xywh[:, 3]
    return out


def compute(method: str, gt_paths: dict):
    """Return (per_camera_idf1_mean, pooled_idf1, pooled_switches, within_switch_sum)."""
    pooled = mm.MOTAccumulator(auto_id=False)
    percam_idf1 = []
    within_sw = 0

    # preload per-method predictions
    if method in ("liezl", "muppet"):
        luc = json.load(open(LIEZL_JSON if method == "liezl" else MUPPET_JSON))
        det_all = {c: h5py.File(KEEPTRACK_H5 / f"{c}_predictions.h5")["tracks"][70] for c in CAMS}
    elif method == "sleap":
        det_all = {c: h5py.File(SLEAP_H5 / f"{c}_predictions.h5")["tracks"][70] for c in CAMS}
    elif method == "byte":
        det_all = {c: h5py.File(BYTE_DIR / f"{c}.h5")["tracks"][:] for c in CAMS}
    elif method == "udmt":
        det_all = {c: udmt_boxes(c) for c in CAMS}

    for ci, cam in enumerate(CAMS):
        gt, occ = gt_for_cam(gt_paths[cam])
        d = det_all[cam]
        if method in ("liezl", "muppet", "sleap", "byte"):
            nf = min(gt.shape[0], d.shape[0])
        else:  # udmt: d is (A,F,4)
            nf = min(gt.shape[0], d.shape[1])
        if method in ("liezl", "muppet"):
            ndet = d.shape[1]
            ids = ev.luc3d_assignments_for_cam(luc, cam, nf, ndet)
        pc = mm.MOTAccumulator(auto_id=False)
        for fi in range(nf):
            gtb, gti = [], []
            for t in range(gt.shape[1]):
                if not occ[fi, t]:
                    continue
                b = ev.bbox_from_kpts(gt[fi, t])
                if b is not None:
                    gtb.append(b); gti.append(int(t))          # GLOBAL animal id
            prb, pid_local, pid_global = [], [], []
            if method in ("liezl", "muppet", "sleap"):
                for a in range(d.shape[1]):
                    b = ev.bbox_from_kpts(d[fi, a])
                    if b is None:
                        continue
                    if method in ("liezl", "muppet"):
                        g = int(ids[fi, a])
                        if g < 0:
                            continue
                        prb.append(b); pid_local.append(g); pid_global.append(g)          # global
                    else:  # sleap: slot a = stable SLEAP track, camera-scoped
                        prb.append(b); pid_local.append(a); pid_global.append(ci * 100 + a)
            elif method == "byte":
                for a in range(d.shape[1]):
                    row = d[fi, a]
                    if not np.isfinite(row[0]):
                        continue
                    tid = int(row[0])
                    prb.append(row[1:5]); pid_local.append(tid); pid_global.append(ci * 100000 + tid)
            else:  # udmt: per-camera track a, camera-scoped
                for a in range(d.shape[0]):
                    box = d[a, fi]
                    if np.all(np.isfinite(box)) and box[2] > box[0] and box[3] > box[1]:
                        prb.append(box); pid_local.append(a); pid_global.append(ci * 100000 + a)
            gtn = np.array(gtb) if gtb else np.empty((0, 4))
            prn = np.array(prb) if prb else np.empty((0, 4))
            dist = mm.distances.iou_matrix(gtn, prn, max_iou=1.0 - IOU_THRESH)
            pc.update(gti, pid_local, dist, frameid=fi)
            pooled.update(gti, pid_global, dist, frameid=ci * 10000000 + fi)
        s = mm.metrics.create().compute(pc, metrics=["idf1", "num_switches"], name="c")
        percam_idf1.append(float(s["idf1"]["c"]))
        within_sw += int(s["num_switches"]["c"])
        print(f"    {method:6s} {cam:5s} within-IDF1={percam_idf1[-1]:.3f}")
    sp = mm.metrics.create().compute(pooled, metrics=["idf1", "num_switches"], name="p")
    return float(np.mean(percam_idf1)), float(sp["idf1"]["p"]), int(sp["num_switches"]["p"]), within_sw


def main():
    df = pd.read_csv(MASTER, sep="\t")
    r = df[df.session.astype(str) == SESSION].iloc[0]
    gt_paths = {c: r[f"{c}_proofread_h5"] for c in CAMS}

    order = [
        ("Liezl — FILTER+TRACK + 3D-weight×6", "liezl"),
        ("3D-MuPPET (SORT+xview)",             "muppet"),
        ("UDMT (off-pool, per-camera)",        "udmt"),
        ("SLEAP per-camera 2D (ref)",          "sleap"),
        ("ByteTrack",                          "byte"),
    ]
    # Optional: restrict to a subset of methods (comma list) and MERGE into the
    # existing CSV, so we don't recompute the slow ones. e.g. `... muppet`.
    only = set(sys.argv[1].split(",")) if len(sys.argv) > 1 else None
    out = REPO / "outputs" / "subgroup_7_7_2026" / "crossview_hardsession_metric.csv"
    prev = pd.read_csv(out) if (only and out.exists()) else None

    rows = []
    for label, method in order:
        if only and method not in only:
            if prev is not None:
                r = prev[prev.method == method]
                if len(r):
                    rows.append(r.iloc[0].to_dict())
            continue
        print(f"  {label}")
        pcam, pool_idf1, pool_sw, within_sw = compute(method, gt_paths)
        rows.append(dict(tracker=label, method=method, per_camera_idf1=pcam,
                         crossview_idf1=pool_idf1, crossview_switches=pool_sw,
                         within_switches_sum=within_sw))
        print(f"  => {label:38s} within={pcam:.3f}  CROSS-VIEW={pool_idf1:.3f}  2Dsw={within_sw}")
    # preserve the canonical order
    om = {m: i for i, (_, m) in enumerate(order)}
    rows.sort(key=lambda d: om.get(d["method"], 99))
    pd.DataFrame(rows).to_csv(out, index=False)
    print("wrote", out)


if __name__ == "__main__":
    main()
