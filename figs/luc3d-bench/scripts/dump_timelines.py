#!/usr/bin/env python3
"""One parallel pass: dump per-session GT->predicted-id timelines + 3D covariates.

For each multi-animal session and camera we compute, per frame, the predicted Liezl
identity matched (IoU>=0.5) to each GT track (or -1 if that GT had no detection in
that camera that frame -> a DROPOUT). Plus each GT track's session-wide CORRECT id and
the per-frame 3D inter-animal distance (points3d GT). Saved to one npz per session so
the error-classification + video selection can iterate without the ~13-min recompute.

Output: outputs/PAF_3d_kalman/_diag/timelines/<session>.npz with
  matched : (n_cams, n_frames, n_tracks) int  (predicted id or -1=no detection)
  cams    : list[str]
  correct : (n_tracks,) int   session-wide modal id per GT track
  inter3d : (n_frames,) float  min pairwise inter-animal 3D distance (mm)
  difficulty, animals : int
"""
import argparse
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from scripts.evaluate import load_gt, bbox_from_kpts, luc3d_assignments_for_cam

REPO = Path(__file__).resolve().parents[1]
MASTER = REPO / "outputs" / "detections_only_master_sheet.tsv"
SLAP2M = Path("/root/talmolab-smb/eric/slap_2m")
CAMS = ["back", "backL", "mid", "midL", "top", "topL"]
OUT = REPO / "outputs/PAF_3d_kalman/_diag/timelines"


def iou(a, b):
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1]); ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0, ix1 - ix0), max(0, iy1 - iy0); inter = iw * ih
    ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def one_session(args):
    sid, h5idx, results_dir, pred_dir, gtpaths, p3d_path, difficulty, animals = args
    jd = json.load(open(f"{results_dir}/{sid}.json"))
    cams = [c for c in CAMS if c in gtpaths]
    # frame count from first cam GT
    matched_by_cam = []
    nt = None; nframes = None
    for cam in cams:
        with h5py.File(f"{pred_dir}/{cam}_predictions.h5", "r") as f:
            raw = f["tracks"][h5idx]
        gt, occ = load_gt(gtpaths[cam])
        n = min(raw.shape[0], gt.shape[0]); na = raw.shape[1]; nt = gt.shape[1]
        nframes = n if nframes is None else min(nframes, n)
        ids = luc3d_assignments_for_cam(jd, cam, n, na)
        m = np.full((n, nt), -1, np.int64)
        for fi in range(n):
            gb = {}
            for t in range(nt):
                if occ[fi, t]:
                    b = bbox_from_kpts(gt[fi, t])
                    if b is not None:
                        gb[t] = b
            pb = []
            for s in range(na):
                b = bbox_from_kpts(raw[fi, s])
                if b is None:
                    continue
                iv = int(ids[fi, s])
                if iv >= 0:
                    pb.append((b, iv))
            pairs = sorted(((iou(gb[t], pbx), t, iv) for t in gb for pbx, iv in pb), reverse=True)
            ut = set()
            for io, t, iv in pairs:
                if io < 0.5 or t in ut:
                    continue
                m[fi, t] = iv; ut.add(t)
        matched_by_cam.append(m)
    nf = min(m.shape[0] for m in matched_by_cam)
    matched = np.stack([m[:nf] for m in matched_by_cam], axis=0)  # (n_cams, nf, nt)
    # correct id per track (session-wide modal over all cams)
    correct = np.full(nt, -1, np.int64)
    for t in range(nt):
        vals = matched[:, :, t].ravel()
        vals = vals[vals >= 0]
        if len(vals):
            correct[t] = np.bincount(vals).argmax()
    # 3D inter-animal distance
    inter3d = np.full(nf, np.nan)
    if p3d_path and Path(p3d_path).exists():
        with h5py.File(p3d_path, "r") as f:
            P = f["tracks"][:]
        for fi in range(min(nf, P.shape[0])):
            cents = [np.nanmean(P[fi, t], axis=0) for t in range(P.shape[1])]
            cents = [c for c in cents if np.isfinite(c).all()]
            if len(cents) >= 2:
                inter3d[fi] = min(np.linalg.norm(cents[i] - cents[j])
                                  for i in range(len(cents)) for j in range(i + 1, len(cents)))
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / f"{sid}.npz", matched=matched, cams=np.array(cams),
                        correct=correct, inter3d=inter3d,
                        difficulty=difficulty, animals=animals)
    return sid, matched.shape


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", required=True)
    ap.add_argument("--pred-h5-dir", required=True)
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()
    ms = pd.read_csv(MASTER, sep="\t"); ms["session"] = ms["session"].astype(str)
    multi = ms[ms["num_animals"] >= 2]
    tasks = []
    for _, row in multi.iterrows():
        sid = row["session"]
        if not Path(f"{a.results_dir}/{sid}.json").exists():
            continue
        h5idx = int(np.where(ms["session"].values == sid)[0][0])
        gtpaths = {c: row[f"{c}_proofread_h5"] for c in CAMS
                   if isinstance(row[f"{c}_proofread_h5"], str) and row[f"{c}_proofread_h5"]}
        p3d = str(SLAP2M / row["points_3D"]) if isinstance(row["points_3D"], str) else None
        tasks.append((sid, h5idx, a.results_dir, a.pred_h5_dir, gtpaths, p3d,
                      int(row["difficulty"]), int(row["num_animals"])))
    done = 0
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for fut in as_completed([ex.submit(one_session, t) for t in tasks]):
            sid, sh = fut.result(); done += 1
            print(f"[{done}/{len(tasks)}] {sid} {sh}")
    print("wrote timelines to", OUT)


if __name__ == "__main__":
    main()
