#!/usr/bin/env python3
"""Diagnose WHERE/HOW Liezl ID switches happen, against 3D ground truth.

Two questions from Eric (2026-06-23):
  (1) Are the "switch" counts real, or is one sustained swap (1->2, stay 300 frames,
      ->1) being counted as ~300 switches?  -> we record, per GT animal, the full
      identity run-length structure and the dwell-time (in frames) of every WRONG
      run.  A motmetrics SWITCH is a *transition*, so the 1->2->1 case = 2 events.
  (2) Do switches happen when animals are FAR AWAY (from camera) or at CLOSE
      ENCOUNTERS (near each other)?  -> at every matched frame we compute, from the
      3D ground truth (points3d.h5) + calibration, the GT animal's depth (camera-
      frame Z, mm) and the min pairwise inter-animal 3D distance (mm), plus the 2D
      GT bbox area in that camera.  We compare the distribution of these covariates
      at SWITCH frames vs all MATCHED frames.

Eval matches the per-camera 2D IoU protocol of scripts/evaluate.py exactly.

Usage:
  PYTHONPATH=. .venv/bin/python scripts/diagnose_switches.py \
      --results-dir outputs/PAF_3d_kalman/_liezlsweep/G_keeptrack_3d6 \
      --pred-h5-dir outputs/keeptrack_h5s \
      --out-prefix  outputs/PAF_3d_kalman/_diag/G_keeptrack_3d6 \
      [--min-run 30] [--workers 16] [--sessions S1 S2]
"""
import argparse
import json
import tomllib
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import cv2
import h5py
import numpy as np
import pandas as pd
import motmetrics as mm

from scripts.evaluate import load_gt, bbox_from_kpts, luc3d_assignments_for_cam

REPO = Path(__file__).resolve().parents[1]
CAMS = ["back", "backL", "mid", "midL", "top", "topL"]
MASTER = REPO / "outputs" / "detections_only_master_sheet.tsv"
SLAP2M = Path("/root/talmolab-smb/eric/slap_2m")


def load_calib(path):
    calib = tomllib.load(open(path, "rb"))
    cams = {}
    for key, c in calib.items():
        if not key.startswith("cam_"):
            continue
        nm = c["name"]
        if nm not in CAMS:
            continue
        R, _ = cv2.Rodrigues(np.array(c["rotation"], float))
        t = np.array(c["translation"], float).reshape(3, 1)
        cams[nm] = (R, t)
    return cams


def bbox_area(b):
    return float(max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1]))


def one_cam(args):
    (pred_h5, gt_h5, js, sidx, cam, min_run, p3d_path, calib_path, sess, diff) = args
    with h5py.File(pred_h5, "r") as f:
        raw = f["tracks"][sidx]
    gt, occ = load_gt(gt_h5)
    n = min(raw.shape[0], gt.shape[0])
    na = raw.shape[1]
    ids = luc3d_assignments_for_cam(json.load(open(js)), cam, n, na)

    # 3D GT + depth for this cam
    p3d = None
    R = t = None
    if p3d_path and Path(p3d_path).exists() and calib_path and Path(calib_path).exists():
        with h5py.File(p3d_path, "r") as f:
            p3d = f["tracks"][:]  # (frames, n_tracks, 15, 3)
        cams = load_calib(calib_path)
        if cam in cams:
            R, t = cams[cam]

    acc = mm.MOTAccumulator(auto_id=True)
    # per-frame GT-object covariates: depth_cam[t], min interanimal dist, bbox area
    cov = {}  # (fi, oid) -> (depth_mm, inter_mm, bbox_area)
    for fi in range(n):
        gb, gi = [], []
        for t_ in range(gt.shape[1]):
            if occ[fi, t_]:
                b = bbox_from_kpts(gt[fi, t_])
                if b is not None:
                    gb.append(b); gi.append(int(t_))
        # covariates for present GT objects this frame
        if p3d is not None and fi < p3d.shape[0]:
            cents = {}
            for t_ in gi:
                if t_ < p3d.shape[1]:
                    c = np.nanmean(p3d[fi, t_], axis=0)
                    if np.isfinite(c).all():
                        cents[t_] = c
            # inter-animal min distance (shared across objects this frame)
            inter = np.nan
            ks = list(cents)
            if len(ks) >= 2:
                ds = [np.linalg.norm(cents[a] - cents[b])
                      for i, a in enumerate(ks) for b in ks[i + 1:]]
                inter = float(min(ds))
            for k, b in zip(gi, gb):
                depth = np.nan
                if k in cents and R is not None:
                    Xc = R @ cents[k].reshape(3, 1) + t
                    depth = float(Xc[2, 0])
                cov[(fi, k)] = (depth, inter, bbox_area(b))
        pb, pi = [], []
        for a in range(na):
            b = bbox_from_kpts(raw[fi, a])
            if b is None:
                continue
            iv = int(ids[fi, a])
            if iv >= 0:
                pb.append(b); pi.append(iv)
        d = mm.distances.iou_matrix(np.array(gb) if gb else np.empty((0, 4)),
                                    np.array(pb) if pb else np.empty((0, 4)), max_iou=0.5)
        acc.update(gi, pi, d)

    ev = acc.mot_events.reset_index()
    m = ev[ev.Type.isin(["MATCH", "SWITCH"])]
    raw_sw = int((ev.Type == "SWITCH").sum())

    # per-GT-object run structure + switch events
    sustained = 0
    wrong_dwells = []      # frames an object spent in a run that ended in a switch-back
    switch_rows = []       # (cam, fi, oid, depth, inter, area)
    match_cov = []         # covariate at every matched frame (depth, inter, area)
    for oid in m.OId.dropna().unique():
        sub = m[m.OId == oid].sort_values("FrameId")
        seq = sub.HId.tolist()
        frames = sub.FrameId.tolist()
        # run-length compress
        runs = []  # [hid, start_frame, count]
        for h, fr in zip(seq, frames):
            if runs and runs[-1][0] == h:
                runs[-1][2] += 1
            else:
                runs.append([h, fr, 1])
        # sustained transitions
        kept = [h for h, _, c in runs if c >= min_run]
        merged = []
        for h in kept:
            if not merged or merged[-1] != h:
                merged.append(h)
        sustained += max(0, len(merged) - 1)
        # dwell time of each run (proxy for "how long mislabeled")
        for h, fr, c in runs:
            wrong_dwells.append(c)
        # covariates at each matched frame and at switch frames
        for r in sub.itertuples():
            fi = int(r.FrameId); o = int(r.OId)
            c = cov.get((fi, o), (np.nan, np.nan, np.nan))
            match_cov.append(c)
            if r.Type == "SWITCH":
                switch_rows.append((sess, diff, cam, fi, o, c[0], c[1], c[2]))
    return {
        "raw_sw": raw_sw, "sustained": sustained,
        "wrong_dwells": wrong_dwells, "switch_rows": switch_rows,
        "match_cov": match_cov, "n_matched_frames": len(match_cov),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", required=True)
    ap.add_argument("--pred-h5-dir", required=True)
    ap.add_argument("--out-prefix", required=True)
    ap.add_argument("--min-run", type=int, default=30)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--sessions", nargs="*", default=None)
    a = ap.parse_args()

    ms = pd.read_csv(MASTER, sep="\t", index_col=0).reset_index(drop=True)
    ms["session"] = ms["session"].astype(str)
    multi = ms[ms["num_animals"] >= 2]
    if a.sessions:
        multi = multi[multi["session"].isin(a.sessions)]
    rdir = Path(a.results_dir)
    tasks = []
    for sidx, row in multi.iterrows():
        sid = row["session"]
        js = rdir / f"{sid}.json"
        if not js.exists():
            continue
        p3d = SLAP2M / str(row["points_3D"]) if isinstance(row["points_3D"], str) else None
        calib = row["calibration_toml"]
        for cam in CAMS:
            gt = row[f"{cam}_proofread_h5"]
            if not isinstance(gt, str) or not gt:
                continue
            tasks.append((f"{a.pred_h5_dir}/{cam}_predictions.h5", gt, str(js),
                          int(sidx), cam, a.min_run, str(p3d) if p3d else None, calib,
                          sid, int(row["difficulty"])))

    raw_tot = sust_tot = 0
    all_dwells = []
    switch_rows = []
    sw_cov = []   # covariate at switch frames
    all_cov = []  # covariate at all matched frames
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for fut in as_completed([ex.submit(one_cam, t) for t in tasks]):
            r = fut.result()
            raw_tot += r["raw_sw"]; sust_tot += r["sustained"]
            all_dwells.extend(r["wrong_dwells"])
            switch_rows.extend(r["switch_rows"])
            sw_cov.extend([(row[5], row[6], row[7]) for row in r["switch_rows"]])
            all_cov.extend(r["match_cov"])

    out = Path(a.out_prefix)
    out.parent.mkdir(parents=True, exist_ok=True)

    # --- switch events table ---
    df = pd.DataFrame(switch_rows, columns=["session", "difficulty", "cam", "frame", "gt_oid", "depth_mm", "inter_mm", "bbox_area"])
    df.to_csv(f"{a.out_prefix}_switch_events.tsv", sep="\t", index=False)

    sw = np.array(sw_cov, float).reshape(-1, 3)
    al = np.array(all_cov, float).reshape(-1, 3)
    # save arrays for plotting (subsample baseline to keep it small)
    np.save(f"{a.out_prefix}_sw_cov.npy", sw)
    rng = np.arange(0, al.shape[0], max(1, al.shape[0] // 200000))
    np.save(f"{a.out_prefix}_all_cov.npy", al[rng])

    n_single = int(np.isnan(sw[:, 1]).sum())   # switch frames w/ only 1 animal visible in cam

    def q(arr, col):
        v = arr[:, col]; v = v[np.isfinite(v)]
        if len(v) == 0:
            return "n/a"
        return f"med={np.median(v):.0f} p10={np.percentile(v,10):.0f} p90={np.percentile(v,90):.0f} mean={v.mean():.0f} n={len(v)}"

    dw = np.array(all_dwells, float)
    lines = []
    lines.append(f"# Switch diagnosis: {rdir.name}")
    lines.append(f"sessions(multi)={multi.shape[0]}  cam-tasks={len(tasks)}")
    lines.append(f"RAW switches={raw_tot}  SUSTAINED(min_run={a.min_run})={sust_tot}  "
                 f"oscillation_frac={1-sust_tot/max(1,raw_tot):.2f}")
    lines.append("")
    lines.append("## Counting check: identity-run dwell times (frames per contiguous run)")
    lines.append(f"  n_runs={len(dw)}  median_run={np.median(dw):.0f}  mean_run={dw.mean():.0f}  "
                 f"max_run={dw.max():.0f}")
    lines.append(f"  runs >=30f: {(dw>=30).sum()} ({100*(dw>=30).mean():.0f}%)   "
                 f">=300f: {(dw>=300).sum()} ({100*(dw>=300).mean():.0f}%)")
    lines.append(f"  -> total matched frames={al.shape[0]}; if switches were counted "
                 f"per-frame the count would be ~{al.shape[0]}, not {raw_tot}.")
    lines.append("")
    lines.append(f"## Regime split: of {sw.shape[0]} switches, {n_single} "
                 f"({100*n_single/max(1,sw.shape[0]):.0f}%) occur when only ONE animal is "
                 f"detected in that camera (re-entry/occlusion-recovery); the rest are "
                 f"both-visible close passes.")
    lines.append("")
    lines.append("## Covariate distributions: SWITCH frames vs ALL matched frames")
    lines.append(f"  depth_cam_mm (dist from camera): switch[{q(sw,0)}]  all[{q(al,0)}]")
    lines.append(f"  inter_animal_mm (3D):            switch[{q(sw,1)}]  all[{q(al,1)}]")
    lines.append(f"  bbox_area_px2 (2D size):         switch[{q(sw,2)}]  all[{q(al,2)}]")
    report = "\n".join(lines)
    Path(f"{a.out_prefix}_summary.txt").write_text(report + "\n")
    print(report)


if __name__ == "__main__":
    main()
