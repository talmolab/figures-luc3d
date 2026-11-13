#!/usr/bin/env python3
"""Decompose ID switches into RAW vs SUSTAINED for each tracker (multi-animal only).

Motivation (Eric, 2026-06-22): the raw motmetrics switch count treats a single
sustained swap (tracker locks the wrong identity for thousands of frames) and a
frame-by-frame oscillation identically per-transition. A tracker that swaps once
and stays is far more "accurate" than one that flickers, yet both can show big raw
counts. So we also report a SUSTAINED switch count: per GT animal we drop identity
runs shorter than MIN_RUN frames (brief flicker = noise), re-merge, and count the
remaining identity changes. For a stable tracker raw≈sustained; for an oscillating
one sustained << raw.

Run per tracker over the multi-animal (animals>=2) cam-sessions, in parallel.

Usage:
  PYTHONPATH=. .venv/bin/python scripts/switch_breakdown.py \
      <tracker_results_dir> <pred_h5_dir> [--min-run 30] [--workers 16]
"""
import argparse
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import motmetrics as mm

from scripts.evaluate import load_gt, bbox_from_kpts, luc3d_assignments_for_cam

REPO = Path(__file__).resolve().parents[1]
CAMS = ["back", "backL", "mid", "midL", "top", "topL"]
MASTER = REPO / "outputs" / "detections_only_master_sheet.tsv"


def sustained_transitions(hid_seq, min_run):
    """hid_seq: list of hypothesis ids over matched frames (in time order) for one GT.
    Drop runs shorter than min_run, merge, count remaining identity changes."""
    if not hid_seq:
        return 0, 0
    # compress to runs
    runs = []
    for h in hid_seq:
        if runs and runs[-1][0] == h:
            runs[-1][1] += 1
        else:
            runs.append([h, 1])
    raw_trans = len(runs) - 1
    # keep only sustained runs, then collapse consecutive duplicates
    kept = [h for h, n in runs if n >= min_run]
    merged = []
    for h in kept:
        if not merged or merged[-1] != h:
            merged.append(h)
    sustained = max(0, len(merged) - 1)
    return raw_trans, sustained


def one_cam(args):
    pred_h5, gt_h5, js, sidx, cam, min_run, mode, bt_h5 = args
    with h5py.File(pred_h5, "r") as f:
        raw = f["tracks"][sidx]
    gt, occ = load_gt(gt_h5)
    n = min(raw.shape[0], gt.shape[0]); na = raw.shape[1]
    ids = None
    byte_arr = None
    if mode == "json":
        ids = luc3d_assignments_for_cam(json.load(open(js)), cam, n, na)
    elif mode == "bytetrack":
        if not (bt_h5 and Path(bt_h5).exists()):
            return 0, 0
        with h5py.File(bt_h5, "r") as f:
            byte_arr = f["tracks"][:n]
    # mode == "sleap": identity is the raw slot index a (SLEAP tracker id)
    acc = mm.MOTAccumulator(auto_id=True)
    for fi in range(n):
        gb, gi = [], []
        for t in range(gt.shape[1]):
            if occ[fi, t]:
                b = bbox_from_kpts(gt[fi, t])
                if b is not None:
                    gb.append(b); gi.append(int(t))
        pb, pi = [], []
        if mode == "bytetrack":
            for a in range(min(na, byte_arr.shape[1])):
                row = byte_arr[fi, a]
                if not np.isfinite(row[0]):
                    continue
                pb.append(row[1:5]); pi.append(int(row[0]))
        else:
            for a in range(na):
                b = bbox_from_kpts(raw[fi, a])
                if b is None:
                    continue
                iv = a if mode == "sleap" else int(ids[fi, a])
                if iv >= 0:
                    pb.append(b); pi.append(int(iv))
        d = mm.distances.iou_matrix(np.array(gb) if gb else np.empty((0, 4)),
                                    np.array(pb) if pb else np.empty((0, 4)), max_iou=0.5)
        acc.update(gi, pi, d)
    ev = acc.mot_events
    m = ev[ev.Type.isin(["MATCH", "SWITCH"])]
    raw_sw = int((ev.Type == "SWITCH").sum())
    sust = 0
    for oid in m.OId.dropna().unique():
        seq = m[m.OId == oid].sort_values("FrameId").HId.tolist()
        _, s = sustained_transitions(seq, min_run)
        sust += s
    return raw_sw, sust


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results_dir")
    ap.add_argument("pred_h5_dir")
    ap.add_argument("--min-run", type=int, default=30)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--mode", choices=["json", "bytetrack", "sleap"], default="json")
    ap.add_argument("--bytetrack-dir", default=None)
    ap.add_argument("--label", default=None, help="row name in the tsv (default: dir name)")
    a = ap.parse_args()
    ms = pd.read_csv(MASTER, sep="\t", index_col=0).reset_index(drop=True)
    ms["session"] = ms["session"].astype(str)
    multi = ms[ms["num_animals"] >= 2]
    rdir = Path(a.results_dir)
    tasks = []
    for sidx, row in multi.iterrows():
        sid = row["session"]
        js = rdir / f"{sid}.json"
        if a.mode == "json" and not js.exists():
            continue
        for cam in CAMS:
            gt = row[f"{cam}_proofread_h5"]
            if not isinstance(gt, str) or not gt:
                continue
            bt = f"{a.bytetrack_dir}/{sid}/{cam}.h5" if a.bytetrack_dir else None
            tasks.append((f"{a.pred_h5_dir}/{cam}_predictions.h5", gt, str(js),
                          int(sidx), cam, a.min_run, a.mode, bt))
    raw_tot = sust_tot = 0
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for r, s in [f.result() for f in as_completed([ex.submit(one_cam, t) for t in tasks])]:
            raw_tot += r; sust_tot += s
    print(f"{rdir.name}: cam-sessions={len(tasks)}  RAW_switches={raw_tot}  "
          f"SUSTAINED_switches(min_run={a.min_run})={sust_tot}  "
          f"oscillation_fraction={1 - sust_tot/max(1,raw_tot):.2f}")
    # also stash to a tsv for the plot
    out = REPO / "outputs/PAF_3d_kalman/_liezlsweep/switch_breakdown.tsv"
    out.parent.mkdir(parents=True, exist_ok=True)
    line = f"{a.label or rdir.name}\t{raw_tot}\t{sust_tot}\n"
    with open(out, "a") as f:
        f.write(line)


if __name__ == "__main__":
    main()
