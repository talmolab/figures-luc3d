#!/usr/bin/env python3
"""Conservative 3D swap-corrector: fix ONLY sustained identity swaps in a tracker's
output, trusting the tracker everywhere else.

Unlike temporal_link_3d (which replaced/voted identity everywhere and oscillated),
this keeps the input identity by default and overrides a (frame, identity) only
when a 3D constant-velocity motion track disagrees for a SUSTAINED run
(>= --min-run frames). On the clean keeptrack pool (<=num_animals/frame) the 3D
tracks are reliable, so a sustained disagreement is a real swap.

Usage:
  PYTHONPATH=. .venv/bin/python scripts/swap_corrector_3d.py \
      --in-dir  outputs/PAF_3d_kalman/keeptrack_liezl_results \
      --h5-dir  outputs/keeptrack_h5s \
      --master  outputs/sleap_nn_master_sheet.tsv \
      --out-dir outputs/PAF_3d_kalman/keeptrack_swapfix_results \
      [--min-run 30] [--sessions S1 S2 ...]
"""
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from scripts.temporal_link_3d import load_calib, centroid, triangulate, run_kf, CAMS


def process(in_json, h5s, row, calib_path, args):
    d = json.load(open(in_json))
    P, K, dist = load_calib(calib_path)
    tr = {c: h5s[c]["tracks"][row] for c in CAMS}
    frame_points = {}
    for fr in d["frames"]:
        f = fr["frame"]
        byid = defaultdict(list)
        for k, i in fr["assignments"]:
            cam, pos = k.split(":")
            c = centroid(tr[cam][f, int(pos)])
            if c is not None:
                byid[i].append((cam, c))
        pts = {i: triangulate(o, P, K, dist) for i, o in byid.items() if len(o) >= 2}
        if pts:
            frame_points[f] = pts

    n_targets = max(1, int(d.get("numAnimals", 2)))
    track_map = run_kf(frame_points, args.q_accel, args.r_meas, args.gate,
                       args.max_miss, n_targets, args.const_pos)
    # motion track -> modal input identity
    votes = defaultdict(Counter)
    for (f, lid), tid in track_map.items():
        votes[tid][lid] += 1
    track_modal = {tid: c.most_common(1)[0][0] for tid, c in votes.items()}
    # candidate identity per (frame, input-lid) = the modal id of its motion track
    cand = {(f, lid): track_modal[tid] for (f, lid), tid in track_map.items()}

    # For each input identity stream, override only SUSTAINED disagreement runs.
    override = {}
    by_lid = defaultdict(list)
    for (f, lid) in track_map:
        by_lid[lid].append(f)
    for lid, frames in by_lid.items():
        frames.sort()
        seq = [cand[(f, lid)] for f in frames]
        # runs of equal candidate
        i = 0
        while i < len(seq):
            j = i
            while j < len(seq) and seq[j] == seq[i]:
                j += 1
            if seq[i] != lid and (j - i) >= args.min_run:
                for f in frames[i:j]:
                    override[(f, lid)] = seq[i]
            i = j

    # nearest-in-time override lookup for single-view dets
    ov_by_lid = defaultdict(list)
    for (f, lid), v in override.items():
        ov_by_lid[lid].append((f, v))
    for lid in ov_by_lid:
        ov_by_lid[lid].sort()
    ov_frames = {lid: np.array([f for f, _ in arr]) for lid, arr in ov_by_lid.items()}

    def lookup(f, lid):
        if (f, lid) in override:
            return override[(f, lid)]
        # if this lid was triangulated this frame but NOT overridden, keep lid
        if (f, lid) in track_map:
            return lid
        # single-view: borrow nearest overridden decision within min_run, else keep
        arr = ov_by_lid.get(lid)
        if not arr:
            return lid
        fr = ov_frames[lid]
        k = int(np.searchsorted(fr, f))
        for cand_k in (k, k - 1):
            if 0 <= cand_k < len(fr) and abs(fr[cand_k] - f) <= args.min_run:
                return arr[cand_k][1]
        return lid

    n_changed = 0
    for fr in d["frames"]:
        f = fr["frame"]
        new = []
        for k, i in fr["assignments"]:
            nid = lookup(f, i)
            if nid != i:
                n_changed += 1
            new.append([k, int(nid)])
        fr["assignments"] = new
    return d, n_changed, len(override)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", required=True)
    ap.add_argument("--h5-dir", required=True)
    ap.add_argument("--master", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--min-run", type=int, default=30)
    ap.add_argument("--gate", type=float, default=16.0)
    ap.add_argument("--q-accel", type=float, default=30.0)
    ap.add_argument("--r-meas", type=float, default=100.0)
    ap.add_argument("--max-miss", type=int, default=90)
    ap.add_argument("--const-pos", action="store_true")
    ap.add_argument("--sessions", nargs="*", default=None)
    a = ap.parse_args()
    ms = pd.read_csv(a.master, sep="\t")
    sid_col = "session" if "session" in ms.columns else ms.columns[0]
    sid_arr = ms[sid_col].astype(str).tolist()
    calib_by_sid = dict(zip(ms[sid_col].astype(str), ms["calibration_toml"]))
    out_dir = Path(a.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    h5s = {c: h5py.File(f"{a.h5_dir}/{c}_predictions.h5", "r") for c in CAMS}
    in_dir = Path(a.in_dir)
    targets = a.sessions or [p.stem for p in sorted(in_dir.glob("*.json"))]
    tot = 0
    for sid in targets:
        ij = in_dir / f"{sid}.json"
        if not ij.exists():
            continue
        row = sid_arr.index(str(sid))
        d, nch, nov = process(ij, h5s, row, calib_by_sid[str(sid)], a)
        tot += nch
        json.dump(d, open(out_dir / f"{sid}.json", "w"))
        print(f"  {sid}: overrode {nch} assignments ({nov} sustained-swap frame-slots)")
    print(f"DONE — {tot} assignments overridden -> {out_dir}")


if __name__ == "__main__":
    main()
