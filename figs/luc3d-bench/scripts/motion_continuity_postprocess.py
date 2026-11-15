#!/usr/bin/env python3
"""Motion-continuity correction for detections-only tracker output.

Diagnosis (2026-06-22): in detections-only mode the cross-view trackers swap
identities between WELL-SEPARATED animals (>150px apart) — i.e. they have no
working temporal motion model and re-decide identity almost per-frame. This
post-process fixes that the cheap way: per camera, link detections across frames
into motion trajectories (constant-velocity NN with a distance gate — robust
precisely because the animals are well separated), then assign each trajectory
its MODAL identity from the tracker. Identity swaps within a continuous
trajectory get voted away; cross-view identity (the tracker's global id set) is
preserved because we only ever re-use the tracker's own ids.

Input : <tracker>_results/<session>.json  (assignments keyed "{cam}:{det_pos}")
        widened detection H5s (bbox source, same det_pos indexing)
Output: <out_dir>/<session>.json  (same schema, identities regularized)

Usage:
  PYTHONPATH=. .venv/bin/python scripts/motion_continuity_postprocess.py \
      --in-dir  outputs/PAF_3d_kalman/detok_liezl_results \
      --h5-dir  outputs/detections_only_h5s \
      --master  outputs/detections_only_master_sheet.tsv \
      --out-dir outputs/PAF_3d_kalman/detok_liezl_motionfix_results \
      [--gate 70] [--max-gap 15] [--sessions S1 S2 ...]
"""
import argparse
import json
from collections import Counter
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
CAMS = ["back", "backL", "mid", "midL", "top", "topL"]


def centroid(kpts):
    c = np.nanmean(kpts, axis=0)
    return c if np.isfinite(c).all() else None


def link_trajectories(frame_dets, gate=70.0, max_gap=15):
    """frame_dets: dict frame -> list of (det_pos, centroid).
    Greedy constant-velocity NN linking. Returns list of trajectories, each a
    list of (frame, det_pos)."""
    trajectories = []
    active = []  # each: dict(pos, vel, last_frame, members[list of (frame,det_pos)])
    for f in sorted(frame_dets.keys()):
        dets = frame_dets[f]
        # drop stale tracks
        still = []
        for tr in active:
            if f - tr["last_frame"] > max_gap:
                trajectories.append(tr["members"])
            else:
                still.append(tr)
        active = still
        # predicted positions
        preds = []
        for tr in active:
            dt = f - tr["last_frame"]
            preds.append(tr["pos"] + tr["vel"] * dt)
        # greedy match by ascending distance
        pairs = []
        for di, (dp, c) in enumerate(dets):
            for ti, p in enumerate(preds):
                d = float(np.hypot(*(c - p)))
                if d <= gate:
                    pairs.append((d, di, ti))
        pairs.sort()
        used_d, used_t = set(), set()
        matched = {}
        for d, di, ti in pairs:
            if di in used_d or ti in used_t:
                continue
            used_d.add(di); used_t.add(ti); matched[di] = ti
        # update / create
        for di, (dp, c) in enumerate(dets):
            if di in matched:
                tr = active[matched[di]]
                dt = max(1, f - tr["last_frame"])
                tr["vel"] = (c - tr["pos"]) / dt
                tr["pos"] = c
                tr["last_frame"] = f
                tr["members"].append((f, dp))
            else:
                active.append(dict(pos=c, vel=np.zeros(2), last_frame=f,
                                   members=[(f, dp)]))
    for tr in active:
        trajectories.append(tr["members"])
    return trajectories


def process_session(in_json, h5s, session_row, gate, max_gap):
    d = json.load(open(in_json))
    # frame -> cam -> {det_pos: id}
    asg = {}
    for fr in d["frames"]:
        m = asg.setdefault(fr["frame"], {})
        for k, i in fr["assignments"]:
            cam, pos = k.split(":")
            m.setdefault(cam, {})[int(pos)] = i

    # new mapping: (frame, cam, det_pos) -> corrected id
    corrected = {}
    n_changed = 0
    for cam in CAMS:
        t = h5s[cam]["tracks"][session_row]  # (n_frames, n_dets, n_nodes, 2)
        frame_dets = {}
        labels = {}  # (frame, det_pos) -> original id
        for f, cammap in asg.items():
            posmap = cammap.get(cam)
            if not posmap:
                continue
            lst = []
            for pos, i in posmap.items():
                c = centroid(t[f, pos])
                if c is None:
                    continue
                lst.append((pos, c))
                labels[(f, pos)] = i
            if lst:
                frame_dets[f] = lst
        for traj in link_trajectories(frame_dets, gate, max_gap):
            ids = [labels[m] for m in traj if m in labels]
            if not ids:
                continue
            modal = Counter(ids).most_common(1)[0][0]
            for (f, pos) in traj:
                corrected[(f, cam, pos)] = modal
                if labels.get((f, pos)) != modal:
                    n_changed += 1

    # rewrite JSON preserving everything else
    for fr in d["frames"]:
        new = []
        for k, i in fr["assignments"]:
            cam, pos = k.split(":")
            ni = corrected.get((fr["frame"], cam, int(pos)), i)
            new.append([k, int(ni)])
        fr["assignments"] = new
    return d, n_changed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", required=True)
    ap.add_argument("--h5-dir", required=True)
    ap.add_argument("--master", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--gate", type=float, default=70.0)
    ap.add_argument("--max-gap", type=int, default=15)
    ap.add_argument("--sessions", nargs="*", default=None,
                    help="limit to these session ids (default: all)")
    a = ap.parse_args()

    ms = pd.read_csv(a.master, sep="\t")
    sid_col = "session" if "session" in ms.columns else ms.columns[0]
    sid_arr = ms[sid_col].astype(str).tolist()
    out_dir = Path(a.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    h5s = {c: h5py.File(f"{a.h5_dir}/{c}_predictions.h5", "r") for c in CAMS}

    in_dir = Path(a.in_dir)
    targets = a.sessions or [p.stem for p in sorted(in_dir.glob("*.json"))]
    total_changed = 0
    for sid in targets:
        ij = in_dir / f"{sid}.json"
        if not ij.exists():
            print(f"  skip {sid}: no input json"); continue
        row = sid_arr.index(str(sid))
        d, nch = process_session(ij, h5s, row, a.gate, a.max_gap)
        total_changed += nch
        json.dump(d, open(out_dir / f"{sid}.json", "w"))
        print(f"  {sid}: relabeled {nch} assignments")
    print(f"DONE — {len(targets)} sessions, {total_changed} assignments relabeled -> {out_dir}")


if __name__ == "__main__":
    main()
