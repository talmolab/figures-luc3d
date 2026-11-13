#!/usr/bin/env python3
"""Global trajectory-smoothness swap fixer (2-animal): Viterbi over the whole session.

Unlike the online KF (which oscillated) this is a GLOBAL dynamic program. For each
frame where both identities are present we have two 3D points; the binary state
s(f) says whether identity-0 is point-0 (keep) or point-1 (swap, relative to Liezl).
We choose the s(f) sequence minimizing total 3D motion of both identity trajectories,
with a switch penalty so the labeling only flips when the geometry strongly demands
it (commits per segment, no per-frame flicker). Then relabel.

This is hypothesis testing at the trajectory level: away from close encounters the
smooth assignment is unambiguous and Liezl's spurious swaps get undone; at genuine
crossings both hypotheses cost the same and we keep Liezl's call (switch penalty).

Only 2-animal sessions are handled; others are copied through unchanged.
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from scripts.temporal_link_3d import load_calib, centroid, triangulate, CAMS


def fix_two(frames_pts, switch_penalty):
    """frames_pts: sorted list of (frame, p0, p1) with both identities triangulated.
    Returns dict frame -> swap_bool (whether to swap Liezl's id0/id1 at that frame)."""
    n = len(frames_pts)
    if n == 0:
        return {}
    INF = 1e18
    # cost[k] = min total motion ending at frame i in state k (0 keep / 1 swap)
    prev_cost = [0.0, switch_penalty]   # frame 0: prefer keep (Liezl); swap costs penalty
    back = []
    for i in range(1, n):
        f0, a0, b0 = frames_pts[i-1]
        f1, a1, b1 = frames_pts[i]
        # identity positions under state s at frame i: s=0 -> (a,b), s=1 -> (b,a)
        pos = {0: (a1, b1), 1: (b1, a1)}
        pos_prev = {0: (a0, b0), 1: (b0, a0)}
        cur = [INF, INF]; bk = [0, 0]
        for s in (0, 1):
            id0_i, id1_i = pos[s]
            for sp in (0, 1):
                id0_p, id1_p = pos_prev[sp]
                move = np.linalg.norm(id0_i-id0_p) + np.linalg.norm(id1_i-id1_p)
                c = prev_cost[sp] + move + (switch_penalty if s != sp else 0.0)
                if c < cur[s]:
                    cur[s] = c; bk[s] = sp
        prev_cost = cur; back.append(bk)
    # backtrack
    s = 0 if prev_cost[0] <= prev_cost[1] else 1
    states = [s]
    for i in range(len(back)-1, -1, -1):
        s = back[i][s]
        states.append(s)
    states.reverse()
    return {frames_pts[i][0]: bool(states[i]) for i in range(n)}


def process(in_json, h5s, row, calib_path, switch_penalty):
    d = json.load(open(in_json))
    if int(d.get("numAnimals", 2)) != 2:
        return d, 0, False
    P, K, dist = load_calib(calib_path)
    tr = {c: h5s[c]["tracks"][row] for c in CAMS}
    # per frame: liezl_id -> 3D point
    fp = {}
    for fr in d["frames"]:
        f = fr["frame"]
        byid = defaultdict(list)
        for k, i in fr["assignments"]:
            cam, pos = k.split(":")
            c = centroid(tr[cam][f, int(pos)])
            if c is not None:
                byid[i].append((cam, c))
        pts = {i: triangulate(o, P, K, dist) for i, o in byid.items() if len(o) >= 2}
        fp[f] = pts
    # identify the two dominant liezl ids
    import collections
    cnt = collections.Counter()
    for f, pts in fp.items():
        for i in pts:
            cnt[i] += 1
    if len(cnt) < 2:
        return d, 0, False
    id0, id1 = [i for i, _ in cnt.most_common(2)]
    frames_pts = []
    for f in sorted(fp):
        pts = fp[f]
        if id0 in pts and id1 in pts:
            frames_pts.append((f, pts[id0], pts[id1]))
    swap_at = fix_two(frames_pts, switch_penalty)

    # Build per-frame relabel map for the two dominant ids. A frame not in swap_at
    # (only one id present) inherits the nearest decided frame's swap state.
    decided = sorted(swap_at)
    dec_arr = np.array(decided) if decided else np.array([])

    def swap_state(f):
        if f in swap_at:
            return swap_at[f]
        if len(dec_arr) == 0:
            return False
        j = int(np.searchsorted(dec_arr, f))
        j = min(max(j, 0), len(dec_arr)-1)
        return swap_at[dec_arr[j]]

    n_changed = 0
    for fr in d["frames"]:
        f = fr["frame"]
        if not swap_state(f):
            continue
        new = []
        for k, i in fr["assignments"]:
            ni = i
            if i == id0:
                ni = id1
            elif i == id1:
                ni = id0
            if ni != i:
                n_changed += 1
            new.append([k, int(ni)])
        fr["assignments"] = new
    return d, n_changed, True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", required=True)
    ap.add_argument("--h5-dir", required=True)
    ap.add_argument("--master", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--switch-penalty", type=float, default=200.0,
                    help="mm cost to flip the labeling; higher = more conservative")
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
        d, nch, did = process(ij, h5s, row, calib_by_sid[str(sid)], a.switch_penalty)
        tot += nch
        json.dump(d, open(out_dir / f"{sid}.json", "w"))
        print(f"  {sid}: swapped {nch} assignments {'(2-animal)' if did else '(skipped)'}")
    print(f"DONE — {tot} relabeled -> {out_dir}")


if __name__ == "__main__":
    main()
