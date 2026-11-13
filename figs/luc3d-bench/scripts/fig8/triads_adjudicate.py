"""When LUC3D's grouping disagrees with s-DANNCE's channel identity, which one is right?

This is the question the TRIADS experiment exists to answer, and it cannot be
answered by scoring LUC3D against the channel, because the channel is not ground
truth -- it is the COM network's own per-camera output index, fixed by a human once
per cohort during labelling. If the network puts animal A in channel 0 in five
cameras and channel 1 in the sixth, s-DANNCE's 3D COM for that animal silently
blends two animals, and any metric that treats the channel as truth would score
LUC3D's *correct* regrouping as an error.

Geometry can adjudicate. A grouping that bundles detections of the SAME animal
across cameras triangulates to a point that reprojects near all of them; a grouping
that mixes two animals does not. So for every frame where the two disagree, we
triangulate both groupings and compare their mean reprojection residual. Lower wins.

This is not circular: LUC3D's cost function uses reprojection, but it is scored here
on a quantity computed independently for BOTH candidate groupings from the same
detections, and the channel grouping is free to win -- and does, on some frames.
The residual margin is reported so near-ties can be excluded rather than counted.
"""
from __future__ import annotations

import argparse
import itertools
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from sdannce_prepare import load_dannce_cam


def dlt(pts, Ps):
    A = []
    for xy, P in zip(pts, Ps):
        A.append(xy[0] * P[2] - P[0])
        A.append(xy[1] * P[2] - P[1])
    if len(A) < 4:
        return None
    _, _, Vt = np.linalg.svd(np.asarray(A))
    X = Vt[-1]
    if abs(X[3]) < 1e-12:
        return None
    return X[:3] / X[3]


def grouping_residual(groups, Ps):
    """IN-SAMPLE residual: triangulate each identity from ALL its members and
    reproject into those same cameras.

    THIS IS CIRCULAR as a way to adjudicate LUC3D against the channel, because
    reprojection error is LUC3D's own association objective -- it disagrees with the
    channel precisely when it has found a lower-residual grouping. Retained only for
    reference; use `grouping_residual_heldout`."""
    errs = []
    for g in groups:
        if len(g) < 2:
            continue
        X = dlt([xy for _, xy in g], [Ps[c] for c, _ in g])
        if X is None:
            continue
        for c, xy in g:
            p = Ps[c] @ np.r_[X, 1]
            if abs(p[2]) < 1e-9:
                continue
            errs.append(np.linalg.norm(p[:2] / p[2] - xy))
    return float(np.mean(errs)) if errs else np.nan


def grouping_residual_heldout(groups, Ps):
    """LEAVE-ONE-CAMERA-OUT residual: for each member camera h of each identity,
    triangulate that identity from its OTHER cameras only and reproject into h.

    Neither method optimises this quantity: the 3D point is built without camera h,
    so the residual there is a genuine prediction. A grouping that has bundled two
    different animals predicts the held-out view badly no matter which method
    proposed it. Same protocol as Fig 4d, for the same reason."""
    errs = []
    for g in groups:
        if len(g) < 3:            # need >=2 to triangulate after holding one out
            continue
        for h in range(len(g)):
            rest = [g[i] for i in range(len(g)) if i != h]
            X = dlt([xy for _, xy in rest], [Ps[c] for c, _ in rest])
            if X is None:
                continue
            c, xy = g[h]
            p = Ps[c] @ np.r_[X, 1]
            if abs(p[2]) < 1e-9:
                continue
            errs.append(np.linalg.norm(p[:2] / p[2] - xy))
    return float(np.mean(errs)) if errs else np.nan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--track", type=Path, required=True)
    ap.add_argument("--prepared", type=Path, required=True)
    ap.add_argument("--session-dir", type=Path, required=True)
    ap.add_argument("--margin", type=float, default=1.0,
                    help="px; disagreements closer than this are counted as ties")
    ap.add_argument("--out", type=Path)
    ap.add_argument("--in-sample", action="store_true",
                    help="use the CIRCULAR in-sample residual (reference only)")
    args = ap.parse_args()

    run = json.loads(args.track.read_text())
    meta = json.loads(args.prepared.read_text())
    blob = np.fromfile(meta["binary"], dtype=np.float64).reshape(-1, 2)
    names = meta["cameras"]
    A = meta["n_animals"]

    cal = sorted((args.session_dir / "calibration").glob("hires_cam*_params.mat"),
                 key=lambda p: int("".join(c for c in p.stem if c.isdigit())))
    cams = [load_dannce_cam(p) for p in cal]
    Ps = [c["K"] @ np.hstack([c["R"], c["tvec"].reshape(3, 1)]) for c in cams]

    perms = list(itertools.permutations(range(A)))
    n_agree = n_dis = 0
    luc_win = chan_win = tie = 0
    margins = []

    for fr, tf in zip(meta["frames"], run["frames"]):
        assign = tf.get("assignments") or {}
        if not assign:
            continue
        # channel grouping and LUC3D grouping, both as identity -> [(cam, xy)]
        chan = {a: [] for a in range(A)}
        luc = {}
        for ci, cam in enumerate(names):
            for k, (true_id, off) in enumerate(fr["c"][cam]):
                xy = blob[off]
                chan[true_id].append((ci, xy))
                ident = assign.get(f"{cam}:{k}")
                if ident is None:
                    continue
                luc.setdefault(ident, []).append((ci, xy))

        cg = [v for v in chan.values() if v]
        lg = [v for v in luc.values() if v]
        # Same partition? Compare as sets of (cam, instance) memberships.
        def sig(groups):
            return frozenset(frozenset((c, float(xy[0]), float(xy[1])) for c, xy in g)
                             for g in groups)
        if sig(cg) == sig(lg):
            n_agree += 1
            continue
        n_dis += 1
        scorer = grouping_residual if args.in_sample else grouping_residual_heldout
        rc = scorer(cg, Ps)
        rl = scorer(lg, Ps)
        if not (np.isfinite(rc) and np.isfinite(rl)):
            continue
        margins.append(rc - rl)
        if rc - rl > args.margin:
            luc_win += 1
        elif rl - rc > args.margin:
            chan_win += 1
        else:
            tie += 1

    m = np.asarray(margins, dtype=float)
    row = {
        "session": run["session"], "frames": n_agree + n_dis,
        "agree": n_agree, "disagree": n_dis,
        "disagree_frac": n_dis / max(1, n_agree + n_dis),
        "luc3d_better": luc_win, "channel_better": chan_win, "tie": tie,
        "scorer": "in_sample(CIRCULAR)" if args.in_sample else "held_out_camera",
        "median_margin_px": float(np.median(m)) if m.size else np.nan,
        "mean_margin_px": float(np.mean(m)) if m.size else np.nan,
    }
    print(pd.DataFrame([row]).to_string(index=False))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([row]).to_csv(args.out, index=False)


if __name__ == "__main__":
    main()
