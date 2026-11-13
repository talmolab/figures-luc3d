#!/usr/bin/env python3
"""Rank the swap episodes that cause the LONGEST stretches of wrong-identity frames.

For each multi-animal session we establish each GT animal's CORRECT tracker identity
(the id it is matched to most often, session-wide across all 6 cameras). Then per
camera we find every contiguous run where that GT animal is matched to a DIFFERENT
valid id — a sustained mislabel — and record its duration in frames. We rank globally
by duration so we can make grid videos of the worst offenders.

Output: outputs/PAF_3d_kalman/_diag/long_swaps.tsv
"""
import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from scripts.evaluate import load_gt, bbox_from_kpts, luc3d_assignments_for_cam

REPO = Path(__file__).resolve().parents[1]
MASTER = REPO / "outputs" / "detections_only_master_sheet.tsv"
CAMS = ["back", "backL", "mid", "midL", "top", "topL"]


def iou(a, b):
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0, ix1 - ix0), max(0, iy1 - iy0)
    inter = iw * ih
    ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def gt_to_id_timeline(raw, gt, occ, ids, n, na, nt):
    """Return matched[fi, t] = predicted id assigned to GT track t (or -1)."""
    matched = np.full((n, nt), -1, np.int64)
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
        # greedy max-IoU GT<->pred
        pairs = []
        for t, gbx in gb.items():
            for j, (pbx, iv) in enumerate(pb):
                io = iou(gbx, pbx)
                if io >= 0.5:
                    pairs.append((io, t, j, iv))
        pairs.sort(reverse=True)
        ut, uj = set(), set()
        for io, t, j, iv in pairs:
            if t in ut or j in uj:
                continue
            matched[fi, t] = iv; ut.add(t); uj.add(j)
    return matched


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", required=True)
    ap.add_argument("--pred-h5-dir", required=True)
    ap.add_argument("--min-dur", type=int, default=150)
    a = ap.parse_args()
    ms = pd.read_csv(MASTER, sep="\t"); ms["session"] = ms["session"].astype(str)
    multi = ms[ms["num_animals"] >= 2]
    rdir = Path(a.results_dir)

    # cache per-cam pred H5 handles
    h5 = {c: h5py.File(f"{a.pred_h5_dir}/{c}_predictions.h5", "r") for c in CAMS}
    rows = []
    for sidx, row in multi.iterrows():
        sid = row["session"]
        js = rdir / f"{sid}.json"
        if not js.exists():
            continue
        jd = json.load(open(js))
        cam_h5idx = int(np.where(ms["session"].values == sid)[0][0])
        # session-level vote: id histogram per GT track over all cams
        votes = {}
        timelines = {}
        for cam in CAMS:
            gtp = row[f"{cam}_proofread_h5"]
            if not isinstance(gtp, str) or not gtp:
                continue
            raw = h5[cam]["tracks"][cam_h5idx]
            gt, occ = load_gt(gtp)
            n = min(raw.shape[0], gt.shape[0]); na = raw.shape[1]; nt = gt.shape[1]
            ids = luc3d_assignments_for_cam(jd, cam, n, na)
            matched = gt_to_id_timeline(raw, gt, occ, ids, n, na, nt)
            timelines[cam] = matched
            for t in range(nt):
                col = matched[:, t]
                for iv in col[col >= 0]:
                    votes.setdefault(t, {}).setdefault(int(iv), 0)
                    votes[t][int(iv)] += 1
        correct = {t: max(v, key=v.get) for t, v in votes.items()}
        # find wrong runs per cam/track
        for cam, matched in timelines.items():
            for t in range(matched.shape[1]):
                if t not in correct:
                    continue
                cid = correct[t]
                col = matched[:, t]
                fi = 0; N = len(col)
                while fi < N:
                    iv = col[fi]
                    if iv >= 0 and iv != cid:
                        j = fi
                        while j < N and col[j] == iv:
                            j += 1
                        dur = j - fi
                        if dur >= a.min_dur:
                            rows.append((sid, cam, t, int(cid), int(iv), fi, j, dur,
                                         int(row["difficulty"]), int(row["num_animals"])))
                        fi = j
                    else:
                        fi += 1
    for c in h5.values():
        c.close()
    df = pd.DataFrame(rows, columns=["session", "cam", "gt_track", "correct_id", "wrong_id",
                                     "start", "end", "duration", "difficulty", "animals"])
    df = df.sort_values("duration", ascending=False).reset_index(drop=True)
    out = REPO / "outputs/PAF_3d_kalman/_diag/long_swaps.tsv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, sep="\t", index=False)
    print(f"{len(df)} wrong-runs >= {a.min_dur}f. total mislabeled frames = {df.duration.sum():,}")
    print("\nTop 15 longest sustained mislabels:")
    print(df.head(15).to_string(index=False))
    # session-level: which sessions have the most total wrong-frames
    print("\nWorst sessions by total mislabeled frames:")
    print(df.groupby(["session", "difficulty"]).duration.sum().sort_values(ascending=False).head(10).to_string())


if __name__ == "__main__":
    main()
