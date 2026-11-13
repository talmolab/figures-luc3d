#!/usr/bin/env python3
"""Classify Liezl ID-error episodes into a taxonomy, from dumped timelines.

An EPISODE = a maximal run (>= min_dur frames) where a GT animal is mislabeled in the
MAJORITY of cameras that detect it (cross-view consensus that the id is wrong). For each
episode we compute features and assign one class:

  FLICKER         duration < 30f  (transient; self-recovers; 72% of raw switches)
  CROWD           session has >=3 animals (multi-animal pileup)
  DROPOUT_REENTRY focal animal undetected in most cameras just before onset, then
                  reappears with the wrong id  (occlusion -> re-entry)
  CROSSVIEW_LOCK  focal animal stays detected, id flips across >=3 cameras at once and
                  persists (locked-in global swap; the worst / longest)
  CLOSE_CROSSING  both animals detected & close in 3D through the swap (classic crossing)

Outputs a per-episode TSV + per-class summary, and prints representative episodes
(longest per class) for video rendering.

Usage: PYTHONPATH=. .venv/bin/python scripts/classify_errors.py [--min-dur 30]
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
TL = REPO / "outputs/PAF_3d_kalman/_diag/timelines"


def runs_of(mask):
    """yield (start, end) for maximal True runs in 1D bool mask."""
    i = 0; n = len(mask)
    while i < n:
        if mask[i]:
            j = i
            while j < n and mask[j]:
                j += 1
            yield i, j
            i = j
        else:
            i += 1


def classify(ep, animals):
    if ep["duration"] < 30:
        return "FLICKER"
    if animals >= 3:
        return "CROWD"
    if ep["dropout_pre"] >= 0.4:
        return "DROPOUT_REENTRY"
    if ep["max_cams_wrong"] >= 3 and ep["duration"] >= 300:
        return "CROSSVIEW_LOCK"
    return "CLOSE_CROSSING"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-dur", type=int, default=30)
    a = ap.parse_args()
    rows = []
    for npz in sorted(TL.glob("*.npz")):
        sid = npz.stem
        d = np.load(npz, allow_pickle=True)
        matched = d["matched"]              # (ncams, nf, nt)
        correct = d["correct"]; inter3d = d["inter3d"]
        animals = int(d["animals"]); difficulty = int(d["difficulty"])
        ncams, nf, nt = matched.shape
        for t in range(nt):
            if correct[t] < 0:
                continue
            mt = matched[:, :, t]                       # (ncams, nf)
            detected = mt >= 0                          # has a detection in that cam
            wrong = detected & (mt != correct[t])
            ncam_det = detected.sum(0)                  # (nf,)
            ncam_wrong = wrong.sum(0)
            wrong_frac = np.where(ncam_det > 0, ncam_wrong / np.maximum(ncam_det, 1), 0.0)
            mask = wrong_frac >= 0.5
            for s, e in runs_of(mask):
                dur = e - s
                if dur < a.min_dur:
                    cls_dur = dur  # still record short as FLICKER below if >= min_dur filter
                # pre-onset dropout: fraction of [s-30,s) frames where focal undetected in majority of cams
                w0 = max(0, s - 30)
                pre = ncam_det[w0:s]
                dropout_pre = float(np.mean(pre < (ncams / 2.0))) if len(pre) else 0.0
                onset_inter = np.nanmean(inter3d[max(0, s - 5):s + 5])
                ep = dict(session=sid, gt_track=t, correct_id=int(correct[t]),
                          start=int(s), end=int(e), duration=int(dur),
                          onset=int(s), max_cams_wrong=int(ncam_wrong[s:e].max()),
                          dropout_pre=dropout_pre, onset_inter_mm=float(onset_inter),
                          recovered=int(e < nf - 60), animals=animals, difficulty=difficulty)
                ep["cls"] = classify(ep, animals)
                rows.append(ep)
    df = pd.DataFrame(rows)
    df = df[df.duration >= a.min_dur].reset_index(drop=True)
    out = REPO / "outputs/PAF_3d_kalman/_diag/error_episodes.tsv"
    df.to_csv(out, sep="\t", index=False)

    print(f"{len(df)} episodes (>= {a.min_dur}f).  total mislabeled frames = {df.duration.sum():,}\n")
    summ = df.groupby("cls").agg(n=("duration", "size"), frames=("duration", "sum"),
                                 med_dur=("duration", "median"), med_inter=("onset_inter_mm", "median"),
                                 frac_recovered=("recovered", "mean")).sort_values("frames", ascending=False)
    summ["frac_recovered"] = (100 * summ["frac_recovered"]).round(0)
    summ["med_inter"] = summ["med_inter"].round(0)
    print(summ.to_string())
    print("\n=== representative episodes (longest 3 per class) ===")
    for cls, g in df.groupby("cls"):
        print(f"\n# {cls}")
        cols = ["session", "gt_track", "correct_id", "start", "end", "duration",
                "max_cams_wrong", "dropout_pre", "onset_inter_mm", "recovered", "animals", "difficulty"]
        print(g.sort_values("duration", ascending=False).head(3)[cols].to_string(index=False))


if __name__ == "__main__":
    main()
