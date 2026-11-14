#!/usr/bin/env python3
"""Mouse-Dyad-10M: cross-camera identity mismatches left in the reference 3D (read-only QC).

Identities were proofread per camera (the lab's per-camera .slp, track_0 = male). A swap
that is consistent within one camera, typically started while the mice touch, passes that
proofreading, but the camera then disagrees with the other four and the 3D triangulated
through it mixes the two animals. This screen finds those stretches.

Per session, camera and frame: which reference animal the proofread track_0 sits on (median
keypoint distance to the reprojected reference 3D). A mismatch run is >= 15 frames (0.1 s) of
track_0 on the female. A run CORRUPTS the 3D if in more than half its frames the reference 3D
is >20 px from every detection in >= 3 of the 5 cameras (detections = the identity-free
SLEAP predictions of the tracker pool, so this only asks whether the 3D lands on a mouse).

Inputs (read only):
  <BMimica>/<session>/<ts>_points3d_translated_rotated_metric.h5   reference 3D
  <BMimica>/<session>/<cam>/<cam>-<ts>.analysis.h5                 proofread tracks (export of the .slp)
  <bench>/outputs/bmimica/gt/<session>/<cam>/proofread.analysis.h5 reference 3D reprojected
  <bench>/outputs/bmimica/det_h5_bvis5/<session>/<cam>_predictions.h5  detections

Writes data/qc/crossview_mismatch.csv (one row per mismatch run) and
out/qc_crossview_mismatch.json (per-session frames and the corpus summary the tex quotes).

    cd figs && $PY qc/crossview_mismatch.py
"""
import csv, glob, json, sys
from multiprocessing import Pool
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.qc_exclude import excluded  # noqa: E402

BMIMICA = "/root/vast/eric/BMimica"
BENCH = "/root/vast/eric/luc3d-bench/outputs/bmimica"
CAMS = ["21241563", "21369048", "21372315", "21372316", "22085397"]
MIN_RUN, PX, MIN_CAMS = 15, 20, 3


def runs(x):
    d = np.diff(np.r_[0, x.astype(int), 0])
    return [(int(a), int(b)) for a, b in zip(np.where(d == 1)[0], np.where(d == -1)[0])
            if b - a >= MIN_RUN]


def session(s):
    with h5py.File(glob.glob(f"{BMIMICA}/{s}/*_points3d*.h5")[0]) as h:
        F = len(h["tracks"])
    off, on_female = [], {}
    for c in CAMS:
        with h5py.File(f"{BENCH}/gt/{s}/{c}/proofread.analysis.h5") as h:
            g = np.transpose(h["tracks"][:], (3, 0, 2, 1))
        with h5py.File(f"{BENCH}/det_h5_bvis5/{s}/{c}_predictions.h5") as h:
            d = h["tracks"][0][...]
        with h5py.File(glob.glob(f"{BMIMICA}/{s}/{c}/*.analysis.h5")[0]) as h:
            r = np.transpose(h["tracks"][:], (3, 0, 2, 1))[:, :2]
        F = min(F, len(g), len(d), len(r))
        with np.errstate(all="ignore"):
            md = np.nanmedian(np.linalg.norm(g[:F, :, None] - d[:F, None, :], axis=-1), -1)
            off.append(np.nanmin(np.where(np.isnan(md), np.inf, md), -1) > PX)
            d0 = np.nanmedian(np.linalg.norm(r[:F, 0] - g[:F, 0], axis=-1), -1)
            d1 = np.nanmedian(np.linalg.norm(r[:F, 0] - g[:F, 1], axis=-1), -1)
        on_female[c] = ~np.isnan(d0) & ~np.isnan(d1) & (d1 < d0)
    bad3d = (np.stack([o[:F] for o in off], 1).sum(1) >= MIN_CAMS).any(1)
    rows, corrupt = [], np.zeros(F, bool)
    for c in CAMS:
        for a, b in runs(on_female[c][:F]):
            frac = float(bad3d[a:b].mean())
            rows.append(dict(session=s, camera=c, start=a, end=b, frames=b - a,
                             frac_3d_off=round(frac, 3), corrupts_3d=frac > 0.5))
            if frac > 0.5:
                corrupt[a:b] = True
    spans = runs(corrupt)
    return dict(session=s, frames=int(F), corrupted_frames=int(corrupt.sum()),
                longest_span=max((b - a for a, b in spans), default=0)), rows


def main():
    figs = Path(__file__).resolve().parents[1]
    sessions = sorted(Path(p).name for p in glob.glob(f"{BENCH}/gt/*"))
    with Pool(8) as p:
        res = p.map(session, sessions)
    per = [r for r, _ in res]
    rows = [x for _, rr in res for x in rr]
    kept = [r for r in per if not excluded(r["session"])]
    hit = [r for r in kept if r["corrupted_frames"]]
    F = sum(r["frames"] for r in kept); C = sum(r["corrupted_frames"] for r in kept)
    longest = max(hit, key=lambda r: r["longest_span"])
    summary = dict(
        sessions=len(per), kept_sessions=len(kept), kept_frames=F,
        kept_corrupted_frames=C, kept_corrupted_pct=round(100 * C / F, 3),
        kept_sessions_affected=len(hit),
        kept_sessions_over_1s=sum(r["corrupted_frames"] >= 150 for r in hit),
        kept_longest_span_frames=longest["longest_span"],
        kept_longest_span_s=round(longest["longest_span"] / 150, 1),
        kept_longest_session=longest["session"],
        mismatch_runs=len(rows), mismatch_runs_corrupting_3d=sum(r["corrupts_3d"] for r in rows),
        excluded={r["session"]: r for r in per if excluded(r["session"])},
        params=dict(min_run_frames=MIN_RUN, px=PX, min_cams=MIN_CAMS, fps=150))
    out = figs / "data/qc/crossview_mismatch.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(sorted(rows, key=lambda r: (r["session"], r["camera"], r["start"])))
    json.dump(dict(summary=summary, sessions=per), open(figs / "out/qc_crossview_mismatch.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in summary.items() if k != "excluded"}, indent=1))
    print("excluded:", {k: (v["corrupted_frames"], v["longest_span"]) for k, v in summary["excluded"].items()})


if __name__ == "__main__":
    main()
