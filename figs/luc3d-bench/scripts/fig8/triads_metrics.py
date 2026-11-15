"""IDF1 (within-view and cross-view), identity accuracy, and a purity/coverage
curve for LUC3D on social-DANNCE's real per-camera COM detections.

IDF1 has a closed form in this setting and it is worth stating why, because it is
not the general MOT case. The detections are SHARED: ground truth and prediction
are the same set of 2D points, and the only thing that differs is the identity
label attached to each. So there are no detection false positives or negatives, and
    IDF1 = 2*IDTP / (2*IDTP + IDFP + IDFN) = IDTP / N
where IDTP is the number of detections matched under the best ONE-TO-ONE mapping
from predicted identities to ground-truth animals (Hungarian over the
prediction x truth count matrix). One-to-one is what makes this a real metric: if
the tracker fragments an animal across several identities, only one of them can be
matched and the rest count against it.

  within-view IDF1   one mapping fitted PER CAMERA, then averaged over cameras.
                     This is what a per-camera tracker is scored on, and it does
                     not care whether camera 1's "animal 2" is camera 4's "animal 2".
  cross-view IDF1    ONE mapping fitted over all cameras pooled, so an identity has
                     to mean the same animal in every view. This is the quantity
                     Fig 7a is about, and the gap between the two is the whole
                     cross-view claim. A tracker that labels each camera
                     independently is capped near 1/C here; LUC3D assigns one global
                     identity, so its two numbers should be close -- and if they are
                     not, that is a real finding, not a formality.

Ground truth is s-DANNCE's output CHANNEL. Audited against 698 human clicks on 113
frames with zero channel/animal conflicts (`triads_channel_audit.py`), so it is
trustworthy where we can check it -- but those are the COM net's own training
frames, so treat it as a good reference, not a perfect one.

The curve is purity vs coverage, not a classifier PR curve: every detection already
has an identity, so there is nothing to threshold except CONFIDENCE. Each assignment
is scored by the reprojection residual of its detection against its identity's 3D
point triangulated from the OTHER cameras (leave-one-out, so the residual is not
fitted to the point it judges). Sweeping that threshold answers the question a
proofreader actually has: if I only keep the assignments the geometry is confident
about, how pure are they and how much of the data do I keep?
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from sdannce_prepare import load_dannce_cam


def idf1_from_counts(counts: np.ndarray) -> tuple[float, int]:
    """counts[pred, truth] -> (IDF1, IDTP) under the best one-to-one mapping."""
    n = counts.sum()
    if n == 0:
        return float("nan"), 0
    r, c = linear_sum_assignment(-counts)
    idtp = int(counts[r, c].sum())
    return idtp / n, idtp


def dlt(pts, Ps):
    A = []
    for xy, P in zip(pts, Ps):
        A.append(xy[0] * P[2] - P[0])
        A.append(xy[1] * P[2] - P[1])
    if len(A) < 4:
        return None
    _, _, Vt = np.linalg.svd(np.asarray(A))
    X = Vt[-1]
    return None if abs(X[3]) < 1e-12 else X[:3] / X[3]


def analyse(track: Path, prepared: Path, session_dir: Path, max_frames: int = 0):
    run = json.loads(track.read_text())
    meta = json.loads(prepared.read_text())
    blob = np.fromfile(meta["binary"], dtype=np.float64).reshape(-1, 2)
    names = meta["cameras"]
    A = meta["n_animals"]

    cal = sorted((session_dir / "calibration").glob("hires_cam*_params.mat"),
                 key=lambda p: int("".join(ch for ch in p.stem if ch.isdigit())))
    cams = [load_dannce_cam(p) for p in cal]
    Ps = [c["K"] @ np.hstack([c["R"], c["tvec"].reshape(3, 1)]) for c in cams]

    frames = list(zip(meta["frames"], run["frames"]))
    if max_frames:
        frames = frames[:max_frames]

    # identity id -> column, over the whole session
    ids = {}
    for _, tf in frames:
        for v in (tf.get("assignments") or {}).values():
            if v is not None and v not in ids:
                ids[v] = len(ids)

    pooled = np.zeros((len(ids), A), dtype=np.int64)
    percam = {c: np.zeros((len(ids), A), dtype=np.int64) for c in names}
    records = []

    for fr, tf in frames:
        assign = tf.get("assignments") or {}
        if not assign:
            continue
        # group members for the leave-one-out residual
        groups = {}
        for ci, cam in enumerate(names):
            for k, (true_id, off) in enumerate(fr["c"][cam]):
                ident = assign.get(f"{cam}:{k}")
                if ident is None:
                    continue
                groups.setdefault(ident, []).append((ci, blob[off], true_id, cam))

        for ident, mem in groups.items():
            row = ids[ident]
            for j, (ci, xy, true_id, cam) in enumerate(mem):
                pooled[row, true_id] += 1
                percam[cam][row, true_id] += 1
                resid = np.nan
                if len(mem) >= 3:
                    rest = [mem[i] for i in range(len(mem)) if i != j]
                    X = dlt([m[1] for m in rest], [Ps[m[0]] for m in rest])
                    if X is not None:
                        p = Ps[ci] @ np.r_[X, 1]
                        if abs(p[2]) > 1e-9:
                            resid = float(np.linalg.norm(p[:2] / p[2] - xy))
                records.append((row, true_id, resid))

    cross_idf1, cross_tp = idf1_from_counts(pooled)
    within = [idf1_from_counts(percam[c])[0] for c in names]

    # purity / coverage using the GLOBAL (cross-view) mapping, which is the one a
    # user of the 3D output actually relies on.
    r, c = linear_sum_assignment(-pooled)
    mapping = {int(rr): int(cc) for rr, cc in zip(r, c)}
    rec = np.array([(row, tid, res) for row, tid, res in records], dtype=float)
    correct = np.array([mapping.get(int(row), -1) == int(tid) for row, tid, _ in records])
    resid = rec[:, 2]

    curve = []
    have = np.isfinite(resid)
    for thr in [0.5, 1, 2, 3, 5, 7.5, 10, 15, 20, 30, 50, 75, 100, 200, 500, np.inf]:
        keep = have & (resid <= thr)
        if keep.sum() == 0:
            continue
        curve.append({"threshold_px": thr,
                      "coverage": float(keep.sum() / len(resid)),
                      "purity": float(correct[keep].mean())})

    return {
        "session": run["session"],
        "frames": len(frames),
        "detections": int(pooled.sum()),
        "n_identities": len(ids),
        "idf1_cross_view": cross_idf1,
        "idf1_within_view": float(np.mean(within)),
        "idf1_within_min": float(np.min(within)),
        "idf1_within_max": float(np.max(within)),
        "accuracy_pct": 100.0 * float(correct.mean()),
    }, pd.DataFrame(curve).assign(session=run["session"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--track-dir", type=Path,
                    default=Path("/root/vast/eric/luc3d-bench/outputs/fig8/triads"))
    ap.add_argument("--sessions-root", type=Path,
                    default=Path("/root/vast/eric/luc3d-bench/sdannce_triads"))
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--out", type=Path,
                    default=Path("/root/vast/eric/luc3d-bench/outputs/fig8/triads_idf1.csv"))
    ap.add_argument("--curve-out", type=Path,
                    default=Path("/root/vast/eric/luc3d-bench/outputs/fig8/triads_purity_curve.csv"))
    args = ap.parse_args()

    rows, curves = [], []
    for t in sorted(glob.glob(str(args.track_dir / "track_*.json"))):
        t = Path(t)
        name = t.stem.replace("track_", "")
        prep = args.track_dir / f"{name}_com.json"
        sd = args.sessions_root / name
        if not prep.exists() or not sd.exists():
            continue
        r, cv = analyse(t, prep, sd, args.max_frames)
        rows.append(r)
        curves.append(cv)
        print(f"{r['session']:24s} frames {r['frames']:6d}  det {r['detections']:8d}  "
              f"IDF1 cross {r['idf1_cross_view']:.4f}  within {r['idf1_within_view']:.4f}  "
              f"acc {r['accuracy_pct']:.2f}%  ids {r['n_identities']}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)
    pd.concat(curves).to_csv(args.curve_out, index=False)
    print(f"\nwrote {args.out}\nwrote {args.curve_out}")
    if len(df):
        w = df.detections / df.detections.sum()
        print(f"\nPOOLED  IDF1 cross-view {(df.idf1_cross_view * w).sum():.4f}   "
              f"within-view {(df.idf1_within_view * w).sum():.4f}   "
              f"accuracy {(df.accuracy_pct * w).sum():.2f}%")


if __name__ == "__main__":
    main()
