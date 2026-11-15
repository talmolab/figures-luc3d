"""Per-camera tracker baselines on the SAME identity-stripped s-DANNCE COM
detections LUC3D gets, so the cross-view claim has something to be measured against.

This is Fig 7a's construction carried onto third-party data: run an independent
tracker in each camera, then score identity BOTH ways.

  within-view IDF1   one identity mapping fitted per camera. A per-camera tracker is
                     designed for this and can score well.
  cross-view IDF1    ONE mapping over all cameras pooled, so a label has to mean the
                     same animal in every view. A tracker that labels each camera
                     independently cannot clear ~1/C here however good it is
                     within a view -- its track 0 is a different animal in different
                     cameras and one global mapping cannot satisfy them all.

Two baselines:

  simple    SLEAP's `simple` tracker with centroid similarity, reimplemented.
            Hungarian assignment of this frame's detections to recent tracks by
            centroid distance, a candidate window over the last `window` frames,
            and a distance cutoff past which a detection starts a new track.
            **This is a reimplementation, not SLEAP itself** -- no SLEAP install
            exists on this machine and the Fig 7 SLEAP baseline was precomputed
            elsewhere. For single-node COM detections SLEAP's instance similarity
            degenerates to exactly this centroid distance, so the reimplementation
            is faithful for THIS data and would not be for multi-node poses. Label
            it "centroid tracker (SLEAP-style)" on any artwork, never "SLEAP".

  bytetrack The real ByteTrack, via `supervision`, run per camera on boxes
            synthesised around each COM point (ByteTrack is a box tracker; the
            bench's own ByteTrack baseline does the same thing from pose keypoints).
            The box size is fixed, so it carries no information the point does not.
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment


# --------------------------------------------------------------------------- #
def track_simple(dets_by_frame, max_dist=120.0, window=10):
    """dets_by_frame: list over frames of (n,2) arrays. -> list of (n,) track ids."""
    out = []
    tracks = {}          # id -> (last_frame, last_xy)
    next_id = 0
    for fi, dets in enumerate(dets_by_frame):
        n = len(dets)
        ids = np.full(n, -1, dtype=int)
        live = [(tid, xy) for tid, (lf, xy) in tracks.items() if fi - lf <= window]
        if n and live:
            cost = np.zeros((n, len(live)))
            for i in range(n):
                for j, (_, xy) in enumerate(live):
                    cost[i, j] = np.linalg.norm(dets[i] - xy)
            r, c = linear_sum_assignment(cost)
            for i, j in zip(r, c):
                if cost[i, j] <= max_dist:
                    ids[i] = live[j][0]
        for i in range(n):
            if ids[i] < 0:
                ids[i] = next_id
                next_id += 1
            tracks[ids[i]] = (fi, dets[i])
        out.append(ids)
    return out


def track_bytetrack(dets_by_frame, box=60.0, fps=30):
    import supervision as sv
    tr = sv.ByteTrack(frame_rate=fps)
    out = []
    for dets in dets_by_frame:
        n = len(dets)
        if n == 0:
            out.append(np.zeros(0, dtype=int))
            continue
        xyxy = np.stack([dets[:, 0] - box / 2, dets[:, 1] - box / 2,
                         dets[:, 0] + box / 2, dets[:, 1] + box / 2], axis=1)
        d = sv.Detections(xyxy=xyxy.astype(np.float32),
                          confidence=np.ones(n, dtype=np.float32),
                          class_id=np.zeros(n, dtype=int))
        t = tr.update_with_detections(d)
        ids = np.full(n, -1, dtype=int)
        # ByteTrack may drop detections; match survivors back by box centre.
        if len(t):
            tc = np.stack([(t.xyxy[:, 0] + t.xyxy[:, 2]) / 2,
                           (t.xyxy[:, 1] + t.xyxy[:, 3]) / 2], axis=1)
            for i in range(n):
                j = int(np.argmin(np.linalg.norm(tc - dets[i], axis=1)))
                if np.linalg.norm(tc[j] - dets[i]) < 1.0:
                    ids[i] = int(t.tracker_id[j])
        out.append(ids)
    return out


# --------------------------------------------------------------------------- #
def idf1(counts):
    n = counts.sum()
    if n == 0:
        return float("nan")
    r, c = linear_sum_assignment(-counts)
    return float(counts[r, c].sum()) / n


def run_session(prepared: Path, method: str, max_frames: int = 0):
    meta = json.loads(prepared.read_text())
    blob = np.fromfile(meta["binary"], dtype=np.float64).reshape(-1, 2)
    names = meta["cameras"]
    A = meta["n_animals"]
    frames = meta["frames"][:max_frames] if max_frames else meta["frames"]

    percam_ids, percam_truth = {}, {}
    for cam in names:
        dets, truth = [], []
        for fr in frames:
            ent = fr["c"][cam]
            dets.append(np.array([blob[off] for _, off in ent]).reshape(-1, 2))
            truth.append([t for t, _ in ent])
        ids = (track_simple(dets) if method == "simple" else track_bytetrack(dets))
        percam_ids[cam] = ids
        percam_truth[cam] = truth

    # label space: raw track id, shared across cameras (a per-camera tracker's
    # "track 0" is a different animal in different views -- that is the point)
    allids = sorted({int(i) for cam in names for fr in percam_ids[cam] for i in fr if i >= 0})
    col = {t: k for k, t in enumerate(allids)}
    pooled = np.zeros((len(col), A), dtype=np.int64)
    per = {cam: np.zeros((len(col), A), dtype=np.int64) for cam in names}
    for cam in names:
        for ids, tru in zip(percam_ids[cam], percam_truth[cam]):
            for i, t in zip(ids, tru):
                if i < 0:
                    continue
                pooled[col[int(i)], t] += 1
                per[cam][col[int(i)], t] += 1
    return {
        "session": meta["session"], "method": method,
        "frames": len(frames), "detections": int(pooled.sum()),
        "n_tracks": len(col),
        "idf1_cross_view": idf1(pooled),
        "idf1_within_view": float(np.mean([idf1(per[c]) for c in names])),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--track-dir", type=Path,
                    default=Path("/root/vast/eric/luc3d-bench/outputs/fig8/triads"))
    ap.add_argument("--methods", nargs="+", default=["simple", "bytetrack"])
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--out", type=Path,
                    default=Path("/root/vast/eric/luc3d-bench/outputs/fig8/triads_baselines.csv"))
    args = ap.parse_args()

    rows = []
    for p in sorted(glob.glob(str(args.track_dir / "*_com.json"))):
        for m in args.methods:
            r = run_session(Path(p), m, args.max_frames)
            rows.append(r)
            print(f"{r['session']:24s} {m:10s} IDF1 within {r['idf1_within_view']:.4f}  "
                  f"cross {r['idf1_cross_view']:.4f}  tracks {r['n_tracks']}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)
    print(f"\nwrote {args.out}")
    print(df.groupby("method")[["idf1_within_view", "idf1_cross_view"]].mean().round(4).to_string())


if __name__ == "__main__":
    main()
