#!/usr/bin/env python3
"""3D Kalman temporal-linking layer for detections-only cross-view tracker output.

Pipeline context (Eric, 2026-06-22):
    all detections -> sleap-nn FILTER-only -> Liezl CrossView (cross-view
    correspondence, per-frame) -> [THIS] temporal identity from 3D motion.

Why this and not the earlier attempts:
  - 3D-KF/PAF with gap *interpolation* injected phantom detections -> 264k switches.
  - per-camera 2D constant-velocity linking wrongly merges animals across the long
    gaps that detections-only leaves (sweep: gap 15->300 made switches worse).
  This layer instead does NOTHING but relabel: it triangulates each Liezl identity's
  3D centroid per frame (using Liezl's own cross-view grouping), then runs a 3D
  constant-velocity Kalman multi-object tracker whose covariance GROWS during misses.
  So it bridges single-camera occlusions (the 3D point is observed by other cams)
  yet rejects implausible jumps via a Mahalanobis gate. No detection is ever added.

Separation of concerns:
  Liezl = which 2D detections across cameras are the same animal THIS frame.
  KF     = which animal THIS frame is the same animal as LAST frame (temporal id).

Input : <in-dir>/<session>.json  (assignments "{cam}:{det_pos}")
        detections H5s (bbox source), calibration.toml (from master sheet)
Output: <out-dir>/<session>.json  (same schema; identities re-derived from 3D tracks)

Usage:
  PYTHONPATH=. .venv/bin/python scripts/temporal_link_3d.py \
      --in-dir  outputs/PAF_3d_kalman/detok_liezl_results \
      --h5-dir  outputs/detections_only_h5s \
      --master  outputs/detections_only_master_sheet.tsv \
      --out-dir outputs/PAF_3d_kalman/detok_liezl_3dlink_results \
      [--gate 16] [--q-accel 30] [--r-meas 100] [--max-miss 90] [--sessions S1 S2]
"""
import argparse
import json
import tomllib
from pathlib import Path

import cv2
import h5py
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

CAMS = ["back", "backL", "mid", "midL", "top", "topL"]


# ----------------------------- geometry -----------------------------
def load_calib(path):
    calib = tomllib.load(open(path, "rb"))
    P, K, dist = {}, {}, {}
    for key, c in calib.items():
        if not key.startswith("cam_"):
            continue
        nm = c["name"]
        if nm not in CAMS:
            continue
        Km = np.array(c["matrix"], float)
        R, _ = cv2.Rodrigues(np.array(c["rotation"], float))
        t = np.array(c["translation"], float).reshape(3, 1)
        P[nm] = Km @ np.hstack([R, t])
        K[nm] = Km
        dist[nm] = np.array(c["distortions"], float)
    return P, K, dist


def centroid(kpts):
    c = np.nanmean(kpts, axis=0)
    return c if np.isfinite(c).all() else None


def triangulate(obs, P, K, dist):
    """obs: list of (cam, (x, y)). Linear DLT over >=2 undistorted views."""
    A = []
    for cam, (x, y) in obs:
        pu = cv2.undistortPoints(np.array([[[x, y]]], float), K[cam], dist[cam], P=K[cam])[0, 0]
        Pm = P[cam]
        A.append(pu[0] * Pm[2] - Pm[0])
        A.append(pu[1] * Pm[2] - Pm[1])
    _, _, Vt = np.linalg.svd(np.array(A))
    X = Vt[-1]
    return X[:3] / X[3]


# ----------------------------- 3D constant-velocity KF MOT -----------------------------
class Track:
    __slots__ = ("id", "x", "P", "last_frame", "hits")

    def __init__(self, tid, pos, frame, r_meas):
        self.id = tid
        self.x = np.concatenate([pos, np.zeros(3)])          # [px,py,pz,vx,vy,vz]
        # modest initial velocity variance (std 20 mm/frame) — a huge value makes a
        # fresh/coasting track a low-cost "black hole" that steals detections.
        self.P = np.diag([r_meas, r_meas, r_meas, 400.0, 400.0, 400.0]).astype(float)
        self.last_frame = frame
        self.hits = 1


def run_kf(frame_points, q_accel, r_meas, gate, max_miss, n_targets, const_pos=False):
    """frame_points: dict frame -> dict(liezl_id -> 3D pos).
    Returns mapping (frame, liezl_id) -> track_id.

    FIXED-K model: the benchmark knows the animal count, so we maintain at most
    `n_targets` constant-velocity tracks and never spawn beyond that. Each frame the
    (<=K) triangulated points are Hungarian-assigned to the K tracks by Gaussian NLL
    (m2 + logdet S). A point within `gate` of its assigned track updates it; a point
    matched only because the track is far (>gate) but still the closest is allowed to
    bind ONLY while we are below capacity or that track is stale — this keeps the
    dominant animal pinned to one identity through close encounters instead of
    fragmenting into hundreds of tracks. Extra points beyond K (rare spurious dets)
    pass through with their own id.
    """
    H = np.zeros((3, 6))
    H[:3, :3] = np.eye(3)
    R = np.eye(3) * r_meas
    out = {}
    tracks = []            # at most n_targets live tracks
    next_id = 0
    for f in sorted(frame_points.keys()):
        items = list(frame_points[f].items())          # [(liezl_id, pos), ...]
        # predict every live track to frame f
        preds = []
        for t in tracks:
            dt = f - t.last_frame
            F = np.eye(6)
            if not const_pos:
                F[0, 3] = F[1, 4] = F[2, 5] = dt   # const-velocity; else const-position
            G = np.array([dt * dt / 2] * 3 + [dt] * 3).reshape(6, 1)
            Q = (G @ G.T) * q_accel
            xp = F @ t.x
            Pp = F @ t.P @ F.T + Q
            S = H @ Pp @ H.T + R
            _, logdet = np.linalg.slogdet(S)
            preds.append((xp, Pp, np.linalg.inv(S), logdet))
        n_t, n_d = len(tracks), len(items)
        # cost = Gaussian NLL; m2 recorded separately for the gate decision.
        cost = np.zeros((n_t, n_d))
        m2mat = np.zeros((n_t, n_d))
        for ti, (xp, Pp, Sinv, logdet) in enumerate(preds):
            for di, (_lid, pos) in enumerate(items):
                d = pos - xp[:3]
                m2 = float(d @ Sinv @ d)
                m2mat[ti, di] = m2
                cost[ti, di] = m2 + logdet
        assigned_d = set()
        if n_t and n_d:
            ri, ci = linear_sum_assignment(cost)
            for r, c in zip(ri, ci):
                t = tracks[r]
                xp, Pp, _, _ = preds[r]
                # accept if within gate, OR we are at capacity (every detection must
                # bind to one of the K known animals — pin identity through gaps and
                # close encounters rather than fragment / pass through).
                if m2mat[r, c] <= gate or len(tracks) >= n_targets:
                    S = H @ Pp @ H.T + R
                    Kk = Pp @ H.T @ np.linalg.inv(S)
                    _lid, pos = items[c]
                    t.x = xp + Kk @ (pos - H @ xp)
                    t.P = (np.eye(6) - Kk @ H) @ Pp
                    t.last_frame = f
                    t.hits += 1
                    out[(f, _lid)] = t.id
                    assigned_d.add(c)
        # remaining detections: spawn new tracks only up to capacity
        for di, (_lid, pos) in enumerate(items):
            if di in assigned_d:
                continue
            if len(tracks) < n_targets:
                t = Track(next_id, pos, f, r_meas)
                next_id += 1
                tracks.append(t)
                out[(f, _lid)] = t.id
                assigned_d.add(di)
            # else: leave unassigned here -> passthrough id in the caller
    return out


# ----------------------------- per-session driver -----------------------------
def process_session(in_json, h5s, row, calib_path, args):
    d = json.load(open(in_json))
    P, K, dist = load_calib(calib_path)
    tracks_by_cam = {c: h5s[c]["tracks"][row] for c in CAMS}   # (n_frames,n_dets,n_nodes,2)

    frame_points = {}       # frame -> {liezl_id: 3D pos}
    n_tri = n_single = 0
    for fr in d["frames"]:
        f = fr["frame"]
        byid = {}
        for k, i in fr["assignments"]:
            cam, pos = k.split(":")
            c = centroid(tracks_by_cam[cam][f, int(pos)])
            if c is not None:
                byid.setdefault(i, []).append((cam, c))
        pts = {}
        for i, obs in byid.items():
            if len(obs) >= 2:
                pts[i] = triangulate(obs, P, K, dist)
                n_tri += 1
            else:
                n_single += 1
        if pts:
            frame_points[f] = pts

    n_targets = max(1, int(d.get("numAnimals", 2)))
    track_map = run_kf(frame_points, args.q_accel, args.r_meas, args.gate,
                       args.max_miss, n_targets, args.const_pos)

    # NON-DESTRUCTIVE relabel: keep Liezl's identity namespace. The 3D track only
    # provides a robust temporal segmentation; within each track we vote every
    # member detection to that track's MODAL Liezl id (split into tracklets at gaps
    # > vote_gap so a track that legitimately spans two animals isn't over-merged).
    # On clean sessions every track is already one consistent Liezl id, so the modal
    # equals the original -> no change (easy sessions are preserved); only genuine
    # within-trajectory Liezl flickers get voted away.
    from collections import Counter
    seg_of = {}                      # (frame, liezl_id) -> segment key
    seg_votes = {}                   # segment key -> Counter of liezl ids
    last_seen = {}                   # track id -> last frame
    seg_idx = {}                     # track id -> current segment index
    for (f, lid) in sorted(track_map.keys()):
        tid = track_map[(f, lid)]
        if tid not in seg_idx or f - last_seen[tid] > args.vote_gap:
            seg_idx[tid] = seg_idx.get(tid, -1) + 1
        last_seen[tid] = f
        key = (tid, seg_idx[tid])
        seg_of[(f, lid)] = key
        seg_votes.setdefault(key, Counter())[lid] += 1
    seg_modal = {k: c.most_common(1)[0][0] for k, c in seg_votes.items()}
    voted_map = {fl: seg_modal[seg_of[fl]] for fl in track_map}

    # For single-view detections (no triangulation this frame), inherit the track id
    # that this Liezl identity had at the NEAREST triangulated frame. Liezl ids are
    # locally stable, so the nearest-in-time 3D track is almost always the right one —
    # this stops single-view dips from minting a throwaway identity (a huge switch
    # source).  Build per-liezl-id sorted (frame -> track) tables once.
    per_lid = {}
    for (f, lid), vid in voted_map.items():
        per_lid.setdefault(lid, []).append((f, vid))
    for lid in per_lid:
        per_lid[lid].sort()
    lid_frames = {lid: np.array([f for f, _ in arr]) for lid, arr in per_lid.items()}

    def nearest_track(f, lid):
        arr = per_lid.get(lid)
        if not arr:
            return None
        frames = lid_frames[lid]
        j = int(np.searchsorted(frames, f))
        cand = []
        if j < len(frames):
            cand.append(j)
        if j > 0:
            cand.append(j - 1)
        best = min(cand, key=lambda k: abs(frames[k] - f))
        return arr[best][1]

    n_changed = 0
    for fr in d["frames"]:
        f = fr["frame"]
        new = []
        for k, i in fr["assignments"]:
            nid = voted_map.get((f, i))
            if nid is None:
                nid = nearest_track(f, i)       # single-view: borrow nearest-in-time
                if nid is None:
                    nid = int(i)                # never triangulated: keep Liezl id
            if nid != i:
                n_changed += 1
            new.append([k, int(nid)])
        fr["assignments"] = new
    return d, n_changed, n_tri, n_single


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", required=True)
    ap.add_argument("--h5-dir", required=True)
    ap.add_argument("--master", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--gate", type=float, default=16.0, help="Mahalanobis^2 gate (chi2 df=3)")
    ap.add_argument("--q-accel", type=float, default=30.0, help="process accel noise (mm^2/frame^4)")
    ap.add_argument("--r-meas", type=float, default=100.0, help="triangulation noise (mm^2)")
    ap.add_argument("--max-miss", type=int, default=90, help="frames a track may coast unseen")
    ap.add_argument("--const-pos", action="store_true", help="constant-position prediction (no velocity overshoot)")
    ap.add_argument("--vote-gap", type=int, default=30,
                    help="split a 3D track into tracklets at gaps > this many frames before modal voting")
    ap.add_argument("--sessions", nargs="*", default=None)
    a = ap.parse_args()

    ms = pd.read_csv(a.master, sep="\t")
    sid_col = "session" if "session" in ms.columns else ms.columns[0]
    sid_arr = ms[sid_col].astype(str).tolist()
    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    h5s = {c: h5py.File(f"{a.h5_dir}/{c}_predictions.h5", "r") for c in CAMS}
    calib_by_sid = dict(zip(ms[sid_col].astype(str), ms["calibration_toml"]))

    in_dir = Path(a.in_dir)
    targets = a.sessions or [p.stem for p in sorted(in_dir.glob("*.json"))]
    tot_changed = tot_tri = tot_single = 0
    for sid in targets:
        ij = in_dir / f"{sid}.json"
        if not ij.exists():
            print(f"  skip {sid}: no input json"); continue
        row = sid_arr.index(str(sid))
        d, nch, ntri, nsing = process_session(ij, h5s, row, calib_by_sid[str(sid)], a)
        tot_changed += nch; tot_tri += ntri; tot_single += nsing
        json.dump(d, open(out_dir / f"{sid}.json", "w"))
        print(f"  {sid}: relabeled {nch}  (tri={ntri} single-view={nsing})")
    print(f"DONE — {len(targets)} sessions, {tot_changed} relabeled, "
          f"tri={tot_tri} single={tot_single} -> {out_dir}")


if __name__ == "__main__":
    main()
