"""Appearance ReID relabeling on top of Liezl's tracker (unsupervised, Graph 1).

For each (session, camera) we keep Liezl's *set* of per-frame detection assignments but
re-derive the identity LABELS from appearance:

  1. seed one centroid per identity = mean OSNet embedding of the detections Liezl gave
     that identity (this is the "on top of Liezl" part);
  2. if the centroids are not appearance-separable, keep Liezl unchanged (never worse);
  3. otherwise run a few constrained-k-means iterations: each frame's detections are
     assigned to distinct centroids (Hungarian) by cosine similarity, then centroids are
     recomputed. This undoes identity swaps because the majority-correct frames dominate
     each centroid.

Because identity i stays anchored to "the animal Liezl labeled i most often", a no-op
relabel reproduces Liezl exactly (no spurious global flips).

Usage:
  PYTHONPATH=. reid_env/bin/python scripts/reid/relabel.py --session 10072022131531
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from itertools import permutations
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from reid.crop_detections import BENCH_CAMERAS

REPO = Path(__file__).resolve().parents[2]
MASTER = REPO / "outputs/detections_only_master_sheet.tsv"
CACHE_ROOT = REPO / "outputs/PAF_3d_kalman/_reid_cache"
LIEZL_DIR = REPO / "outputs/PAF_3d_kalman/detok_liezl_results"
OUT_DIR = REPO / "outputs/PAF_3d_kalman/detok_liezl_reid_results"

# If any identity-centroid pair is more similar than this, appearance can't separate
# them — keep Liezl unchanged for that camera.
SEP_MAX_COS = float(os.environ.get("REID_SEP_MAX", "0.96"))
# Per-boundary swap penalty for the permutation-Viterbi: higher => smoother (fewer, only
# well-supported swaps). Emission is a per-frame cosine sum, so this is in cosine units.
LAMBDA = float(os.environ.get("REID_LAMBDA", "6.0"))
N_ITERS = 3


def load_cache(session: str, cam: str, weights_tag: str):
    npz = CACHE_ROOT / weights_tag / session / f"{cam}.npz"
    if not npz.exists():
        return {}
    d = np.load(npz)
    feats = d["feats"].astype(np.float32)
    return {(int(f), int(s)): feats[i] for i, (f, s) in enumerate(zip(d["frames"], d["slots"]))}


def _seed_centroids(frames_for_cam, emb, n_animals):
    sums = np.zeros((n_animals, 512), np.float32)
    cnts = np.zeros(n_animals, np.int64)
    for fi, dets in frames_for_cam:
        for slot, lid in dets:
            v = emb.get((fi, slot))
            if v is not None and 0 <= lid < n_animals:
                sums[lid] += v; cnts[lid] += 1
    if (cnts == 0).any():
        return None
    cent = sums / cnts[:, None]
    cent /= (np.linalg.norm(cent, axis=1, keepdims=True) + 1e-12)
    return cent


def relabel_cam(frames_for_cam, emb, n_animals):
    """frames_for_cam: list of (frame_idx, [(slot, liezl_id), ...]).
    Returns dict (frame_idx, slot) -> new_id, or None to keep Liezl unchanged.

    Permutation-Viterbi: keep Liezl's labels except flip a *contiguous* region when the
    appearance evidence is persistent. State = a permutation pi mapping Liezl id -> true
    identity; det with Liezl id `lid` is scored against centroid[pi[lid]]. A swap costs
    LAMBDA once at its boundary but emission accrues every frame, so brief flicker is
    suppressed while sustained swaps survive. The identity permutation == "keep Liezl",
    and centroids are seeded from Liezl labels, so there are no spurious global flips.
    """
    cent = _seed_centroids(frames_for_cam, emb, n_animals)
    if cent is None:
        return None
    # Separability gate: if any identity pair is not appearance-separable, keep Liezl.
    G = cent @ cent.T
    if float(G[np.triu_indices(n_animals, 1)].max(initial=-1.0)) > SEP_MAX_COS:
        return None

    perms = list(permutations(range(n_animals)))
    P = len(perms)
    # transition cost matrix: LAMBDA * (number of positions that differ) between perms
    trans = np.array([[LAMBDA * sum(a != b for a, b in zip(p, q)) for q in perms]
                      for p in perms], dtype=np.float32)

    # Precompute, per frame with >=1 embedded det, the active (lid, vec) list.
    seq = []                                          # (frame_idx, [(lid, vec)])
    for fi, dets in frames_for_cam:
        active = [(lid, emb[(fi, s)]) for s, lid in dets
                  if emb.get((fi, s)) is not None and 0 <= lid < n_animals]
        if active:
            seq.append((fi, active))
    if not seq:
        return None

    for _ in range(N_ITERS):
        # Forward Viterbi over the observed frames.
        T = len(seq)
        dp = np.zeros((T, P), np.float32)
        bp = np.zeros((T, P), np.int32)
        # emission for a frame under perm p = sum cos(vec, cent[p[lid]])
        def emit(active):
            e = np.zeros(P, np.float32)
            for lid, vec in active:
                sims = cent @ vec                     # (n_animals,)
                for pi, p in enumerate(perms):
                    e[pi] += sims[p[lid]]
            return e
        dp[0] = emit(seq[0][1])
        for t in range(1, T):
            prev = dp[t - 1][:, None] - trans          # (P_prev, P_cur)
            bp[t] = np.argmax(prev, axis=0)
            dp[t] = emit(seq[t][1]) + prev[bp[t], np.arange(P)]
        # Backtrack.
        state = np.empty(T, np.int32)
        state[T - 1] = int(np.argmax(dp[T - 1]))
        for t in range(T - 2, -1, -1):
            state[t] = bp[t + 1][state[t + 1]]
        frame_perm = {seq[t][0]: perms[state[t]] for t in range(T)}

        # Recompute centroids from the corrected assignment (k-means style refine).
        sums = np.zeros((n_animals, 512), np.float32); cnts = np.zeros(n_animals, np.int64)
        for fi, active in seq:
            p = frame_perm[fi]
            for lid, vec in active:
                j = p[lid]; sums[j] += vec; cnts[j] += 1
        if (cnts == 0).any():
            break
        cent = sums / cnts[:, None]; cent /= (np.linalg.norm(cent, axis=1, keepdims=True) + 1e-12)

    # Emit corrected ids for every det; frames with no observation carry the last perm.
    new = {}
    last_perm = tuple(range(n_animals))
    obs_perm = {fi: p for fi, p in frame_perm.items()}
    for fi, dets in frames_for_cam:
        p = obs_perm.get(fi, last_perm)
        last_perm = p
        for s, lid in dets:
            new[(fi, s)] = p[lid] if 0 <= lid < n_animals else lid
    return new


def process_session(session: str, weights_tag: str, base_dir: Path = LIEZL_DIR, out_dir: Path = OUT_DIR):
    liezl_path = base_dir / f"{session}.json"
    data = json.loads(liezl_path.read_text())
    OUT = out_dir
    n_animals = int(data["numAnimals"])
    cams = data["cameras"]

    if n_animals < 2:                                 # nothing to relabel
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / f"{session}.json").write_text(json.dumps(data))
        return "copied (single-animal)"

    # Index Liezl assignments per cam: cam -> list of (frame, [(slot, id)]).
    per_cam = {c: {} for c in cams}
    for e in data["frames"]:
        fi = e["frame"]
        for key, lid in e["assignments"]:
            cam, slot = key.split(":")
            per_cam.setdefault(cam, {}).setdefault(fi, []).append((int(slot), int(lid)))

    # Compute new ids per (cam, frame, slot).
    new_ids = {}                                      # (cam, frame, slot) -> id
    stats = {}
    for cam in cams:
        emb = load_cache(session, cam, weights_tag)
        frames_for_cam = sorted(per_cam.get(cam, {}).items())
        res = relabel_cam(frames_for_cam, emb, n_animals) if emb else None
        if res is None:
            stats[cam] = "kept"
            for fi, dets in frames_for_cam:
                for slot, lid in dets:
                    new_ids[(cam, fi, slot)] = lid
        else:
            changed = 0; total = 0
            for fi, dets in frames_for_cam:
                for slot, lid in dets:
                    nid = res[(fi, slot)]
                    new_ids[(cam, fi, slot)] = nid
                    total += 1; changed += (nid != lid)
            stats[cam] = f"relabeled {changed}/{total}"

    # Rewrite assignments.
    for e in data["frames"]:
        fi = e["frame"]
        e["assignments"] = [[key, new_ids[(key.split(":")[0], fi, int(key.split(":")[1]))]]
                            for key, _ in e["assignments"]]
    data["reidWeightsTag"] = weights_tag
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{session}.json").write_text(json.dumps(data))
    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", required=True)
    ap.add_argument("--weights-tag", default="imagenet")
    ap.add_argument("--base-dir", default=str(LIEZL_DIR), help="base tracker result JSON dir to relabel")
    ap.add_argument("--out-dir", default=str(OUT_DIR), help="output result dir")
    args = ap.parse_args()
    print(args.session, process_session(args.session, args.weights_tag,
                                        Path(args.base_dir), Path(args.out_dir)))


if __name__ == "__main__":
    main()
