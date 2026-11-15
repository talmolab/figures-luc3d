"""Track-C variant of stitch_postprocess. Adds `tracklet_vote_reid` mode.

This is a fork of `scripts/stitch_postprocess.py` (which Track A owns); we keep
the existing strategies plus a new ReID-aware tracklet vote that augments the
3D-anchored score with a learned-embedding similarity term.

NOTE: do not modify scripts/stitch_postprocess.py while Track A is running.

Strategies (existing):

A. **flip_revert**: Find short identity flips on each (cam, trackIdx) timeline
   and revert them to the surrounding identity. Configurable maxGap (default 15)
   and max flip length (default 4). Pattern matched: X → Y...Y → X where the
   Y-run has length ≤ maxFlipLen and the X-to-X span is ≤ maxGap frames.
   This is the "extended stitcher" — same idea as the existing JS
   `stitchPerFrame` but more aggressive (longer windows, longer flip patterns).

B. **tracklet_vote**: Per (cam, trackIdx) timeline, split into tracklets at
   detection gaps > maxTrackletGap frames. Each tracklet gets a single
   identity assigned by majority vote of its per-frame entries. This is
   the min-cost-flow approximation — under the constraint that each tracklet
   must have exactly one identity, the minimum-switch assignment is the
   modal identity per tracklet. Drastically reduces switches because within
   a tracklet no per-frame changes are allowed.

C. **tracklet_vote_3d**: Same tracklet split as B, but the per-tracklet identity
   is chosen by a 3D-anchored score combining (1) per-frame vote count,
   (2) spatial agreement between each candidate identity's reprojected 3D
   centroid and the tracklet's 2D centroid in the host camera, and
   (3) velocity agreement between the identity's 3D-velocity (projected to 2D)
   and the tracklet's 2D-velocity. Picks the highest-scoring identity, NOT the
   modal one. Tunable weights `w_geom`, `w_velo` and gaussians `σ_geom`, `σ_velo`.

Usage:
    python scripts/stitch_postprocess.py --in DIR --out DIR --mode flip_revert --max-gap 15
    python scripts/stitch_postprocess.py --in DIR --out DIR --mode tracklet_vote --tracklet-gap 30
    python scripts/stitch_postprocess.py --in DIR --out DIR --mode both --max-gap 15 --tracklet-gap 30
    python scripts/stitch_postprocess.py --in DIR --out DIR --mode tracklet_vote_3d \\
        --tracklet-gap 30 --w-geom 1.0 --w-velo 0.5 --sigma-geom 50 --sigma-velo 20 \\
        --pred-h5-dir outputs/predictions_h5s \\
        --master-sheet outputs/predictions_master_sheet.tsv
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from tqdm import tqdm

# Reuse the calibration TOML parser already in this repo to avoid duplication.
# Track-C: walk up to /scripts/ to find inject_phantoms.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from inject_phantoms import parse_calibration_toml  # noqa: E402

BENCH_CAMERAS = ("back", "backL", "mid", "midL", "top", "topL")


def revert_short_flips(timelines: dict[str, list[dict]],
                       max_flip_len: int = 4, max_gap: int = 15) -> int:
    """Walk each (cam, trackIdx) timeline; if we see X then a run of ≤max_flip_len
    Ys (within max_gap frames), then X again, revert the Y-run to X.

    Returns the number of (frame, key) entries it changed.
    """
    changes = 0
    for key, entries in timelines.items():
        # entries are sorted by frame already.
        i = 0
        while i < len(entries):
            x = entries[i]
            # Look ahead for a run of length L (1..max_flip_len) with a different id,
            # then return to x.id within max_gap frames of x.frame.
            best_end = None
            for L in range(1, max_flip_len + 1):
                if i + L + 1 >= len(entries): break
                run = entries[i + 1 : i + 1 + L]
                ret = entries[i + 1 + L]
                if any(r["id"] == x["id"] for r in run): continue
                if len(set(r["id"] for r in run)) != 1: continue
                if ret["id"] != x["id"]: continue
                if ret["frame"] - x["frame"] > max_gap: continue
                best_end = (L, run)
            if best_end is not None:
                L, run = best_end
                for r in run:
                    r["id"] = x["id"]
                    r["_reverted"] = True
                    changes += 1
                i += L + 1  # skip past the now-reverted run and the returning x
            else:
                i += 1
    return changes


def tracklet_vote(timelines: dict[str, list[dict]],
                  tracklet_gap: int = 30,
                  majority_threshold: float = 0.0) -> int:
    """For each (cam, trackIdx) timeline, split into tracklets at frame gaps
    > tracklet_gap. Each tracklet's identity = mode of per-frame ids in that tracklet.

    If majority_threshold > 0, only commit the modal identity when it accounts
    for at least that fraction of the tracklet's frames. Otherwise the per-frame
    assignments are left untouched (preserves IDF1 on ambiguous tracklets).

    Returns the number of (frame, key) entries it changed.
    """
    changes = 0
    for key, entries in timelines.items():
        # Split into tracklets at gaps.
        tracklets = []
        cur = []
        for e in entries:
            if not cur or e["frame"] - cur[-1]["frame"] <= tracklet_gap:
                cur.append(e)
            else:
                tracklets.append(cur); cur = [e]
        if cur: tracklets.append(cur)

        # Assign each tracklet the modal identity.
        for trk in tracklets:
            ids = [e["id"] for e in trk]
            modal_id, modal_count = Counter(ids).most_common(1)[0]
            if majority_threshold > 0 and modal_count / max(1, len(ids)) < majority_threshold:
                continue  # ambiguous tracklet — keep per-frame as-is
            for e in trk:
                if e["id"] != modal_id:
                    e["id"] = modal_id
                    e["_voted"] = True
                    changes += 1
    return changes


def _project_3d_to_2d(point3d: np.ndarray, cam: dict) -> np.ndarray | None:
    """Project a single 3D point into camera C using cv2.projectPoints.

    Returns (2,) array or None if input has any NaN.
    """
    import cv2
    if point3d is None or np.isnan(point3d).any():
        return None
    pts3d = point3d.reshape(1, 1, 3).astype(np.float64)
    pts2d, _ = cv2.projectPoints(pts3d, cam["rotation"], cam["translation"],
                                  cam["matrix"], cam["distortions"])
    return pts2d.reshape(2)


def _identity_3d_centroid(target3d_entry: dict) -> np.ndarray | None:
    """Average non-NaN 3D keypoints; return (3,) or None if all-NaN/empty."""
    pts = target3d_entry.get("points3d")
    if not pts:
        return None
    arr = np.array([(p if p is not None else [np.nan, np.nan, np.nan]) for p in pts],
                   dtype=np.float64)
    if arr.size == 0:
        return None
    valid = ~np.isnan(arr).any(axis=1)
    if not valid.any():
        return None
    return arr[valid].mean(axis=0)


def _tracklet_2d_centroid(tracks_session: np.ndarray, frame: int, slot: int) -> np.ndarray | None:
    """Average non-NaN 2D keypoints from pred_h5 tracks[session, frame, slot].

    tracks_session: (n_frames, n_animals, n_nodes, 2) for one session.
    Returns (2,) or None.
    """
    if frame >= tracks_session.shape[0] or slot >= tracks_session.shape[1]:
        return None
    kpts = tracks_session[frame, slot]   # (n_nodes, 2)
    valid = ~np.isnan(kpts).any(axis=1)
    if not valid.any():
        return None
    return kpts[valid].mean(axis=0)


def tracklet_vote_3d(timelines: dict[str, list[dict]],
                     targets3d_by_frame: dict[int, dict[int, np.ndarray]],
                     pred_tracks_by_cam: dict[str, np.ndarray],
                     cam_by_name: dict[str, dict],
                     tracklet_gap: int = 30,
                     w_geom: float = 1.0,
                     w_velo: float = 0.5,
                     sigma_geom: float = 50.0,
                     sigma_velo: float = 20.0) -> tuple[int, dict]:
    """3D-anchored tracklet voting.

    score(T, i) = vote_count(i, T)
                + w_geom * exp(-mean_distance² / σ_geom²)
                + w_velo * exp(-mean_velo_distance² / σ_velo²)

    Arguments:
        timelines: per (cam:slot) key, list of {frame, id, pf_idx, slot_idx}.
        targets3d_by_frame: frame -> {identity_id -> (3,) centroid}.
        pred_tracks_by_cam: cam_name -> (n_frames, n_animals, n_nodes, 2).
        cam_by_name: cam_name -> calibration dict (matrix, rotation, ...).

    Returns (n_changes, stats_dict).
    """
    s2_geom = sigma_geom * sigma_geom
    s2_velo = sigma_velo * sigma_velo

    # All candidate identities seen in the entire session.
    all_ids = set()
    for fr_dict in targets3d_by_frame.values():
        all_ids.update(fr_dict.keys())
    # Also include any ids that appear in assignments (defensive).
    for entries in timelines.values():
        for e in entries:
            all_ids.add(e["id"])
    all_ids = sorted(all_ids)

    changes = 0
    stats = {"tracklets": 0, "tracklets_changed_to_nonmodal": 0,
             "tracklets_kept_modal": 0, "frames_changed": 0,
             "skipped_no_cam_calib": 0, "skipped_no_pred_h5": 0}

    for key, entries in timelines.items():
        cam_name, slot_str = key.rsplit(":", 1)
        slot = int(slot_str)
        cam = cam_by_name.get(cam_name)
        pred_tracks = pred_tracks_by_cam.get(cam_name)
        if cam is None:
            stats["skipped_no_cam_calib"] += 1
            continue
        if pred_tracks is None:
            stats["skipped_no_pred_h5"] += 1
            continue

        # Split into tracklets at gaps.
        tracklets = []
        cur = []
        for e in entries:
            if not cur or e["frame"] - cur[-1]["frame"] <= tracklet_gap:
                cur.append(e)
            else:
                tracklets.append(cur); cur = [e]
        if cur: tracklets.append(cur)

        for trk in tracklets:
            stats["tracklets"] += 1
            n = len(trk)

            # Per-frame 2D centroid for this tracklet.
            frames_T = [e["frame"] for e in trk]
            tracklet_2d = []
            for f in frames_T:
                c2 = _tracklet_2d_centroid(pred_tracks, f, slot)
                tracklet_2d.append(c2)

            # Per-frame, per-identity 2D-projected 3D centroid.
            # Build a dict id -> list of (frame, proj_2d or None).
            id_proj_2d: dict[int, list[np.ndarray | None]] = {i: [None] * n for i in all_ids}
            for fi, f in enumerate(frames_T):
                fr_targets = targets3d_by_frame.get(f, {})
                for ident, centroid_3d in fr_targets.items():
                    if ident not in id_proj_2d:
                        continue
                    p2 = _project_3d_to_2d(centroid_3d, cam)
                    id_proj_2d[ident][fi] = p2

            # Per-frame tracklet 2D velocity (between consecutive in-tracklet frames).
            tracklet_2d_vel: list[np.ndarray | None] = [None] * n
            for fi in range(1, n):
                a = tracklet_2d[fi - 1]
                b = tracklet_2d[fi]
                if a is None or b is None:
                    continue
                dt = max(1, frames_T[fi] - frames_T[fi - 1])
                tracklet_2d_vel[fi] = (b - a) / dt

            # Vote counts.
            vote_counts: dict[int, int] = Counter(e["id"] for e in trk)

            # Now score each candidate identity.
            best_id = None
            best_score = -math.inf
            modal_id, modal_count = vote_counts.most_common(1)[0]
            for ident in all_ids:
                vc = vote_counts.get(ident, 0)

                # Spatial fit: mean Euclidean distance between projected 3D centroid
                # and the tracklet 2D centroid, where both are available.
                dists = []
                vel_dists = []
                for fi in range(n):
                    p2 = id_proj_2d[ident][fi]
                    t2 = tracklet_2d[fi]
                    if p2 is None or t2 is None:
                        continue
                    if np.isnan(p2).any() or np.isnan(t2).any():
                        continue
                    dists.append(float(np.linalg.norm(p2 - t2)))

                    if fi >= 1:
                        p2_prev = id_proj_2d[ident][fi - 1]
                        tv = tracklet_2d_vel[fi]
                        if p2_prev is None or tv is None:
                            continue
                        if np.isnan(p2_prev).any():
                            continue
                        dt = max(1, frames_T[fi] - frames_T[fi - 1])
                        id_vel = (p2 - p2_prev) / dt
                        vel_dists.append(float(np.linalg.norm(id_vel - tv)))

                if dists:
                    mean_d = float(np.mean(dists))
                    spatial_fit = math.exp(-(mean_d * mean_d) / s2_geom)
                else:
                    spatial_fit = 0.0   # no evidence either way

                if vel_dists:
                    mean_vd = float(np.mean(vel_dists))
                    velocity_match = math.exp(-(mean_vd * mean_vd) / s2_velo)
                else:
                    velocity_match = 0.0

                score = vc + w_geom * spatial_fit + w_velo * velocity_match
                if score > best_score:
                    best_score = score
                    best_id = ident

            if best_id is None:
                best_id = modal_id

            if best_id == modal_id:
                stats["tracklets_kept_modal"] += 1
            else:
                stats["tracklets_changed_to_nonmodal"] += 1

            for e in trk:
                if e["id"] != best_id:
                    e["id"] = best_id
                    e["_voted3d"] = True
                    changes += 1
                    stats["frames_changed"] += 1

    return changes, stats


def tracklet_vote_reid(
    timelines: dict[str, list[dict]],
    targets3d_by_frame: dict[int, dict[int, np.ndarray]],
    pred_tracks_by_cam: dict[str, np.ndarray],
    cam_by_name: dict[str, dict],
    embeddings: np.ndarray,          # (n_frames_emb, n_ids_emb, d_embed)
    valid_mask: np.ndarray,          # (n_frames_emb, n_ids_emb) bool
    frame_to_emb_row: dict[int, int],
    id_to_emb_col: dict[int, int],
    tracklet_gap: int = 30,
    w_vote: float = 1.0,
    w_geom: float = 1.0,
    w_velo: float = 0.5,
    w_reid: float = 1.0,
    sigma_geom: float = 50.0,
    sigma_velo: float = 20.0,
    reid_k_frames: int = 30,
) -> tuple[int, dict]:
    """3D + ReID anchored tracklet voting.

    score(T, i) = w_vote * vote_count(i, T)
                + w_geom * spatial_fit(i, T)
                + w_velo * velocity_match(i, T)
                + w_reid * reid_similarity(i, T)

    reid_similarity(i, T) = mean cosine similarity between the identity's pose-history
    embeddings in a window of `reid_k_frames` ending at the tracklet's first frame
    (i.e. the identity *before* the tracklet started) and the tracklet's own
    embeddings during its first `reid_k_frames` frames. If neither is available it
    falls back to 0.
    """
    s2_geom = sigma_geom * sigma_geom
    s2_velo = sigma_velo * sigma_velo

    all_ids = set()
    for fr_dict in targets3d_by_frame.values():
        all_ids.update(fr_dict.keys())
    for entries in timelines.values():
        for e in entries:
            all_ids.add(e["id"])
    all_ids = sorted(all_ids)

    changes = 0
    stats = {"tracklets": 0, "tracklets_changed_to_nonmodal": 0,
             "tracklets_kept_modal": 0, "frames_changed": 0,
             "skipped_no_cam_calib": 0, "skipped_no_pred_h5": 0,
             "tracklets_with_reid": 0,
             "reid_changed_decision": 0}

    # Lookup helpers
    def _emb_window(ident: int, frame_start: int, frame_end: int) -> np.ndarray | None:
        """Mean embedding of identity `ident` over frames [frame_start, frame_end)."""
        col = id_to_emb_col.get(ident)
        if col is None:
            return None
        rows = []
        for f in range(frame_start, frame_end):
            r = frame_to_emb_row.get(f)
            if r is None:
                continue
            if valid_mask[r, col]:
                rows.append(embeddings[r, col])
        if not rows:
            return None
        v = np.mean(rows, axis=0)
        n = np.linalg.norm(v)
        if n < 1e-6:
            return None
        return v / n

    for key, entries in timelines.items():
        cam_name, slot_str = key.rsplit(":", 1)
        slot = int(slot_str)
        cam = cam_by_name.get(cam_name)
        pred_tracks = pred_tracks_by_cam.get(cam_name)
        if cam is None:
            stats["skipped_no_cam_calib"] += 1
            continue
        if pred_tracks is None:
            stats["skipped_no_pred_h5"] += 1
            continue

        tracklets = []
        cur = []
        for e in entries:
            if not cur or e["frame"] - cur[-1]["frame"] <= tracklet_gap:
                cur.append(e)
            else:
                tracklets.append(cur); cur = [e]
        if cur:
            tracklets.append(cur)

        for trk in tracklets:
            stats["tracklets"] += 1
            n = len(trk)
            frames_T = [e["frame"] for e in trk]
            tracklet_2d = []
            for f in frames_T:
                c2 = _tracklet_2d_centroid(pred_tracks, f, slot)
                tracklet_2d.append(c2)

            id_proj_2d: dict[int, list[np.ndarray | None]] = {i: [None] * n for i in all_ids}
            for fi, f in enumerate(frames_T):
                fr_targets = targets3d_by_frame.get(f, {})
                for ident, centroid_3d in fr_targets.items():
                    if ident not in id_proj_2d:
                        continue
                    p2 = _project_3d_to_2d(centroid_3d, cam)
                    id_proj_2d[ident][fi] = p2

            tracklet_2d_vel: list[np.ndarray | None] = [None] * n
            for fi in range(1, n):
                a = tracklet_2d[fi - 1]
                b = tracklet_2d[fi]
                if a is None or b is None:
                    continue
                dt = max(1, frames_T[fi] - frames_T[fi - 1])
                tracklet_2d_vel[fi] = (b - a) / dt

            vote_counts: dict[int, int] = Counter(e["id"] for e in trk)
            modal_id, modal_count = vote_counts.most_common(1)[0]

            # ReID similarity: continuity score per candidate identity.
            # For each candidate id, compare:
            #   * its pose-history embedding during the K frames BEFORE the tracklet
            #   * its pose-history embedding during the K frames AT the start of the tracklet
            # If the identity's 3D pose history is continuous across the tracklet
            # boundary, these will match closely (similarity -> 1). If LUC3D's
            # `targets3d` for this id is internally inconsistent through the tracklet
            # boundary (a sign that the id is incorrectly placed), the similarity
            # drops. This term should favour identities whose 3D pose history is
            # geometrically consistent with the tracklet's frames.
            #
            # NOTE: the embedding is a function of the 3D pose history only. It does
            # NOT encode which 2D-detection slot the points came from. So the only
            # signal is whether the candidate's 3D trajectory looks like the same
            # mouse on both sides of the tracklet boundary.
            trk_start = frames_T[0]
            trk_end_for_self = min(frames_T[-1] + 1, trk_start + reid_k_frames)
            cand_start = max(0, trk_start - reid_k_frames)
            cand_end = trk_start

            reid_similarities: dict[int, float] = {}
            have_reid = False
            for ident in all_ids:
                before_emb = _emb_window(ident, cand_start, cand_end)
                during_emb = _emb_window(ident, trk_start, trk_end_for_self)
                if before_emb is None or during_emb is None:
                    reid_similarities[ident] = 0.0
                    continue
                have_reid = True
                # cosine similarity rescaled to [0, 1]
                reid_similarities[ident] = (float(before_emb @ during_emb) + 1.0) / 2.0

            if have_reid:
                stats["tracklets_with_reid"] += 1
            # All-zero else case is fine; the term contributes nothing.

            best_id = None
            best_score = -math.inf
            for ident in all_ids:
                vc = vote_counts.get(ident, 0)

                dists = []
                vel_dists = []
                for fi in range(n):
                    p2 = id_proj_2d[ident][fi]
                    t2 = tracklet_2d[fi]
                    if p2 is None or t2 is None:
                        continue
                    if np.isnan(p2).any() or np.isnan(t2).any():
                        continue
                    dists.append(float(np.linalg.norm(p2 - t2)))
                    if fi >= 1:
                        p2_prev = id_proj_2d[ident][fi - 1]
                        tv = tracklet_2d_vel[fi]
                        if p2_prev is None or tv is None:
                            continue
                        if np.isnan(p2_prev).any():
                            continue
                        dt = max(1, frames_T[fi] - frames_T[fi - 1])
                        id_vel = (p2 - p2_prev) / dt
                        vel_dists.append(float(np.linalg.norm(id_vel - tv)))

                if dists:
                    mean_d = float(np.mean(dists))
                    spatial_fit = math.exp(-(mean_d * mean_d) / s2_geom)
                else:
                    spatial_fit = 0.0
                if vel_dists:
                    mean_vd = float(np.mean(vel_dists))
                    velocity_match = math.exp(-(mean_vd * mean_vd) / s2_velo)
                else:
                    velocity_match = 0.0

                # Normalise vote count so all terms are in [0, 1].
                vote_frac = vc / max(1, n)
                score = (
                    w_vote * vote_frac
                    + w_geom * spatial_fit
                    + w_velo * velocity_match
                    + w_reid * reid_similarities[ident]
                )
                if score > best_score:
                    best_score = score
                    best_id = ident

            # Diagnostic: what was the decision without the ReID term?
            without_reid_best = None
            without_reid_score = -math.inf
            for ident in all_ids:
                vc = vote_counts.get(ident, 0)
                vote_frac = vc / max(1, n)
                dists2 = []
                for fi in range(n):
                    p2 = id_proj_2d[ident][fi]; t2 = tracklet_2d[fi]
                    if p2 is None or t2 is None: continue
                    if np.isnan(p2).any() or np.isnan(t2).any(): continue
                    dists2.append(float(np.linalg.norm(p2 - t2)))
                spatial_fit = math.exp(-(float(np.mean(dists2)) ** 2) / s2_geom) if dists2 else 0.0
                score2 = w_vote * vote_frac + w_geom * spatial_fit
                if score2 > without_reid_score:
                    without_reid_score = score2
                    without_reid_best = ident
            if best_id != without_reid_best and have_reid:
                stats["reid_changed_decision"] += 1

            if best_id is None:
                best_id = modal_id
            if best_id == modal_id:
                stats["tracklets_kept_modal"] += 1
            else:
                stats["tracklets_changed_to_nonmodal"] += 1

            for e in trk:
                if e["id"] != best_id:
                    e["id"] = best_id
                    e["_voted_reid"] = True
                    changes += 1
                    stats["frames_changed"] += 1

    return changes, stats


def build_targets3d_by_frame(d: dict) -> dict[int, dict[int, np.ndarray]]:
    """Extract per-frame {identity_id -> 3D centroid (3,)} from LUC3D JSON."""
    out: dict[int, dict[int, np.ndarray]] = {}
    for fr in d["frames"]:
        per_id = {}
        for t in fr.get("targets3d", []) or []:
            ident = t.get("identityId")
            if ident is None:
                continue
            c = _identity_3d_centroid(t)
            if c is None:
                continue
            per_id[int(ident)] = c
        out[int(fr["frame"])] = per_id
    return out


def process_session_3d(json_path: Path, out_path: Path,
                       tracklet_gap: int,
                       w_geom: float, w_velo: float,
                       sigma_geom: float, sigma_velo: float,
                       pred_tracks_by_cam: dict[str, np.ndarray],
                       cam_by_name: dict[str, dict]) -> dict:
    """Run tracklet_vote_3d on a single session JSON and write the output."""
    d = json.loads(json_path.read_text())

    timelines: dict[str, list[dict]] = defaultdict(list)
    for pf_idx, fr in enumerate(d["frames"]):
        for slot_idx, (key, ident) in enumerate(fr["assignments"]):
            timelines[key].append({"frame": fr["frame"], "id": ident,
                                    "pf_idx": pf_idx, "slot_idx": slot_idx})
    for k in timelines:
        timelines[k].sort(key=lambda e: e["frame"])

    targets3d_by_frame = build_targets3d_by_frame(d)

    changes, stats = tracklet_vote_3d(
        timelines, targets3d_by_frame, pred_tracks_by_cam, cam_by_name,
        tracklet_gap=tracklet_gap,
        w_geom=w_geom, w_velo=w_velo,
        sigma_geom=sigma_geom, sigma_velo=sigma_velo,
    )

    if changes > 0:
        new_frames = [dict(fr, assignments=[list(a) for a in fr["assignments"]])
                      for fr in d["frames"]]
        for key, entries in timelines.items():
            for e in entries:
                new_frames[e["pf_idx"]]["assignments"][e["slot_idx"]] = [key, e["id"]]
        d["frames"] = new_frames

    d["tracklet3dStats"] = stats

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(d))

    return {
        "session": json_path.stem,
        "voted3d_changes": changes,
        "n_frames": len(d["frames"]),
        **stats,
    }


def process_session_reid(json_path: Path, out_path: Path,
                         tracklet_gap: int,
                         w_vote: float, w_geom: float, w_velo: float, w_reid: float,
                         sigma_geom: float, sigma_velo: float,
                         reid_k_frames: int,
                         pred_tracks_by_cam: dict[str, np.ndarray],
                         cam_by_name: dict[str, dict],
                         embeddings: np.ndarray, valid_mask: np.ndarray,
                         frame_to_emb_row: dict[int, int],
                         id_to_emb_col: dict[int, int]) -> dict:
    d = json.loads(json_path.read_text())
    timelines: dict[str, list[dict]] = defaultdict(list)
    for pf_idx, fr in enumerate(d["frames"]):
        for slot_idx, (key, ident) in enumerate(fr["assignments"]):
            timelines[key].append({"frame": fr["frame"], "id": ident,
                                    "pf_idx": pf_idx, "slot_idx": slot_idx})
    for k in timelines:
        timelines[k].sort(key=lambda e: e["frame"])

    targets3d_by_frame = build_targets3d_by_frame(d)

    changes, stats = tracklet_vote_reid(
        timelines, targets3d_by_frame, pred_tracks_by_cam, cam_by_name,
        embeddings=embeddings, valid_mask=valid_mask,
        frame_to_emb_row=frame_to_emb_row, id_to_emb_col=id_to_emb_col,
        tracklet_gap=tracklet_gap,
        w_vote=w_vote, w_geom=w_geom, w_velo=w_velo, w_reid=w_reid,
        sigma_geom=sigma_geom, sigma_velo=sigma_velo,
        reid_k_frames=reid_k_frames,
    )

    if changes > 0:
        new_frames = [dict(fr, assignments=[list(a) for a in fr["assignments"]])
                      for fr in d["frames"]]
        for key, entries in timelines.items():
            for e in entries:
                new_frames[e["pf_idx"]]["assignments"][e["slot_idx"]] = [key, e["id"]]
        d["frames"] = new_frames

    d["trackletReidStats"] = stats

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(d))

    return {
        "session": json_path.stem,
        "voted_reid_changes": changes,
        "n_frames": len(d["frames"]),
        **stats,
    }


def process_session(json_path: Path, mode: str, max_flip_len: int, max_gap: int,
                    tracklet_gap: int, out_path: Path,
                    majority_threshold: float = 0.0) -> dict:
    d = json.loads(json_path.read_text())

    # Build per-key timelines: list of {frame, id, pf_idx, slot_idx}
    timelines: dict[str, list[dict]] = defaultdict(list)
    for pf_idx, fr in enumerate(d["frames"]):
        for slot_idx, (key, ident) in enumerate(fr["assignments"]):
            timelines[key].append({"frame": fr["frame"], "id": ident,
                                    "pf_idx": pf_idx, "slot_idx": slot_idx})

    # Sort each timeline by frame (they should already be sorted but be safe).
    for k in timelines:
        timelines[k].sort(key=lambda e: e["frame"])

    flip_changes = 0
    voted_changes = 0
    if mode in ("flip_revert", "both"):
        flip_changes = revert_short_flips(timelines, max_flip_len, max_gap)
    if mode in ("tracklet_vote", "both"):
        voted_changes = tracklet_vote(timelines, tracklet_gap, majority_threshold)

    # Write the updated identities back into d["frames"][pf_idx]["assignments"].
    if flip_changes + voted_changes > 0:
        # Rebuild assignments per frame from the timelines.
        new_frames = [dict(fr, assignments=[list(a) for a in fr["assignments"]])
                      for fr in d["frames"]]
        for key, entries in timelines.items():
            for e in entries:
                new_frames[e["pf_idx"]]["assignments"][e["slot_idx"]] = [key, e["id"]]
        d["frames"] = new_frames

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(d))

    return {
        "session": json_path.stem,
        "flip_changes": flip_changes,
        "voted_changes": voted_changes,
        "n_frames": len(d["frames"]),
    }


def _load_master_sheet(path: str):
    import pandas as pd
    df = pd.read_csv(path, sep="\t", index_col=0)
    df["session"] = df["session"].astype(str)
    return df


def _run_tracklet_3d(args):
    """Run the 3D-anchored tracklet vote mode. Loads pred_h5 and calibrations per session."""
    import h5py
    import pandas as pd

    in_dir = Path(args.in_dir); out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_files = sorted(in_dir.glob("*.json"))
    print(f"[tracklet_vote_3d] processing {len(json_files)} sessions from {in_dir} → {out_dir}")
    print(f"  tracklet_gap={args.tracklet_gap}  w_geom={args.w_geom}  w_velo={args.w_velo}"
          f"  sigma_geom={args.sigma_geom}  sigma_velo={args.sigma_velo}")

    master = _load_master_sheet(args.master_sheet)
    sess_to_row = {row["session"]: (i, row) for i, row in master.iterrows()}
    cams_list = list(BENCH_CAMERAS)

    # Pre-load all pred_h5 datasets once.
    print(f"[tracklet_vote_3d] loading pred_h5 from {args.pred_h5_dir}")
    pred_h5_full: dict[str, np.ndarray] = {}
    for cam in cams_list:
        p = Path(args.pred_h5_dir) / f"{cam}_predictions.h5"
        if not p.exists():
            print(f"  WARN: missing {p}")
            continue
        with h5py.File(p, "r") as f:
            pred_h5_full[cam] = f["tracks"][...]
        print(f"  loaded {cam}: shape={pred_h5_full[cam].shape}")

    # Cache calibrations by toml path (many sessions reuse the same one).
    calib_cache: dict[str, dict[str, dict]] = {}

    total_voted = 0
    total_changed_to_nonmodal = 0
    total_kept_modal = 0
    total_tracklets = 0
    progress_log = args.progress_log
    if progress_log:
        from datetime import datetime
        Path(progress_log).parent.mkdir(parents=True, exist_ok=True)

    def log(msg: str):
        from datetime import datetime
        line = f"{datetime.now().isoformat(timespec='seconds')} {msg}"
        print(line)
        if progress_log:
            with open(progress_log, "a") as f:
                f.write(line + "\n")

    log(f"[tracklet_vote_3d] BEGIN config w_geom={args.w_geom} w_velo={args.w_velo} "
        f"sigma_geom={args.sigma_geom} sigma_velo={args.sigma_velo} gap={args.tracklet_gap} "
        f"out={out_dir}")

    for jp in tqdm(json_files, desc="sessions"):
        sess_id = jp.stem
        if sess_id not in sess_to_row:
            log(f"  skip {sess_id}: not in master sheet")
            continue
        sess_idx, row = sess_to_row[sess_id]
        calib_path = str(row["calibration_toml"])
        if calib_path not in calib_cache:
            cams = parse_calibration_toml(Path(calib_path).read_text())
            calib_cache[calib_path] = {c["name"]: c for c in cams}
        cam_by_name = calib_cache[calib_path]

        # Slice pred_h5 for this session.
        pred_tracks_by_cam = {}
        for cam, arr in pred_h5_full.items():
            pred_tracks_by_cam[cam] = arr[int(sess_idx)]  # (n_frames, n_animals, n_nodes, 2)

        info = process_session_3d(
            jp, out_dir / jp.name,
            tracklet_gap=args.tracklet_gap,
            w_geom=args.w_geom, w_velo=args.w_velo,
            sigma_geom=args.sigma_geom, sigma_velo=args.sigma_velo,
            pred_tracks_by_cam=pred_tracks_by_cam,
            cam_by_name=cam_by_name,
        )
        total_voted += info["voted3d_changes"]
        total_changed_to_nonmodal += info.get("tracklets_changed_to_nonmodal", 0)
        total_kept_modal += info.get("tracklets_kept_modal", 0)
        total_tracklets += info.get("tracklets", 0)

    log(f"[tracklet_vote_3d] DONE total_tracklets={total_tracklets} "
        f"changed_to_nonmodal={total_changed_to_nonmodal} kept_modal={total_kept_modal} "
        f"voted3d_changes={total_voted}")


def _run_tracklet_reid(args):
    """Run the ReID-augmented tracklet vote mode. Same as 3D but also loads embeddings."""
    import h5py
    import pandas as pd

    in_dir = Path(args.in_dir); out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_files = sorted(in_dir.glob("*.json"))
    print(f"[tracklet_vote_reid] processing {len(json_files)} sessions from {in_dir} -> {out_dir}")
    print(f"  w_vote={args.w_vote}  w_geom={args.w_geom}  w_velo={args.w_velo}  w_reid={args.w_reid}")
    print(f"  sigma_geom={args.sigma_geom}  sigma_velo={args.sigma_velo}  reid_k_frames={args.reid_k_frames}")
    print(f"  embeddings_dir={args.embeddings_dir}")

    master = _load_master_sheet(args.master_sheet)
    sess_to_row = {row["session"]: (i, row) for i, row in master.iterrows()}
    cams_list = list(BENCH_CAMERAS)

    print(f"[tracklet_vote_reid] loading pred_h5 from {args.pred_h5_dir}")
    pred_h5_full: dict[str, np.ndarray] = {}
    for cam in cams_list:
        p = Path(args.pred_h5_dir) / f"{cam}_predictions.h5"
        if not p.exists():
            print(f"  WARN: missing {p}")
            continue
        with h5py.File(p, "r") as f:
            pred_h5_full[cam] = f["tracks"][...]

    calib_cache: dict[str, dict[str, dict]] = {}

    progress_log = args.progress_log
    if progress_log:
        Path(progress_log).parent.mkdir(parents=True, exist_ok=True)

    def log(msg: str):
        from datetime import datetime
        line = f"{datetime.now().isoformat(timespec='seconds')} {msg}"
        print(line)
        if progress_log:
            with open(progress_log, "a") as f:
                f.write(line + "\n")

    log(f"[tracklet_vote_reid] BEGIN config w_vote={args.w_vote} w_geom={args.w_geom} "
        f"w_velo={args.w_velo} w_reid={args.w_reid} sigma_geom={args.sigma_geom} "
        f"sigma_velo={args.sigma_velo} gap={args.tracklet_gap} k={args.reid_k_frames} "
        f"out={out_dir}")

    total_tracklets = 0
    total_changes = 0
    total_reid_dec_changed = 0
    total_with_reid = 0
    for jp in tqdm(json_files, desc="sessions"):
        sess_id = jp.stem
        if sess_id not in sess_to_row:
            log(f"  skip {sess_id}: not in master sheet")
            continue
        sess_idx, row = sess_to_row[sess_id]

        emb_path = Path(args.embeddings_dir) / (sess_id + ".npz")
        if not emb_path.exists():
            log(f"  skip {sess_id}: missing {emb_path}")
            continue
        e = np.load(emb_path)
        embeddings = e["embeddings"]
        valid_mask = e["valid_mask"]
        emb_frames = e["frames"]
        emb_ids = e["ids"]
        frame_to_emb_row = {int(f): i for i, f in enumerate(emb_frames)}
        id_to_emb_col = {int(i): c for c, i in enumerate(emb_ids)}

        calib_path = str(row["calibration_toml"])
        if calib_path not in calib_cache:
            cams = parse_calibration_toml(Path(calib_path).read_text())
            calib_cache[calib_path] = {c["name"]: c for c in cams}
        cam_by_name = calib_cache[calib_path]

        pred_tracks_by_cam = {}
        for cam, arr in pred_h5_full.items():
            pred_tracks_by_cam[cam] = arr[int(sess_idx)]

        info = process_session_reid(
            jp, out_dir / jp.name,
            tracklet_gap=args.tracklet_gap,
            w_vote=args.w_vote, w_geom=args.w_geom, w_velo=args.w_velo, w_reid=args.w_reid,
            sigma_geom=args.sigma_geom, sigma_velo=args.sigma_velo,
            reid_k_frames=args.reid_k_frames,
            pred_tracks_by_cam=pred_tracks_by_cam,
            cam_by_name=cam_by_name,
            embeddings=embeddings, valid_mask=valid_mask,
            frame_to_emb_row=frame_to_emb_row, id_to_emb_col=id_to_emb_col,
        )
        total_tracklets += info.get("tracklets", 0)
        total_changes += info["voted_reid_changes"]
        total_reid_dec_changed += info.get("reid_changed_decision", 0)
        total_with_reid += info.get("tracklets_with_reid", 0)

    log(f"[tracklet_vote_reid] DONE total_tracklets={total_tracklets} "
        f"with_reid={total_with_reid} reid_changed_decision={total_reid_dec_changed} "
        f"voted_reid_changes={total_changes}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_dir", required=True)
    ap.add_argument("--out", dest="out_dir", required=True)
    ap.add_argument("--mode", choices=["flip_revert", "tracklet_vote", "tracklet_vote_3d",
                                       "tracklet_vote_reid", "both"],
                    default="both")
    ap.add_argument("--max-flip-len", type=int, default=4,
                    help="max length of the Y-run to revert (e.g. X→YY→X is len 2)")
    ap.add_argument("--max-gap", type=int, default=15,
                    help="max frame gap between the two Xs in X→Y...→X")
    ap.add_argument("--tracklet-gap", type=int, default=30,
                    help="frame gap that splits one (cam,trackIdx) timeline into tracklets")
    ap.add_argument("--majority-threshold", type=float, default=0.0,
                    help="only commit modal if it has >= this fraction of tracklet frames")
    # tracklet_vote_3d-specific knobs:
    ap.add_argument("--w-geom", type=float, default=1.0,
                    help="weight on the spatial Gaussian fit term in tracklet_vote_3d")
    ap.add_argument("--w-velo", type=float, default=0.5,
                    help="weight on the velocity match term in tracklet_vote_3d")
    ap.add_argument("--sigma-geom", type=float, default=50.0,
                    help="px std for spatial Gaussian in tracklet_vote_3d")
    ap.add_argument("--sigma-velo", type=float, default=20.0,
                    help="px/frame std for velocity Gaussian in tracklet_vote_3d")
    ap.add_argument("--pred-h5-dir", default="outputs/predictions_h5s",
                    help="dir holding {cam}_predictions.h5 — only used by tracklet_vote_3d")
    ap.add_argument("--master-sheet", default="outputs/predictions_master_sheet.tsv",
                    help="master TSV with calibration_toml column")
    ap.add_argument("--progress-log", default=None,
                    help="optional path to append progress messages")
    # tracklet_vote_reid-specific knobs:
    ap.add_argument("--w-vote", type=float, default=1.0,
                    help="weight on the vote count term in tracklet_vote_reid")
    ap.add_argument("--w-reid", type=float, default=1.0,
                    help="weight on the ReID similarity term in tracklet_vote_reid")
    ap.add_argument("--reid-k-frames", type=int, default=30,
                    help="number of frames before/at tracklet start to aggregate ReID embeddings")
    ap.add_argument("--embeddings-dir", default=None,
                    help="dir of per-session npz embeddings, only used by tracklet_vote_reid")
    args = ap.parse_args()

    if args.mode == "tracklet_vote_3d":
        _run_tracklet_3d(args)
        return
    if args.mode == "tracklet_vote_reid":
        assert args.embeddings_dir, "--embeddings-dir is required for tracklet_vote_reid"
        _run_tracklet_reid(args)
        return

    in_dir = Path(args.in_dir); out_dir = Path(args.out_dir)
    json_files = sorted(in_dir.glob("*.json"))
    print(f"processing {len(json_files)} sessions from {in_dir} → {out_dir} "
          f"(mode={args.mode}, max_flip_len={args.max_flip_len}, max_gap={args.max_gap}, "
          f"tracklet_gap={args.tracklet_gap})")

    total_flip = 0
    total_voted = 0
    for jp in tqdm(json_files):
        info = process_session(jp, args.mode, args.max_flip_len, args.max_gap,
                                args.tracklet_gap, out_dir / jp.name,
                                majority_threshold=args.majority_threshold)
        total_flip += info["flip_changes"]
        total_voted += info["voted_changes"]

    print(f"\ntotal flip-reverts: {total_flip}")
    print(f"total tracklet-votes: {total_voted}")
    print(f"wrote {len(json_files)} JSONs to {out_dir}")


if __name__ == "__main__":
    main()
