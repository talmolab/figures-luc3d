"""Phantom-reprojection injection — the "smart" temporal-PAF method for missing instances.

For each frame f and each known identity i:
  1. Take the last K (default 3) observed 3D poses for identity i from targets3d.points3d.
  2. Compute a constant-velocity prediction of i's 3D pose at frame f.
     (Per-keypoint extrapolation. Drops keypoints that don't have a usable history.)
  3. Reproject the predicted 3D pose into every camera in which i is currently missing.
  4. Write the reprojected 2D pose into a previously-NaN slot in pred_h5
     (NEW H5 dir, doesn't touch the original). Add the corresponding identity
     assignment to a NEW luc3d_json.

The phantom carries the temporal PAF flow integral: its 2D position came from the
KF-like prediction. Its confidence is bounded by how recently it was observed and
how stable the velocity estimate is. Phantoms older than `maxPhantomAge` frames
are dropped — we don't fabricate detections far past the last real observation.

Inputs:
    --luc3d-dir       outputs/PAF_3d_kalman/luc3d_l1_only_results
    --pred-h5-dir     outputs/predictions_h5s (original)
    --master-sheet    outputs/predictions_master_sheet.tsv
Outputs:
    --out-pred-h5-dir outputs/PAF_3d_kalman/luc3d_paf_phantom_predictions_h5s
    --out-luc3d-dir   outputs/PAF_3d_kalman/luc3d_paf_phantom_results

After running this, eval with:
    evaluate.py --luc3d-dir <out-luc3d-dir> --pred-h5-dir <out-pred-h5-dir>
(needs evaluate.py --pred-h5-dir flag added)
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from collections import defaultdict

import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Calibration TOML parser (mirrors lucid/file-io.js)
# ---------------------------------------------------------------------------
def parse_calibration_toml(text: str) -> list[dict]:
    """Return list of {name, matrix, distortions, rotation, translation, size}."""
    import re
    cameras = []
    section_re = re.compile(r"^\[([^\]]+)\]\s*$", re.MULTILINE)
    sections = [(m.group(1), m.end()) for m in section_re.finditer(text)]
    starts = [s for _, s in sections]
    for i, (name, start) in enumerate(sections):
        if not name.startswith("cam_"):
            continue
        end = starts[i + 1] if i + 1 < len(sections) else len(text)
        body = text[start:end]
        props = {}
        for line in body.split("\n"):
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("["):
                continue
            if "=" not in line: continue
            k, v = line.split("=", 1)
            v = v.strip()
            if v.startswith('"') and v.endswith('"'):
                props[k.strip()] = v[1:-1]; continue
            # Strip trailing-comma-inside-bracket (JS does this with regex /,\s*\]/g).
            import re as _re
            v = _re.sub(r",\s*\]", "]", v)
            try:
                props[k.strip()] = json.loads(v)
            except Exception:
                props[k.strip()] = v
        cameras.append({
            "name": props.get("name", name),
            "matrix": np.array(props.get("matrix", [[1,0,0],[0,1,0],[0,0,1]]), dtype=float),
            "distortions": np.array(props.get("distortions", [0]*5), dtype=float),
            "rotation": np.array(props.get("rotation", [0,0,0]), dtype=float),
            "translation": np.array(props.get("translation", [0,0,0]), dtype=float),
            "size": tuple(props.get("size", [640, 480])),
        })
    return cameras


def reproject_3d(points3d: np.ndarray, cam: dict) -> np.ndarray:
    """Project (N, 3) 3D points into (N, 2) 2D using OpenCV-style camera model.

    Uses cv2.projectPoints to get exact behavior matching the JS reprojectPoints.
    """
    import cv2
    valid = ~np.isnan(points3d).any(axis=1)
    out = np.full((len(points3d), 2), np.nan)
    if valid.sum() == 0:
        return out
    pts3d = points3d[valid].reshape(-1, 1, 3).astype(np.float64)
    pts2d, _ = cv2.projectPoints(pts3d, cam["rotation"], cam["translation"],
                                  cam["matrix"], cam["distortions"])
    out[valid] = pts2d.reshape(-1, 2)
    return out


# ---------------------------------------------------------------------------
# Per-session phantom injection
# ---------------------------------------------------------------------------
def process_session(session_id: str, sess_idx: int, n_animals: int,
                    calib_path: Path, luc3d_json: Path,
                    pred_h5_paths: dict[str, Path],
                    out_pred_h5_data: dict[str, dict[str, np.ndarray]],
                    history_window: int = 3,
                    max_phantom_age: int = 20,
                    out_luc3d_path: Path | None = None) -> dict:
    """Modify pred_h5 tracks data IN MEMORY (for the given session slice) and
    write a new luc3d_json. Returns stats.

    out_pred_h5_data[cam]["tracks"] is the full (n_sessions, max_frames, max_animals, n_nodes, 2)
    array that we'll write out at the end of all sessions.
    """
    cams = parse_calibration_toml(calib_path.read_text())
    cam_by_name = {c["name"]: c for c in cams}

    d = json.loads(luc3d_json.read_text())
    frames = d["frames"]

    # Build per-identity 3D pose history (list of (frame, points3d_array_or_None)).
    id_history: dict[int, list[tuple[int, np.ndarray]]] = defaultdict(list)
    for fr in frames:
        for t in fr["targets3d"]:
            if t["identityId"] is None or t["points3d"] is None: continue
            pose = np.array([(p if p is not None else [np.nan, np.nan, np.nan])
                             for p in t["points3d"]], dtype=float)
            if np.isnan(pose).all(): continue
            id_history[t["identityId"]].append((fr["frame"], pose))

    # For each identity, sort the history.
    for k in id_history:
        id_history[k].sort(key=lambda x: x[0])

    # For each frame, determine which (cam, identity) pairs are missing and
    # inject a phantom 2D pose at the const-velocity reprojection.
    stats = {"phantoms_injected": 0, "phantoms_skipped_no_history": 0,
             "phantoms_skipped_age": 0, "phantoms_skipped_no_slot": 0}

    # Helper to predict 3D pose for identity at frame_t.
    def predict_pose(identity_id: int, frame_t: int) -> np.ndarray | None:
        hist = id_history.get(identity_id)
        if not hist: return None
        # Find the most recent observation at frame < frame_t.
        last_idx = None
        for i, (fr, _) in enumerate(hist):
            if fr < frame_t:
                last_idx = i
            else:
                break
        if last_idx is None: return None
        last_frame, last_pose = hist[last_idx]
        if frame_t - last_frame > max_phantom_age:
            return None
        # Constant-velocity model: need at least 2 observations.
        if last_idx >= 1:
            prev_frame, prev_pose = hist[last_idx - 1]
            dt_prev = max(1, last_frame - prev_frame)
            v = (last_pose - prev_pose) / dt_prev
            dt = frame_t - last_frame
            return last_pose + v * dt
        return last_pose  # no velocity available, just hold position

    # For each frame, work out what identities each camera saw.
    new_frames = []
    for pf_idx, fr in enumerate(frames):
        # cam_assigned[cam] = set of identity_ids assigned to detections in this cam
        cam_assigned: dict[str, set[int]] = defaultdict(set)
        # cam_used_slots[cam] = set of trackIdx slots already used by real detections
        cam_used_slots: dict[str, set[int]] = defaultdict(set)
        for key, ident in fr["assignments"]:
            cam, slot = key.rsplit(":", 1)
            cam_assigned[cam].add(ident)
            cam_used_slots[cam].add(int(slot))

        # Find which identities should exist this frame: all currently tracked identities.
        live_identities = set()
        for ident, hist in id_history.items():
            for hf, _ in hist:
                if hf <= fr["frame"]:
                    if fr["frame"] - hf <= max_phantom_age:
                        live_identities.add(ident)
                    break
        # Also include identities seen in this frame.
        for ident in cam_assigned.values():
            live_identities.update(ident)

        new_assignments = list(fr["assignments"])

        # For each camera in calibration:
        for cam in cams:
            cam_name = cam["name"]
            if cam_name not in pred_h5_paths: continue
            for ident in live_identities:
                if ident in cam_assigned[cam_name]: continue
                # This (cam, ident) is missing → consider injecting a phantom.
                pred_pose_3d = predict_pose(ident, fr["frame"])
                if pred_pose_3d is None:
                    stats["phantoms_skipped_no_history"] += 1
                    continue
                # Reproject into this camera.
                pose_2d = reproject_3d(pred_pose_3d, cam)
                if np.isnan(pose_2d).all():
                    continue
                # Find an empty animal slot in pred_h5 for this frame.
                tracks = out_pred_h5_data[cam_name]["tracks"]
                _, _, max_animals, n_nodes, _ = tracks.shape
                empty_slot = None
                for slot in range(max_animals):
                    if slot in cam_used_slots[cam_name]: continue
                    if np.isnan(tracks[sess_idx, fr["frame"], slot]).all():
                        empty_slot = slot
                        break
                if empty_slot is None:
                    stats["phantoms_skipped_no_slot"] += 1
                    continue
                # Write the phantom 2D pose to pred_h5 (in-memory).
                # Need to handle node count: pose_2d has 15 nodes (or whatever the 3d had).
                n_phantom_nodes = min(n_nodes, pose_2d.shape[0])
                tracks[sess_idx, fr["frame"], empty_slot, :n_phantom_nodes, :] = pose_2d[:n_phantom_nodes]
                # Add the new assignment to luc3d_json.
                new_assignments.append([f"{cam_name}:{empty_slot}", ident])
                cam_used_slots[cam_name].add(empty_slot)
                cam_assigned[cam_name].add(ident)
                stats["phantoms_injected"] += 1

        new_frames.append({**fr, "assignments": new_assignments})

    if out_luc3d_path:
        out_luc3d_path.parent.mkdir(parents=True, exist_ok=True)
        d_out = {**d, "frames": new_frames, "phantomStats": stats}
        out_luc3d_path.write_text(json.dumps(d_out))

    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--luc3d-dir", required=True)
    ap.add_argument("--pred-h5-dir", default="outputs/predictions_h5s")
    ap.add_argument("--master-sheet", default="outputs/predictions_master_sheet.tsv")
    ap.add_argument("--out-pred-h5-dir", required=True)
    ap.add_argument("--out-luc3d-dir", required=True)
    ap.add_argument("--history-window", type=int, default=3)
    ap.add_argument("--max-phantom-age", type=int, default=20)
    ap.add_argument("--cameras", default="back,backL,mid,midL,top,topL")
    args = ap.parse_args()

    out_pred = Path(args.out_pred_h5_dir); out_pred.mkdir(parents=True, exist_ok=True)
    out_luc3d = Path(args.out_luc3d_dir); out_luc3d.mkdir(parents=True, exist_ok=True)
    cams_list = args.cameras.split(",")

    master = pd.read_csv(args.master_sheet, sep="\t", index_col=0)
    master["session"] = master["session"].astype(str)

    # Pre-load pred_h5 data into memory per camera (read once, modify, write once).
    print("loading pred_h5 datasets…")
    pred_h5_data = {}
    for cam in cams_list:
        p = Path(args.pred_h5_dir) / f"{cam}_predictions.h5"
        with h5py.File(p, "r") as f:
            tracks = f["tracks"][...]
        pred_h5_data[cam] = {"tracks": tracks, "src_path": p}
        print(f"  {cam}: shape={tracks.shape}, dtype={tracks.dtype}")

    luc3d_dir = Path(args.luc3d_dir)
    total_stats = {"phantoms_injected": 0, "phantoms_skipped_no_history": 0,
                   "phantoms_skipped_age": 0, "phantoms_skipped_no_slot": 0}

    pred_h5_paths = {cam: pred_h5_data[cam]["src_path"] for cam in cams_list}

    for idx_in_master, row in tqdm(list(master.iterrows()), desc="sessions"):
        sess_id = str(row["session"])
        jsn = luc3d_dir / f"{sess_id}.json"
        if not jsn.exists():
            continue
        calib = Path(row["calibration_toml"])
        stats = process_session(
            sess_id, int(idx_in_master), int(row["num_animals"]),
            calib, jsn, pred_h5_paths, pred_h5_data,
            history_window=args.history_window,
            max_phantom_age=args.max_phantom_age,
            out_luc3d_path=out_luc3d / f"{sess_id}.json",
        )
        for k, v in stats.items(): total_stats[k] += v

    # Write out the modified pred_h5 files.
    print("writing modified pred_h5 datasets…")
    for cam in cams_list:
        out_p = out_pred / f"{cam}_predictions.h5"
        tracks = pred_h5_data[cam]["tracks"]
        with h5py.File(out_p, "w") as f:
            f.create_dataset("tracks", data=tracks, dtype="float64",
                             chunks=(1, 4096, tracks.shape[2], tracks.shape[3], tracks.shape[4]),
                             compression="gzip")
        print(f"  wrote {out_p}")

    print(f"\ntotal phantoms injected: {total_stats['phantoms_injected']}")
    print(f"skipped (no history): {total_stats['phantoms_skipped_no_history']}")
    print(f"skipped (no empty slot): {total_stats['phantoms_skipped_no_slot']}")


if __name__ == "__main__":
    main()
