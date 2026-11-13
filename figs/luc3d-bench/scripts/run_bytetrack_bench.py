"""ByteTrack baseline: independent per-camera 2D tracking on pose-derived bboxes.

ByteTrack is a single-view tracker. We run it once per camera using bboxes
inferred from pose keypoints, then defer cross-view identity correspondence
to evaluation time (Task 6) — where LUC3D's triangulated 3D targets are used
to align ByteTrack's per-view IDs into a global ID, isolating "temporal
tracking quality" from "cross-view geometry".

Output: outputs/bytetrack_results/{session_id}/{cam}.h5 — per-frame, per-detection
[track_id, x1, y1, x2, y2, score] arrays (NaN-padded to (n_frames, n_animals, 5)).
"""
from __future__ import annotations

import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import supervision as sv
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from lib.slap2m_paths import BENCH_CAMERAS

REPO = Path(__file__).resolve().parents[1]
MASTER = REPO / "outputs" / "predictions_master_sheet.tsv"
PRED_H5_DIR = REPO / "outputs" / "predictions_h5s"
OUT_DIR = REPO / "outputs" / "bytetrack_results"


def bbox_from_keypoints(kpts: np.ndarray, min_valid: int = 3, pad: int = 5) -> tuple[float, float, float, float, float] | None:
    """Return (x1,y1,x2,y2,score) for an instance, or None if too few keypoints."""
    valid = np.isfinite(kpts[:, 0]) & np.isfinite(kpts[:, 1])
    if int(valid.sum()) < min_valid:
        return None
    xs = kpts[valid, 0]
    ys = kpts[valid, 1]
    x1, x2 = float(xs.min()) - pad, float(xs.max()) + pad
    y1, y2 = float(ys.min()) - pad, float(ys.max()) + pad
    return (x1, y1, x2, y2, 1.0)


def run_camera_session(cam_h5: Path, session_idx: int,
                       track_activation_threshold: float = 0.25,
                       lost_track_buffer: int = 60,
                       minimum_matching_threshold: float = 0.9) -> np.ndarray:
    """Return ByteTrack tracks per frame, shape (n_frames, max_tracks, 6): [tid,x1,y1,x2,y2,score]."""
    with h5py.File(cam_h5, "r") as f:
        tracks = f["tracks"][session_idx]  # (n_frames, n_animals, n_nodes, 2)
    n_frames, n_animals, n_nodes, _ = tracks.shape

    # Per-instance bbox per frame
    bboxes = np.full((n_frames, n_animals, 5), np.nan, dtype=np.float64)
    has_det = np.zeros(n_frames, dtype=bool)
    for fi in range(n_frames):
        any_det = False
        for a in range(n_animals):
            b = bbox_from_keypoints(tracks[fi, a])
            if b is not None:
                bboxes[fi, a] = b
                any_det = True
        has_det[fi] = any_det

    # ByteTrack expects detections per frame as (N, 6): [x1,y1,x2,y2,confidence,class_id]
    # Defaults are the tuned winner from scripts/sweep_bytetrack.py (2026-05-17).
    tracker = sv.ByteTrack(track_activation_threshold=track_activation_threshold,
                           lost_track_buffer=lost_track_buffer,
                           minimum_matching_threshold=minimum_matching_threshold,
                           frame_rate=30)
    out = np.full((n_frames, n_animals, 6), np.nan, dtype=np.float64)
    for fi in range(n_frames):
        if not has_det[fi]:
            continue
        rows = []
        for a in range(n_animals):
            if np.isfinite(bboxes[fi, a, 0]):
                rows.append(bboxes[fi, a])
        if not rows:
            continue
        det_arr = np.asarray(rows)
        # supervision.Detections requires xyxy + confidence + class_id
        det = sv.Detections(
            xyxy=det_arr[:, :4].astype(np.float32),
            confidence=det_arr[:, 4].astype(np.float32),
            class_id=np.zeros(len(rows), dtype=int),
        )
        tracked = tracker.update_with_detections(det)
        # tracked.tracker_id holds the assigned IDs, aligned with tracked.xyxy rows
        for j, tid in enumerate(tracked.tracker_id):
            if j >= n_animals:
                break
            x1, y1, x2, y2 = tracked.xyxy[j]
            s = float(tracked.confidence[j]) if tracked.confidence is not None else 1.0
            out[fi, j] = (int(tid), x1, y1, x2, y2, s)
    return out


def process_session(args_tuple):
    idx_in_master, session_id, force, pred_h5_dir, out_dir = args_tuple
    pred_h5_dir = Path(pred_h5_dir)
    sess_dir = Path(out_dir) / session_id
    sess_dir.mkdir(parents=True, exist_ok=True)
    for cam in BENCH_CAMERAS:
        out_path = sess_dir / f"{cam}.h5"
        if not force and out_path.exists():
            continue
        cam_h5 = pred_h5_dir / f"{cam}_predictions.h5"
        arr = run_camera_session(cam_h5, idx_in_master)
        with h5py.File(out_path, "w") as f:
            f.create_dataset("tracks", data=arr, dtype="float64", compression="gzip", compression_opts=4)
            f.attrs["columns"] = "track_id,x1,y1,x2,y2,score"
            f.attrs["camera"] = cam
            f.attrs["session"] = session_id
    return session_id


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--filter-session", default=None)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) // 2))
    ap.add_argument("--master", default=str(MASTER))
    ap.add_argument("--pred-h5-dir", default=str(PRED_H5_DIR),
                    help="per-camera detection H5 dir (detections-only: outputs/detections_only_h5s)")
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    args = ap.parse_args()

    df = pd.read_csv(args.master, sep="\t", index_col=0)
    df = df.reset_index(drop=True)
    if args.filter_session:
        df = df[df["session"].astype(str) == args.filter_session]
    if args.limit:
        df = df.head(args.limit)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tasks = [(int(i), str(r["session"]), args.force, args.pred_h5_dir, args.out_dir)
             for i, r in df.iterrows()]
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for _ in tqdm(as_completed([ex.submit(process_session, t) for t in tasks]),
                      total=len(tasks), desc="sessions"):
            pass


if __name__ == "__main__":
    main()
