"""Evaluate SLEAP (raw) + LUC3D + ByteTrack against proofread ground truth.

For each session and each camera:
  - Load proofread tracks (analysis.h5) — ground truth.
  - SLEAP: use the raw track index from `.predictions.slp` (column `a` of the
    aggregated H5) directly as the predicted identity — i.e. whatever SLEAP's
    own tracker assigned at inference time, no proofreading.
  - LUC3D: load per-frame assignments (json) mapping (cam, raw_track_idx) → identity.
  - ByteTrack: per-camera h5 with its own track IDs.
  - Compute MOTA / IDF1 / ID switches per camera via motmetrics.
  - Aggregate per session and stratify by difficulty / bedding / animals.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import h5py
import motmetrics as mm
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm

# motmetrics 1.4.0 references np.asfarray, removed in numpy 2.0. Shim it so
# downstream functions (iou_matrix, etc.) still work.
if not hasattr(np, "asfarray"):
    np.asfarray = lambda a, dtype=np.float64: np.asarray(a, dtype=dtype)

sys.path.insert(0, str(Path(__file__).parent))
from lib.slap2m_paths import BENCH_CAMERAS

REPO = Path(__file__).resolve().parents[1]
MASTER = REPO / "outputs" / "predictions_master_sheet.tsv"
PRED_H5_DIR = REPO / "outputs" / "predictions_h5s"
LUC3D_DIR = REPO / "outputs" / "luc3d_results"
BYTETRACK_DIR = REPO / "outputs" / "bytetrack_results"
METRICS_DIR = REPO / "outputs" / "metrics"


def bbox_from_kpts(kpts: np.ndarray, pad: float = 5.0, min_valid: int = 3) -> np.ndarray | None:
    valid = np.isfinite(kpts[:, 0]) & np.isfinite(kpts[:, 1])
    if int(valid.sum()) < min_valid:
        return None
    xs, ys = kpts[valid, 0], kpts[valid, 1]
    return np.array([xs.min() - pad, ys.min() - pad, xs.max() + pad, ys.max() + pad], dtype=np.float64)


def load_gt(proofread_h5: Path) -> tuple[np.ndarray, np.ndarray]:
    """Return tracks (n_frames, n_tracks, n_nodes, 2) and occupancy (n_frames, n_tracks)."""
    with h5py.File(proofread_h5, "r") as f:
        tr = f["tracks"][:]   # (n_tracks, 2, n_nodes, n_frames)
        occ = f["track_occupancy"][:]   # (n_frames, n_tracks)
    tr = np.transpose(tr, (3, 0, 2, 1))   # → (n_frames, n_tracks, n_nodes, 2)
    return tr.astype(np.float64), occ.astype(bool)


def luc3d_assignments_for_cam(luc3d_json: dict, cam: str, n_frames: int, n_animals: int) -> np.ndarray:
    """For each frame and each raw_track_idx, return identity id or -1.

    Returns shape (n_frames, n_animals), int.
    """
    out = np.full((n_frames, n_animals), -1, dtype=np.int64)
    for entry in luc3d_json["frames"]:
        fi = entry["frame"]
        if fi >= n_frames:
            continue
        for key, ident in entry["assignments"]:
            cn, ti = key.split(":")
            if cn != cam:
                continue
            ti = int(ti)
            if ti < n_animals:
                out[fi, ti] = int(ident)
    return out


def eval_camera(
    pred_h5: Path, gt_h5: Path, luc3d_json_path: Path, bytetrack_h5: Path | None,
    session_idx: int, iou_thresh: float = 0.5, no_sleap: bool = False,
) -> dict:
    """Run motmetrics for LUC3D and ByteTrack on a single camera/session."""
    with h5py.File(pred_h5, "r") as f:
        raw_tracks = f["tracks"][session_idx]   # (n_frames, n_animals, n_nodes, 2)
    gt_tracks, gt_occ = load_gt(gt_h5)
    n_frames = min(raw_tracks.shape[0], gt_tracks.shape[0])
    n_animals = raw_tracks.shape[1]

    with open(luc3d_json_path) as fj:
        luc3d_json = json.load(fj)
    luc3d_ids = luc3d_assignments_for_cam(luc3d_json, gt_h5.parent.name, n_frames, n_animals)

    byte_arr = None
    if bytetrack_h5 is not None and bytetrack_h5.exists():
        with h5py.File(bytetrack_h5, "r") as f:
            byte_arr = f["tracks"][:n_frames]  # (n_frames, n_animals, 6)

    acc_luc3d = mm.MOTAccumulator(auto_id=True)
    acc_byte = mm.MOTAccumulator(auto_id=True)
    acc_sleap = mm.MOTAccumulator(auto_id=True)
    for fi in range(n_frames):
        gt_boxes, gt_ids = [], []
        for t in range(gt_tracks.shape[1]):
            if not gt_occ[fi, t]:
                continue
            b = bbox_from_kpts(gt_tracks[fi, t])
            if b is None:
                continue
            gt_boxes.append(b)
            gt_ids.append(int(t))   # GT track index as plain int

        # Per-detection bbox extraction shared by SLEAP and LUC3D (same source: raw_tracks).
        raw_box_for_a: dict[int, np.ndarray] = {}
        for a in range(n_animals):
            b = bbox_from_kpts(raw_tracks[fi, a])
            if b is not None:
                raw_box_for_a[a] = b
        gt_np = np.array(gt_boxes) if gt_boxes else np.empty((0, 4))

        # SLEAP predictions: raw track index `a` IS the SLEAP tracker identity.
        sl_boxes = [raw_box_for_a[a] for a in raw_box_for_a]
        sl_ids = [int(a) for a in raw_box_for_a]
        dist_sl = mm.distances.iou_matrix(gt_np,
                                          np.array(sl_boxes) if sl_boxes else np.empty((0, 4)),
                                          max_iou=1.0 - iou_thresh)
        acc_sleap.update(gt_ids, sl_ids, dist_sl)

        # LUC3D predictions (same raw bboxes, identity from luc3d_ids).
        pr_boxes, pr_ids = [], []
        for a, b in raw_box_for_a.items():
            ident = int(luc3d_ids[fi, a])
            if ident < 0:
                continue
            pr_boxes.append(b)
            pr_ids.append(int(ident))
        dist = mm.distances.iou_matrix(gt_np,
                                       np.array(pr_boxes) if pr_boxes else np.empty((0, 4)),
                                       max_iou=1.0 - iou_thresh)
        acc_luc3d.update(gt_ids, pr_ids, dist)

        # ByteTrack predictions
        if byte_arr is not None:
            bt_boxes, bt_ids = [], []
            # ByteTrack H5 may have fewer slots than the detection H5 (n_animals);
            # clamp so a wide detection pool (n_dets>4) can't index past it.
            for a in range(min(n_animals, byte_arr.shape[1])):
                row = byte_arr[fi, a]
                if not np.isfinite(row[0]):
                    continue
                tid = int(row[0])
                bt_boxes.append(row[1:5])
                bt_ids.append(int(tid))
            dist2 = mm.distances.iou_matrix(gt_np,
                                            np.array(bt_boxes) if bt_boxes else np.empty((0, 4)),
                                            max_iou=1.0 - iou_thresh)
            acc_byte.update(gt_ids, bt_ids, dist2)

    metrics = [
        # MOT identity-level
        "mota", "motp", "idf1", "idp", "idr",
        "idtp", "idfp", "idfn",
        # tracking continuity
        "num_switches", "num_fragmentations",
        # detection-level
        "precision", "recall",
        "num_false_positives", "num_misses",
        "num_detections", "num_objects", "num_predictions",
        # GT coverage
        "mostly_tracked", "partially_tracked", "mostly_lost",
        "num_unique_objects",
        "num_frames",
    ]
    mh = mm.metrics.create()
    # In detections-only mode the H5 slot is a per-frame detection position
    # (unstable), so "SLEAP identity = slot" is meaningless — suppress it. The
    # legitimate single-view SLEAP baseline is evaluated separately on the OLD
    # tracked H5 (see the orchestration script).
    if no_sleap:
        accs, names = [acc_luc3d], ["luc3d"]
    else:
        accs, names = [acc_sleap, acc_luc3d], ["sleap", "luc3d"]
    if byte_arr is not None:
        accs.append(acc_byte)
        names.append("bytetrack")
    summary = mh.compute_many(accs, names=names, metrics=metrics)
    return summary.to_dict(orient="index")


def _eval_session_cam(args_tuple):
    (pred_h5_str, gt_h5_str, luc3d_json_str, bt_h5_str, session_idx, session_id,
     cam, difficulty, bedding, animals, no_sleap) = args_tuple
    bt_h5 = Path(bt_h5_str) if bt_h5_str else None
    metrics = eval_camera(Path(pred_h5_str), Path(gt_h5_str), Path(luc3d_json_str),
                          bt_h5 if bt_h5 and bt_h5.exists() else None,
                          session_idx=session_idx, no_sleap=no_sleap)
    rows = []
    for tracker, m in metrics.items():
        row_out = {"session": session_id, "camera": cam, "tracker": tracker,
                   "difficulty": difficulty, "bedding": bedding, "animals": animals}
        row_out.update(m)
        rows.append(row_out)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--filter-session", default=None)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) // 2))
    ap.add_argument("--luc3d-dir", default=None, help="override LUC3D output dir (default: outputs/luc3d_results)")
    ap.add_argument("--pred-h5-dir", default=None, help="override pred_h5 dir (default: outputs/predictions_h5s)")
    ap.add_argument("--bytetrack-dir", default=None, help="override ByteTrack results dir (default: outputs/bytetrack_results)")
    ap.add_argument("--out-csv", default=None, help="override per_camera_session_metrics path")
    ap.add_argument("--no-sleap-baseline", action="store_true",
                    help="suppress the 'sleap' row (detections-only mode: H5 slot "
                         "is an unstable detection position, so slot-as-identity is meaningless)")
    ap.add_argument("--no-bytetrack", action="store_true",
                    help="skip scoring the ByteTrack H5 even if present")
    ap.add_argument("--master", default=None,
                    help="override master sheet (session order MUST match --pred-h5-dir's "
                         "session axis). Default: outputs/predictions_master_sheet.tsv")
    args = ap.parse_args()

    luc3d_dir = Path(args.luc3d_dir) if args.luc3d_dir else LUC3D_DIR
    pred_h5_dir = Path(args.pred_h5_dir) if args.pred_h5_dir else PRED_H5_DIR
    bytetrack_dir = Path(args.bytetrack_dir) if args.bytetrack_dir else BYTETRACK_DIR

    master_path = Path(args.master) if args.master else MASTER
    df = pd.read_csv(master_path, sep="\t", index_col=0)
    df = df.reset_index(drop=True)
    if args.filter_session:
        df = df[df["session"].astype(str) == args.filter_session]
    if args.limit:
        df = df.head(args.limit)

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    tasks = []
    for idx_in_master, row in df.iterrows():
        session_id = str(row["session"])
        luc3d_json_path = luc3d_dir / f"{session_id}.json"
        if not luc3d_json_path.exists():
            print(f"  skip {session_id}: no LUC3D output at {luc3d_json_path}")
            continue
        for cam in BENCH_CAMERAS:
            pred_h5 = pred_h5_dir / f"{cam}_predictions.h5"
            gt_h5_str = row[f"{cam}_proofread_h5"]
            if not isinstance(gt_h5_str, str) or not gt_h5_str:
                continue
            bt_h5 = bytetrack_dir / session_id / f"{cam}.h5"
            bt_h5_arg = str(bt_h5) if (bt_h5.exists() and not args.no_bytetrack) else None
            tasks.append((str(pred_h5), gt_h5_str, str(luc3d_json_path),
                          bt_h5_arg,
                          int(idx_in_master), session_id, cam,
                          int(row["difficulty"]), row["bedding"], int(row["num_animals"]),
                          bool(args.no_sleap_baseline)))

    print(f"running {len(tasks)} (session, cam) evaluations with {args.workers} workers")
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(_eval_session_cam, t) for t in tasks]
        for fut in tqdm(as_completed(futs), total=len(futs), desc="evals"):
            try:
                rows.extend(fut.result())
            except Exception as e:
                print(f"  eval failed: {e}", file=sys.stderr)

    out = pd.DataFrame(rows)
    out_path = Path(args.out_csv) if args.out_csv else METRICS_DIR / "per_camera_session_metrics.csv"
    out.to_csv(out_path, index=False)
    print(f"wrote {out_path} ({len(out)} rows)")

    if len(out) > 0:
        agg = out.groupby(["tracker", "difficulty"]).agg(
            mota=("mota", "mean"), idf1=("idf1", "mean"),
            switches=("num_switches", "mean"),
        ).reset_index()
        agg_path = METRICS_DIR / "by_difficulty.csv"
        agg.to_csv(agg_path, index=False)
        print(f"wrote {agg_path}")


if __name__ == "__main__":
    main()
