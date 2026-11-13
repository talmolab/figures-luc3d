"""Grid-search ByteTrack knobs against IDF1 on the same sweep split as LUC3D.

Knobs:
  - track_activation_threshold (low score wakes a track)
  - lost_track_buffer          (how long to keep a lost track alive)
  - minimum_matching_threshold (IoU gate for re-association)

For each config, runs supervision.ByteTrack on the sweep split's camera-sessions,
scores against GT via motmetrics, writes per-config metrics, and picks the
winner by mean IDF1.

Outputs:
  outputs/sweeps/bytetrack-{run_id}/configs/{cfg_id}/{session}_{cam}.h5
  outputs/sweeps/bytetrack-{run_id}/metrics.csv
  outputs/sweeps/bytetrack-{run_id}/summary.csv
  outputs/sweeps/bytetrack-{run_id}/winner.json
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import h5py
import motmetrics as mm
import numpy as np
import pandas as pd
import supervision as sv
from tqdm import tqdm

if not hasattr(np, "asfarray"):
    np.asfarray = lambda a, dtype=np.float64: np.asarray(a, dtype=dtype)

sys.path.insert(0, str(Path(__file__).parent))
from lib.slap2m_paths import BENCH_CAMERAS
from evaluate import bbox_from_kpts, load_gt
from run_bytetrack_bench import bbox_from_keypoints

REPO = Path(__file__).resolve().parents[1]
MASTER = REPO / "outputs" / "predictions_master_sheet.tsv"
PRED_H5_DIR = REPO / "outputs" / "predictions_h5s"
SWEEPS_DIR = REPO / "outputs" / "sweeps"
SPLIT_PATH = REPO / "outputs" / "splits" / "sweep_test_split.json"

GRID = {
    "track_activation_threshold": [0.1, 0.25, 0.5],
    "lost_track_buffer":          [15, 30, 60, 120],
    "minimum_matching_threshold": [0.5, 0.7, 0.8, 0.9],
}


def cfg_id(cfg):
    return "_".join(f"{k}={v}" for k, v in sorted(cfg.items()))


def run_bytetrack(pred_h5: Path, session_idx: int, cfg: dict) -> np.ndarray:
    with h5py.File(pred_h5, "r") as f:
        tracks = f["tracks"][session_idx]
    n_frames, n_animals = tracks.shape[:2]
    tracker = sv.ByteTrack(
        track_activation_threshold=cfg["track_activation_threshold"],
        lost_track_buffer=cfg["lost_track_buffer"],
        minimum_matching_threshold=cfg["minimum_matching_threshold"],
        frame_rate=30,
    )
    out = np.full((n_frames, n_animals, 6), np.nan, dtype=np.float64)
    for fi in range(n_frames):
        rows = []
        for a in range(n_animals):
            b = bbox_from_keypoints(tracks[fi, a])
            if b is not None:
                rows.append(b)
        if not rows:
            continue
        det_arr = np.asarray(rows)
        det = sv.Detections(
            xyxy=det_arr[:, :4].astype(np.float32),
            confidence=det_arr[:, 4].astype(np.float32),
            class_id=np.zeros(len(rows), dtype=int),
        )
        tracked = tracker.update_with_detections(det)
        for j, tid in enumerate(tracked.tracker_id):
            if j >= n_animals:
                break
            x1, y1, x2, y2 = tracked.xyxy[j]
            s = float(tracked.confidence[j]) if tracked.confidence is not None else 1.0
            out[fi, j] = (int(tid), x1, y1, x2, y2, s)
    return out


def score_bt(bt_arr: np.ndarray, gt_h5: Path, iou_thresh: float = 0.5) -> dict:
    gt_tracks, gt_occ = load_gt(gt_h5)
    n_frames = min(bt_arr.shape[0], gt_tracks.shape[0])
    n_animals = bt_arr.shape[1]

    acc = mm.MOTAccumulator(auto_id=True)
    for fi in range(n_frames):
        gt_boxes, gt_ids = [], []
        for t in range(gt_tracks.shape[1]):
            if not gt_occ[fi, t]:
                continue
            b = bbox_from_kpts(gt_tracks[fi, t])
            if b is None:
                continue
            gt_boxes.append(b)
            gt_ids.append(int(t))
        gt_np = np.array(gt_boxes) if gt_boxes else np.empty((0, 4))

        bt_boxes, bt_ids = [], []
        for a in range(n_animals):
            row = bt_arr[fi, a]
            if not np.isfinite(row[0]):
                continue
            bt_boxes.append(row[1:5])
            bt_ids.append(int(row[0]))
        dist = mm.distances.iou_matrix(gt_np,
                                       np.array(bt_boxes) if bt_boxes else np.empty((0, 4)),
                                       max_iou=1.0 - iou_thresh)
        acc.update(gt_ids, bt_ids, dist)

    mh = mm.metrics.create()
    s = mh.compute(acc, metrics=["mota", "idf1", "idp", "idr", "num_switches", "num_fragmentations"], name="bt")
    return s.iloc[0].to_dict()


def score_one_config(args_tuple):
    cfg, sweep_records = args_tuple
    cid = cfg_id(cfg)
    rows = []
    for rec in sweep_records:
        idx_in_master, sid, gt_paths = rec
        for cam, gt_path in gt_paths.items():
            try:
                bt_arr = run_bytetrack(PRED_H5_DIR / f"{cam}_predictions.h5",
                                       int(idx_in_master), cfg)
                m = score_bt(bt_arr, Path(gt_path))
            except Exception as e:
                print(f"  [{cid}/{sid}/{cam}] failed: {e}", file=sys.stderr)
                continue
            rows.append({
                "cfg_id": cid, "session": sid, "camera": cam,
                **{k: cfg[k] for k in cfg},
                "idf1": m.get("idf1"), "mota": m.get("mota"),
                "idp": m.get("idp"), "idr": m.get("idr"),
                "num_switches": m.get("num_switches"),
                "num_fragmentations": m.get("num_fragmentations"),
            })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", default=time.strftime("bytetrack-%Y%m%d-%H%M%S"))
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) // 4))
    args = ap.parse_args()

    master_df = pd.read_csv(MASTER, sep="\t", index_col=0)
    split = json.loads(SPLIT_PATH.read_text())
    sweep_sessions = set(split["sweep_sessions"])
    sweep_df = master_df[master_df["session"].astype(str).isin(sweep_sessions)]
    print(f"sweep split: {len(sweep_df)} sessions")

    sweep_records = []
    for idx, row in sweep_df.iterrows():
        sid = str(row["session"])
        gt_paths = {}
        for cam in BENCH_CAMERAS:
            p = row.get(f"{cam}_proofread_h5")
            if isinstance(p, str) and p:
                gt_paths[cam] = p
        if gt_paths:
            sweep_records.append((int(idx), sid, gt_paths))

    keys = list(GRID.keys())
    configs = [dict(zip(keys, vals)) for vals in itertools.product(*[GRID[k] for k in keys])]
    print(f"running {len(configs)} configs × {len(sweep_records)} sessions × ~5 cams "
          f"(workers={args.workers})")

    run_dir = SWEEPS_DIR / args.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    all_rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(score_one_config, (cfg, sweep_records)) for cfg in configs]
        for fut in tqdm(as_completed(futs), total=len(futs), desc="configs"):
            try:
                all_rows.extend(fut.result())
            except Exception as e:
                print(f"  config failed: {e}", file=sys.stderr)

    df = pd.DataFrame(all_rows)
    metrics_path = run_dir / "metrics.csv"
    df.to_csv(metrics_path, index=False)
    print(f"wrote {metrics_path} ({len(df)} rows)")

    if df.empty:
        return
    summary = df.groupby("cfg_id").agg(
        mean_idf1=("idf1", "mean"),
        mean_mota=("mota", "mean"),
        mean_idsw=("num_switches", "mean"),
        n=("idf1", "count"),
    ).reset_index().sort_values("mean_idf1", ascending=False)
    summary_path = run_dir / "summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"\ntop 10:\n{summary.head(10).to_string(index=False)}")

    best = summary.iloc[0]
    best_cfg = json.loads(df[df["cfg_id"] == best["cfg_id"]].iloc[0]
                          [list(GRID.keys())].to_json())
    (run_dir / "winner.json").write_text(json.dumps({
        "cfg_id": best["cfg_id"],
        "mean_idf1": float(best["mean_idf1"]),
        "mean_mota": float(best["mean_mota"]),
        "mean_idsw": float(best["mean_idsw"]),
        "params": best_cfg,
    }, indent=2))
    print(f"\nwrote {run_dir / 'winner.json'}")


if __name__ == "__main__":
    main()
