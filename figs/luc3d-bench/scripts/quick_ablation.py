"""Quick ablation: baseline vs +Kalman vs +Stitch vs +both on a few sessions.

Runs each config on N held-out sessions (capped frames), scores LUC3D against GT,
and prints mean IDF1 / IDSW per config.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from lib.slap2m_paths import BENCH_CAMERAS
from evaluate import eval_camera, PRED_H5_DIR

REPO = Path(__file__).resolve().parents[1]
DRIVER = REPO / "scripts" / "luc3d_track_all.mjs"
MASTER = REPO / "outputs" / "predictions_master_sheet.tsv"

ABLATE = {
    "baseline": {"applyKalman": False, "applyStitching": False},
    "kalman":   {"applyKalman": True,  "applyStitching": False, "kalmanAlpha": 1.0},
    "stitch":   {"applyKalman": False, "applyStitching": True,  "stitchMaxGap": 5},
    "both":     {"applyKalman": True,  "applyStitching": True,  "stitchMaxGap": 5, "kalmanAlpha": 1.0},
}


def run_one(args_tuple):
    name, cfg_path, idx, sid, num_animals, calibration, out_json, max_frames = args_tuple
    if out_json.exists():
        return name, sid, True
    cmd = ["node", str(DRIVER),
           "--session-idx", str(int(idx)),
           "--num-animals", str(int(num_animals)),
           "--calibration", str(calibration),
           "--pred-h5-dir", str(PRED_H5_DIR),
           "--out", str(out_json),
           "--params", str(cfg_path),
           "--max-frames", str(max_frames)]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    return name, sid, proc.returncode == 0


def score(out_json, gt_paths, idx):
    rows = []
    for cam, gt_path in gt_paths.items():
        try:
            m = eval_camera(PRED_H5_DIR / f"{cam}_predictions.h5",
                            Path(gt_path), out_json, None, session_idx=idx)
            mm = m.get("luc3d", {})
            rows.append({"camera": cam,
                         "idf1": mm.get("idf1"),
                         "mota": mm.get("mota"),
                         "num_switches": mm.get("num_switches")})
        except Exception as e:
            print(f"  score failed {cam}: {e}", file=sys.stderr)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-sessions", type=int, default=5)
    ap.add_argument("--max-frames", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out-root", default="/tmp/luc3d-ablate")
    args = ap.parse_args()

    master = pd.read_csv(MASTER, sep="\t", index_col=0)
    # Prefer multi-animal harder sessions (where tracking actually matters).
    candidates = master[(master["num_animals"] >= 2) & (master["difficulty"] >= 3)]
    sample = candidates.sample(args.n_sessions, random_state=args.seed)
    print("ablation sessions:", sample["session"].tolist())

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    for name, cfg in ABLATE.items():
        (out_root / f"{name}.json").write_text(json.dumps(cfg))

    tasks = []
    for name in ABLATE:
        cfg_path = out_root / f"{name}.json"
        for idx, row in sample.iterrows():
            sid = str(row["session"])
            out_json = out_root / name / f"{sid}.json"
            out_json.parent.mkdir(parents=True, exist_ok=True)
            tasks.append((name, cfg_path, int(idx), sid,
                          int(row["num_animals"]), str(row["calibration_toml"]),
                          out_json, args.max_frames))

    t0 = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(run_one, t) for t in tasks]
        for fut in futs:
            fut.result()
    print(f"runs done in {time.time()-t0:.1f}s")

    # Score
    all_rows = []
    for name in ABLATE:
        for idx, row in sample.iterrows():
            sid = str(row["session"])
            gt_paths = {}
            for cam in BENCH_CAMERAS:
                p = row.get(f"{cam}_proofread_h5")
                if isinstance(p, str) and p:
                    gt_paths[cam] = p
            out_json = out_root / name / f"{sid}.json"
            if not out_json.exists():
                continue
            for r in score(out_json, gt_paths, int(idx)):
                r.update({"ablation": name, "session": sid, "animals": row["num_animals"],
                          "difficulty": row["difficulty"]})
                all_rows.append(r)

    df = pd.DataFrame(all_rows)
    summary = df.groupby("ablation").agg(
        mean_idf1=("idf1", "mean"),
        mean_mota=("mota", "mean"),
        mean_idsw=("num_switches", "mean"),
        n=("idf1", "count"),
    )
    print("\nAblation summary:")
    print(summary.to_string())


if __name__ == "__main__":
    main()
