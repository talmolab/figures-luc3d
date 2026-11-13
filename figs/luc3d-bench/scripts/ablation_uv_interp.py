"""A/B: baseline vs +UndercountVerify vs +Interpolation vs +both, on hard sessions.

Picks N harder sessions (multi-animal, difficulty>=3), runs each variant on
2000-frame caps, scores LUC3D against GT, and prints deltas vs baseline.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from lib.slap2m_paths import BENCH_CAMERAS
from evaluate import eval_camera, PRED_H5_DIR

REPO = Path(__file__).resolve().parents[1]
DRIVER = REPO / "scripts" / "luc3d_track_all.mjs"
MASTER = REPO / "outputs" / "predictions_master_sheet.tsv"

ABLATE = {
    "baseline":     {"applyUndercountVerify": False, "applyUndercountVerifyClean": False,
                     "applyMotionVerify": False, "applyVarianceInflation": False,
                     "applyInterpolation": False},
    "vi_60_0.6":    {"applyVarianceInflation": True, "viAbsThresh": 60, "viRelThresh": 0.6},
    "vi_40_0.7":    {"applyVarianceInflation": True, "viAbsThresh": 40, "viRelThresh": 0.7},
    "vi_80_0.5":    {"applyVarianceInflation": True, "viAbsThresh": 80, "viRelThresh": 0.5},
    "vi_60_0.6_i":  {"applyVarianceInflation": True, "viAbsThresh": 60, "viRelThresh": 0.6,
                     "applyInterpolation": True, "interpMaxGap": 5},
}


def run_one(args_tuple):
    name, cfg_path, idx, sid, num_animals, calibration, out_json, max_frames = args_tuple
    if out_json.exists() and os.path.getsize(out_json) > 100:
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
                         "idp": mm.get("idp"),
                         "idr": mm.get("idr"),
                         "num_switches": mm.get("num_switches")})
        except Exception as e:
            print(f"  score failed {cam}: {e}", file=sys.stderr)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-sessions", type=int, default=10)
    ap.add_argument("--max-frames", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--out-root", default="/tmp/luc3d-uv-ablate")
    ap.add_argument("--min-difficulty", type=int, default=3)
    ap.add_argument("--min-animals", type=int, default=2)
    args = ap.parse_args()

    master = pd.read_csv(MASTER, sep="\t", index_col=0)
    candidates = master[(master["num_animals"] >= args.min_animals) &
                        (master["difficulty"] >= args.min_difficulty)]
    sample = candidates.sample(min(args.n_sessions, len(candidates)), random_state=args.seed)
    print(f"sessions ({len(sample)}):", sample["session"].tolist())

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
        for fut in ex.map(run_one, tasks):
            pass
    print(f"runs done in {time.time()-t0:.1f}s")

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
                r.update({"ablation": name, "session": sid,
                          "animals": row["num_animals"],
                          "difficulty": row["difficulty"]})
                all_rows.append(r)

    df = pd.DataFrame(all_rows)
    summary = df.groupby("ablation").agg(
        mean_idf1=("idf1", "mean"),
        mean_idp=("idp", "mean"),
        mean_idr=("idr", "mean"),
        mean_mota=("mota", "mean"),
        mean_idsw=("num_switches", "mean"),
        n=("idf1", "count"),
    )
    keys = [k for k in ABLATE.keys() if k in summary.index]
    summary = summary.loc[keys]
    print("\nAblation summary (LUC3D mean across all camera-sessions in sample):")
    print(summary.to_string())
    print("\nDeltas vs baseline:")
    base = summary.loc["baseline"]
    for k in keys[1:]:
        row = summary.loc[k]
        print(f"  {k:22s} | idf1 {row['mean_idf1']-base['mean_idf1']:+.4f} | "
              f"idr {row['mean_idr']-base['mean_idr']:+.4f} | "
              f"idsw {row['mean_idsw']-base['mean_idsw']:+.2f}")


if __name__ == "__main__":
    main()
