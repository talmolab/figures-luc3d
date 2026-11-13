"""Grid-sweep LUC3D tracker parameters on a held-out sweep split.

Each config writes its outputs to outputs/sweeps/{run_id}/configs/{cfg_id}/
and aggregated metrics to outputs/sweeps/{run_id}/metrics.csv.
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import random
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from lib.slap2m_paths import BENCH_CAMERAS

REPO = Path(__file__).resolve().parents[1]
DRIVER = REPO / "scripts" / "luc3d_track_all.mjs"
MASTER = REPO / "outputs" / "predictions_master_sheet.tsv"
PRED_H5_DIR = REPO / "outputs" / "predictions_h5s"

# Defaults from tracker.js (verified against source).
DEFAULTS = {
    "wEpipolar": 0.4,
    "wReprojection": 0.6,
    "reprojSigma": 20.0,
    "scoreThresh": 0.05,
    "continuityWeight": 2.0,
    "thirdCamMaxDist": 100,
}

# Coarse grid; w pairs constrained to sum to 1.
GRID = {
    "wEpipolar":        [0.2, 0.4, 0.6],
    "reprojSigma":      [15.0, 20.0, 30.0],
    "scoreThresh":      [0.01, 0.05, 0.1],
    "continuityWeight": [1.0, 2.0, 3.0],
}

# Narrower 1-D screens — much faster, useful for refining around a winner.
GRID_NARROW = {
    "wEpipolar":        [0.3, 0.4, 0.5],
    "continuityWeight": [1.5, 2.0, 2.5, 3.0],
}


def make_configs(grid: dict | None = None):
    grid = grid or GRID
    keys = list(grid.keys())
    for vals in itertools.product(*[grid[k] for k in keys]):
        cfg = dict(DEFAULTS)
        cfg.update(dict(zip(keys, vals)))
        if "wEpipolar" in cfg:
            cfg["wReprojection"] = 1.0 - cfg["wEpipolar"]
        yield cfg


def cfg_id(cfg):
    return "_".join(f"{k}={v}" for k, v in sorted(cfg.items()))[:120]


def pick_sweep_split(df: pd.DataFrame, n_sweep: int = 10, seed: int = 42):
    rng = random.Random(seed)
    indices = list(df.index)
    rng.shuffle(indices)
    sweep_idx = sorted(indices[:n_sweep])
    test_idx = sorted(indices[n_sweep:])
    return df.loc[sweep_idx], df.loc[test_idx]


def _run_single(args_tuple):
    cdir_str, params_path_str, idx_in_master, sid, num_animals, calibration, max_frames = args_tuple
    cdir = Path(cdir_str)
    out_json = cdir / f"{sid}.json"
    if out_json.exists():
        return sid, True, "skip"
    cmd = [
        "node", str(DRIVER),
        "--session-idx", str(int(idx_in_master)),
        "--num-animals", str(int(num_animals)),
        "--calibration", str(calibration),
        "--pred-h5-dir", str(PRED_H5_DIR),
        "--out", str(out_json),
        "--params", str(params_path_str),
    ]
    if max_frames is not None:
        cmd += ["--max-frames", str(max_frames)]
    log_path = cdir / f"{sid}.log"
    t0 = time.time()
    with log_path.open("w") as logf:
        proc = subprocess.run(cmd, stdout=logf, stderr=subprocess.STDOUT, check=False)
    return sid, proc.returncode == 0, f"{time.time()-t0:.1f}s"


def collect_tasks(configs, sweep_df, out_root, max_frames):
    tasks = []
    for cfg in configs:
        cid = cfg_id(cfg)
        cdir = out_root / "configs" / cid
        cdir.mkdir(parents=True, exist_ok=True)
        params_path = cdir / "params.json"
        params_path.write_text(json.dumps(cfg, indent=2))
        for idx_in_master, row in sweep_df.iterrows():
            sid = str(row["session"])
            tasks.append((str(cdir), str(params_path), int(idx_in_master), sid,
                          int(row["num_animals"]), str(row["calibration_toml"]), max_frames))
    return tasks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", default=time.strftime("run-%Y%m%d-%H%M%S"))
    ap.add_argument("--n-sweep", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-frames", type=int, default=2000, help="cap to keep sweep wall-clock tractable")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) // 2))
    ap.add_argument("--grid", choices=["coarse", "narrow"], default="coarse")
    args = ap.parse_args()

    df = pd.read_csv(MASTER, sep="\t", index_col=0)
    sweep_df, test_df = pick_sweep_split(df, args.n_sweep, args.seed)

    out_root = REPO / "outputs" / "sweeps" / args.run_id
    out_root.mkdir(parents=True, exist_ok=True)
    splits_path = REPO / "outputs" / "splits" / "sweep_test_split.json"
    splits_path.parent.mkdir(parents=True, exist_ok=True)
    splits_path.write_text(json.dumps({
        "seed": args.seed,
        "sweep_sessions": [str(s) for s in sweep_df["session"].tolist()],
        "test_sessions": [str(s) for s in test_df["session"].tolist()],
    }, indent=2))
    print(f"sweep sessions ({len(sweep_df)}):", sweep_df["session"].tolist())

    grid = GRID if args.grid == "coarse" else GRID_NARROW
    configs = list(make_configs(grid))
    tasks = collect_tasks(configs, sweep_df, out_root, args.max_frames)
    print(f"running {len(configs)} configs × {len(sweep_df)} sessions = {len(tasks)} runs "
          f"(workers={args.workers}, grid={args.grid})")

    failures = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(_run_single, t) for t in tasks]
        for fut in tqdm(as_completed(futs), total=len(futs), desc="runs"):
            sid, ok, info = fut.result()
            if not ok:
                failures.append((sid, info))

    print(f"all runs done → {out_root}/configs/")
    if failures:
        print(f"{len(failures)} failed runs (first 10):")
        for sid, info in failures[:10]:
            print(f"  {sid}: {info}")
    print(f"next: uv run python scripts/evaluate_sweep.py --run-id {out_root.name}")


if __name__ == "__main__":
    main()
