"""Batch orchestrator: drive scripts/luc3d_track_all.mjs over all sessions.

Reads outputs/predictions_master_sheet.tsv, shells out to the Node driver per
session in parallel via a process pool, and writes per-session JSON outputs.
Resumable: skips sessions whose output JSON already exists and parses cleanly.
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
from tqdm import tqdm

REPO = Path(__file__).resolve().parents[1]
DRIVER = REPO / "scripts" / "luc3d_track_all.mjs"
MASTER = REPO / "outputs" / "predictions_master_sheet.tsv"
PRED_H5_DIR = REPO / "outputs" / "predictions_h5s"
OUT_DIR = REPO / "outputs" / "luc3d_results"
LOG_DIR = REPO / "logs" / "luc3d_track_all"


def is_valid_output(p: Path) -> bool:
    if not p.exists():
        return False
    try:
        with p.open() as f:
            d = json.load(f)
        return "frames" in d and isinstance(d["frames"], list)
    except Exception:
        return False


def run_one(session_idx: int, session_id: str, num_animals: int, calibration: str,
            max_frames: int | None, out_dir: Path | None = None,
            params_path: str | None = None,
            pred_h5_dir: Path | None = None) -> tuple[str, bool, str]:
    out_dir = out_dir or OUT_DIR
    pred_h5_dir = pred_h5_dir or PRED_H5_DIR
    out_path = out_dir / f"{session_id}.json"
    log_path = LOG_DIR / f"{session_id}.log"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "node", str(DRIVER),
        "--session-idx", str(session_idx),
        "--num-animals", str(num_animals),
        "--calibration", calibration,
        "--pred-h5-dir", str(pred_h5_dir),
        "--out", str(out_path),
    ]
    if max_frames is not None:
        cmd += ["--max-frames", str(max_frames)]
    if params_path is not None:
        cmd += ["--params", str(params_path)]
    t0 = time.time()
    with log_path.open("w") as logf:
        logf.write("$ " + " ".join(cmd) + "\n")
        logf.flush()
        proc = subprocess.run(cmd, stdout=logf, stderr=subprocess.STDOUT)
    elapsed = time.time() - t0
    ok = proc.returncode == 0 and is_valid_output(out_path)
    return session_id, ok, f"{elapsed:.1f}s rc={proc.returncode}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) // 2))
    ap.add_argument("--max-frames", type=int, default=None, help="cap frames per session (debugging)")
    ap.add_argument("--force", action="store_true", help="re-run sessions that already have valid output")
    ap.add_argument("--filter-session", default=None, help="run only the given session ID")
    ap.add_argument("--params", default=None, help="path to TRACKER_PARAMS json (forwarded to driver --params)")
    ap.add_argument("--out-dir", default=None, help="output dir (default: outputs/luc3d_results)")
    ap.add_argument("--master", default=None, help="override master sheet (default: outputs/predictions_master_sheet.tsv)")
    ap.add_argument("--pred-h5-dir", default=None, help="override pred_h5 dir (default: outputs/predictions_h5s)")
    args = ap.parse_args()

    out_dir = Path(args.out_dir) if args.out_dir else OUT_DIR
    pred_h5_dir = Path(args.pred_h5_dir) if args.pred_h5_dir else PRED_H5_DIR
    master_path = Path(args.master) if args.master else MASTER

    df = pd.read_csv(master_path, sep="\t", index_col=0)
    if args.filter_session:
        df = df[df["session"].astype(str) == args.filter_session]
    if args.limit:
        df = df.head(args.limit)

    todo = []
    skipped = 0
    for idx_in_master, row in df.iterrows():
        session_id = str(row["session"])
        out_path = out_dir / f"{session_id}.json"
        if not args.force and is_valid_output(out_path):
            skipped += 1
            continue
        todo.append((int(idx_in_master), session_id, int(row["num_animals"]), str(row["calibration_toml"])))
    print(f"sessions: {len(df)} | skip-existing: {skipped} | run: {len(todo)} | workers: {args.workers} | out_dir: {out_dir}")
    if not todo:
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    failures = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(run_one, i, sid, n, cal, args.max_frames, out_dir, args.params, pred_h5_dir): sid for (i, sid, n, cal) in todo}
        for fut in tqdm(as_completed(futs), total=len(futs)):
            sid = futs[fut]
            try:
                sid_, ok, info = fut.result()
                if not ok:
                    failures.append((sid_, info))
            except Exception as e:
                failures.append((sid, str(e)))
    if failures:
        print(f"\n{len(failures)} failures:")
        for sid, info in failures[:20]:
            print(f"  {sid}: {info}")
        sys.exit(1)


if __name__ == "__main__":
    main()
