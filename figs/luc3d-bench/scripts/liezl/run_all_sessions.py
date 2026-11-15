"""Run CrossViewTracker on all (or N) SLAP-2M sessions in parallel.

Workers run independent processes, each handling one session. Output JSONs are
written to `outputs/PAF_3d_kalman/liezl_cross_view_results/{session_id}.json`.

Usage:
  python scripts/liezl/run_all_sessions.py --workers 8 [--limit N] [--skip-existing]
"""
from __future__ import annotations

import argparse
import json
import logging
import multiprocessing as mp
import os
import sys
import time
import traceback
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_one_session import run_one, setup_logging  # noqa: E402


PROGRESS_LOG = Path("/root/vast/eric/luc3d-bench/outputs/PAF_3d_kalman/track_b_progress.log")


def _log_event(message: str) -> None:
    try:
        PROGRESS_LOG.parent.mkdir(parents=True, exist_ok=True)
        with PROGRESS_LOG.open("a") as f:
            f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}\n")
    except Exception:
        pass


def _worker(row_dict, session_idx, out_dir, kwargs):
    """Worker: import-once, run-once."""
    setup_logging(level=logging.WARNING)
    log = logging.getLogger(f"worker[{row_dict['session']}]")
    try:
        row = pd.Series(row_dict)
        res = run_one(
            session_row=row,
            session_idx=int(session_idx),
            out_dir=Path(out_dir),
            log=log,
            **kwargs,
        )
        return {"ok": True, **res}
    except Exception as e:
        return {"ok": False, "session_id": str(row_dict.get("session", "?")), "error": str(e),
                "traceback": traceback.format_exc()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--master-sheet",
        default="/root/vast/eric/luc3d-bench/outputs/predictions_master_sheet.tsv",
    )
    ap.add_argument(
        "--out-dir",
        default="/root/vast/eric/luc3d-bench/outputs/PAF_3d_kalman/liezl_cross_view_results",
    )
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--only-sessions", nargs="+", default=None)
    ap.add_argument("--corr2d", type=float, default=1.0)
    ap.add_argument("--corr3d", type=float, default=1.0)
    ap.add_argument("--vel-thresh", type=float, default=10.0)
    ap.add_argument("--dist-thresh", type=float, default=50.0)
    ap.add_argument("--time-penalty", type=float, default=0.1)
    args = ap.parse_args()

    setup_logging()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.master_sheet, sep="\t", index_col=0)
    df["session"] = df["session"].astype(str)
    df = df.reset_index(drop=True)

    # Determine sessions to run.
    rows = []
    for idx, row in df.iterrows():
        if args.only_sessions and row["session"] not in args.only_sessions:
            continue
        if args.skip_existing and (out_dir / f"{row['session']}.json").exists():
            continue
        rows.append((row.to_dict(), idx))
    if args.limit:
        rows = rows[: args.limit]

    print(f"Will run {len(rows)} sessions with {args.workers} workers.")
    _log_event(f"run_all_sessions: starting {len(rows)} sessions with {args.workers} workers; "
               f"corr2d={args.corr2d} corr3d={args.corr3d} vel={args.vel_thresh} "
               f"dist={args.dist_thresh} t_pen={args.time_penalty}")

    tracker_kwargs = dict(
        correspondence_weight_2d=args.corr2d,
        correspondence_weight_3d=args.corr3d,
        velocity_threshold=args.vel_thresh,
        distance_threshold=args.dist_thresh,
        time_penalty=args.time_penalty,
    )

    # Spawn pool. Use 'spawn' to avoid CUDA/numpy issues from forking after imports.
    ctx = mp.get_context("spawn")
    results = []
    failures = []
    t0 = time.time()
    with ctx.Pool(processes=args.workers) as pool:
        async_results = [
            pool.apply_async(_worker, (rd, sidx, str(out_dir), tracker_kwargs))
            for (rd, sidx) in rows
        ]
        for i, ar in enumerate(async_results):
            res = ar.get()
            if res.get("ok"):
                results.append(res)
                msg = f"  done {res['session_id']} in {res['elapsed']:.1f}s ({i+1}/{len(rows)})"
                print(msg, flush=True)
                _log_event(msg)
            else:
                failures.append(res)
                msg = f"  FAILED {res['session_id']}: {res['error']}"
                print(msg, flush=True)
                _log_event(msg + "\n" + res.get("traceback", ""))
    total = time.time() - t0
    print(f"All done in {total/60:.1f} min. {len(results)} ok, {len(failures)} failed.")
    _log_event(f"run_all_sessions: complete in {total/60:.1f} min — {len(results)} ok, {len(failures)} failed")

    # Write a summary JSON.
    summary = {
        "n_ok": len(results),
        "n_failed": len(failures),
        "total_seconds": total,
        "results": results,
        "failures": [{k: v for k, v in r.items() if k != "traceback"} for r in failures],
    }
    summary_path = out_dir.parent / "track_b_run_summary.json"
    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
