"""Drive filter_one over the full master sheet (detections-only export).

Mirrors retrack_all.py but calls filter_one (filter-only, no tracking, no cap).
Writes outputs/detections_only/<session>/<cam>.slp and a JSONL ledger so
re-runs resume. Each file is guarded by a per-file timeout (default 1800 s, as
in retrack_one.py:79) so one pathological camera cannot stall the pool.
"""
from __future__ import annotations

import argparse
import json
import signal
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
DEFAULT_CAMS = ["back", "backL", "mid", "midL", "top", "topL"]


def _resolve_raw_slp(row, cam, repo_root):
    rel = row.get(f"{cam}_raw_pred_slp")
    if rel is None or (isinstance(rel, float) and pd.isna(rel)) or rel == "":
        return None
    p = Path(rel)
    if not p.is_absolute():
        for c in (repo_root / rel, Path("/root/vast") / rel):
            if c.exists():
                return c
        return None
    return p if p.exists() else None


class _Timeout(Exception):
    pass


def _run_job(job_with_kwargs):
    """Module-scope worker (picklable). Per-file SIGALRM guard."""
    from scripts.sleap_nn.filter_only import filter_one

    job, kwargs, timeout_s = job_with_kwargs
    sess, cam, src, dst = job

    def _alarm(_signum, _frame):
        raise _Timeout(f"filter_one exceeded {timeout_s}s")

    old = signal.signal(signal.SIGALRM, _alarm)
    signal.alarm(int(timeout_s))
    try:
        info = filter_one(
            src, dst,
            min_instance_score=kwargs["min_instance_score"],
            min_mean_node_score=kwargs["min_mean_node_score"],
            min_visible_nodes=kwargs["min_visible_nodes"],
            overlapping_threshold=kwargs["overlapping_threshold"],
        )
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)
    info["session"] = sess
    info["cam"] = cam
    return info


def filter_all_rows(master, out_dir, cams=None,
                    min_instance_score=0.85, min_mean_node_score=0.55,
                    min_visible_nodes=8, overlapping_threshold=0.50,
                    workers=16, ledger=None, timeout_s=1800,
                    only_sessions=None):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(master, sep="\t")
    if only_sessions:
        only = {str(s) for s in only_sessions}
        df = df[df["session"].astype(str).isin(only)]
    cams = cams or DEFAULT_CAMS
    ledger = Path(ledger) if ledger else out_dir / "_filter_ledger.jsonl"
    done = set()
    if ledger.exists():
        for line in ledger.read_text().splitlines():
            try:
                d = json.loads(line)
                done.add((d["session"], d["cam"]))
            except Exception:
                pass

    jobs = []
    for _, row in df.iterrows():
        sess = str(row["session"])
        for cam in cams:
            if (sess, cam) in done:
                continue
            src = _resolve_raw_slp(row, cam, REPO)
            if src is None:
                continue
            dst = out_dir / sess / f"{cam}.slp"
            jobs.append((sess, cam, src, dst))

    print(f"[filter_only_all] {len(jobs)} jobs to run "
          f"({len(done)} already done; {workers} workers)", flush=True)

    kwargs = dict(
        min_instance_score=min_instance_score,
        min_mean_node_score=min_mean_node_score,
        min_visible_nodes=min_visible_nodes,
        overlapping_threshold=overlapping_threshold,
    )

    n_processed = 0
    global_max_dets = 0

    def _record(info):
        nonlocal n_processed, global_max_dets
        with ledger.open("a") as f:
            f.write(json.dumps(info) + "\n")
        n_processed += 1
        global_max_dets = max(global_max_dets, info.get("max_dets_per_frame", 0))
        if n_processed % 5 == 0:
            print(f"  {n_processed}/{len(jobs)}  (running max dets/frame={global_max_dets})",
                  flush=True)

    if workers <= 1:
        for j in jobs:
            try:
                _record(_run_job((j, kwargs, timeout_s)))
            except Exception as e:
                print(f"  FAILED {j[:2]}: {e}", file=sys.stderr, flush=True)
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(_run_job, (j, kwargs, timeout_s)): j for j in jobs}
            for fut in as_completed(futs):
                try:
                    _record(fut.result())
                except Exception as e:
                    j = futs[fut]
                    print(f"  FAILED {j[:2]}: {e}", file=sys.stderr, flush=True)

    print(f"[filter_only_all] done: {n_processed} processed, "
          f"dataset-wide max dets/frame this run = {global_max_dets}", flush=True)
    return {"n_processed": n_processed, "n_done_total": len(done) + n_processed,
            "max_dets_per_frame_this_run": global_max_dets}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--master", default=str(REPO / "outputs" / "predictions_master_sheet.tsv"))
    ap.add_argument("--out-dir", default=str(REPO / "outputs" / "detections_only"))
    ap.add_argument("--cams", nargs="+", default=DEFAULT_CAMS)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--timeout-s", type=int, default=1800)
    ap.add_argument("--only-sessions", nargs="+", default=None,
                    help="restrict to these session ids (smoke testing)")
    ap.add_argument("--min-instance-score", type=float, default=0.85)
    ap.add_argument("--min-mean-node-score", type=float, default=0.55)
    ap.add_argument("--min-visible-nodes", type=int, default=8)
    ap.add_argument("--overlapping-threshold", type=float, default=0.50)
    a = ap.parse_args()
    info = filter_all_rows(
        a.master, a.out_dir, a.cams,
        min_instance_score=a.min_instance_score,
        min_mean_node_score=a.min_mean_node_score,
        min_visible_nodes=a.min_visible_nodes,
        overlapping_threshold=a.overlapping_threshold,
        workers=a.workers, timeout_s=a.timeout_s,
        only_sessions=a.only_sessions,
    )
    print(info)
