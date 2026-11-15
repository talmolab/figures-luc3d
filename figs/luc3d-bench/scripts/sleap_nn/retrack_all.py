"""Drive retrack_one over the full master sheet.

Iterates session rows × camera columns. Writes
  outputs/sleap_nn_retracked/<session>/<cam>.slp
Logs progress to a JSONL ledger so re-runs can resume.
"""
from __future__ import annotations
import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
DEFAULT_CAMS = ["back", "backL", "mid", "midL", "top", "topL"]


def _resolve_raw_slp(row, cam, repo_root):
    rel = row.get(f"{cam}_raw_pred_slp")
    if not rel or pd.isna(rel):
        return None
    p = Path(rel)
    if not p.is_absolute():
        # Master sheet paths are relative to SLAP-2M dataset root — search:
        candidates = [
            repo_root / rel,
            Path("/root/vast") / rel,  # adjust if dataset root differs
        ]
        for c in candidates:
            if c.exists():
                return c
        return None
    return p if p.exists() else None


def _run_job(job_with_kwargs):
    """Module-scope worker so ProcessPoolExecutor can pickle it."""
    from scripts.sleap_nn.retrack_one import retrack_one
    job, kwargs = job_with_kwargs
    sess, cam, src, dst, n_animals = job
    info = retrack_one(
        src, dst, target_instance_count=n_animals,
        min_instance_score=kwargs["min_instance_score"],
        min_mean_node_score=kwargs["min_mean_node_score"],
        min_visible_nodes=kwargs["min_visible_nodes"],
        overlapping_threshold=kwargs["overlapping_threshold"],
        window_size=kwargs["window_size"],
    )
    info["session"] = sess
    info["cam"] = cam
    return info


def retrack_all_rows(master, out_dir, cams=None, target_col="num_animals",
                     min_instance_score=0.85, min_mean_node_score=0.55,
                     min_visible_nodes=8, overlapping_threshold=0.50,
                     window_size=15, workers=1, ledger=None):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(master, sep="\t")
    cams = cams or DEFAULT_CAMS
    ledger = Path(ledger) if ledger else out_dir / "_retrack_ledger.jsonl"
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
        n_animals = int(row.get(target_col, 2))
        for cam in cams:
            if (sess, cam) in done:
                continue
            src = _resolve_raw_slp(row, cam, REPO)
            if src is None:
                continue
            dst = out_dir / sess / f"{cam}.slp"
            jobs.append((sess, cam, src, dst, n_animals))

    print(f"[retrack_all] {len(jobs)} jobs to run "
          f"({len(done)} already done; {workers} workers)", flush=True)

    kwargs = dict(
        min_instance_score=min_instance_score,
        min_mean_node_score=min_mean_node_score,
        min_visible_nodes=min_visible_nodes,
        overlapping_threshold=overlapping_threshold,
        window_size=window_size,
    )

    n_processed = 0
    if workers <= 1:
        for j in jobs:
            try:
                info = _run_job((j, kwargs))
                with ledger.open("a") as f:
                    f.write(json.dumps(info) + "\n")
                n_processed += 1
                if n_processed % 5 == 0:
                    print(f"  {n_processed}/{len(jobs)}", flush=True)
            except Exception as e:
                print(f"  FAILED {j[:2]}: {e}", file=sys.stderr)
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futs = [ex.submit(_run_job, (j, kwargs)) for j in jobs]
            for fut in as_completed(futs):
                try:
                    info = fut.result()
                    with ledger.open("a") as f:
                        f.write(json.dumps(info) + "\n")
                    n_processed += 1
                    if n_processed % 5 == 0:
                        print(f"  {n_processed}/{len(jobs)}", flush=True)
                except Exception as e:
                    print(f"  FAILED: {e}", file=sys.stderr)
    return {"n_processed": n_processed, "n_done_total": len(done) + n_processed}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--master", default=str(REPO / "outputs" / "predictions_master_sheet.tsv"))
    ap.add_argument("--out-dir", default=str(REPO / "outputs" / "sleap_nn_retracked"))
    ap.add_argument("--cams", nargs="+", default=DEFAULT_CAMS)
    ap.add_argument("--workers", type=int, default=4)
    # Defaults match retrack_one — calibrated for SLAP-2M (see retrack_one.py).
    ap.add_argument("--min-instance-score", type=float, default=0.85)
    ap.add_argument("--min-mean-node-score", type=float, default=0.55)
    ap.add_argument("--min-visible-nodes", type=int, default=8)
    ap.add_argument("--overlapping-threshold", type=float, default=0.50)
    ap.add_argument("--window-size", type=int, default=15)
    a = ap.parse_args()
    info = retrack_all_rows(
        a.master, a.out_dir, a.cams,
        min_instance_score=a.min_instance_score,
        min_mean_node_score=a.min_mean_node_score,
        min_visible_nodes=a.min_visible_nodes,
        overlapping_threshold=a.overlapping_threshold,
        window_size=a.window_size,
        workers=a.workers,
    )
    print(info)
