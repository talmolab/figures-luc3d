"""Sweep sleap-nn filter parameters on the 10-session debug slice and
evaluate each filter against BOTH LUC3D-no-PAF and Liezl's CrossViewTracker.

Stefan's recipe (`min_mean_node_score=0.80, min_visible_nodes=12,
overlap_threshold=0.50`) was tuned on different mouse data. Our SLAP-2M data
has a different score distribution, so the optimal filter cuts at different
thresholds. This script finds them.

For each filter config:
  1. Retrack the 10 debug sessions × 6 cams with the filter values.
  2. Aggregate to H5 (debug-only, in `outputs/_sweep_<tag>/predictions_h5s/`).
  3. Build a sub-master sheet pointed at those retracks.
  4. Run LUC3D no-PAF + tracklet_vote → evaluate → log.
  5. Run Liezl on the same retracked slps → evaluate → log.
  6. Persist a sweep_results.csv with one row per (config, tracker_kind).

Outputs:
  outputs/PAF_3d_kalman/_filter_sweep/<tag>/         per-config artifacts
  outputs/PAF_3d_kalman/_filter_sweep/sweep_results.csv  aggregate
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
DEBUG_MASTER = REPO / "outputs" / "splits" / "debug_master_sheet.tsv"
PY = REPO / ".venv" / "bin" / "python"
SLEAP_NN = REPO / "sleap_nn_env" / "bin" / "python"
LIEZL_PY = REPO / "liezl_env" / "bin" / "python"
SWEEP_DIR = REPO / "outputs" / "PAF_3d_kalman" / "_filter_sweep"


def tag_of(cfg: dict) -> str:
    return (f"mn{cfg['min_mean_node_score']:.2f}"
            f"_in{cfg['min_instance_score']:.2f}"
            f"_vn{cfg['min_visible_nodes']}"
            f"_ov{cfg['overlapping_threshold']:.2f}"
            f"_ws{cfg['window_size']}")


def run_cmd(cmd, log, cwd=str(REPO), env=None, timeout=None):
    """Run a subprocess, append output to log, return (rc, elapsed_seconds)."""
    t0 = time.time()
    with open(log, "ab") as f:
        f.write(b"$ " + " ".join(map(str, cmd)).encode() + b"\n")
        f.flush()
        proc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT,
                              cwd=cwd, env=env, timeout=timeout)
    return proc.returncode, time.time() - t0


def run_one_config(cfg: dict, workers: int) -> dict:
    """Run one filter config end-to-end on debug slice, return metrics dict."""
    tag = tag_of(cfg)
    out_root = SWEEP_DIR / tag
    out_root.mkdir(parents=True, exist_ok=True)
    log = out_root / "run.log"
    log.write_text("")  # truncate

    retrack_dir = out_root / "retracked"
    h5_dir = out_root / "predictions_h5s"
    msheet = out_root / "master_sheet.tsv"
    luc3d_out = out_root / "luc3d_no_paf_results"
    luc3d_tracklet = out_root / "luc3d_no_paf_tracklet_results"
    liezl_out = out_root / "liezl_results"
    eval_luc3d = out_root / "eval_luc3d_no_paf.csv"
    eval_liezl = out_root / "eval_liezl.csv"

    t_start = time.time()

    # Step 1: retrack debug sessions with this filter config
    print(f"[{tag}] retracking 60 cam-sessions...", flush=True)
    rc, dt = run_cmd([
        str(SLEAP_NN), "scripts/sleap_nn/retrack_all.py",
        "--master", str(DEBUG_MASTER),
        "--out-dir", str(retrack_dir),
        "--workers", str(workers),
        "--min-instance-score", str(cfg["min_instance_score"]),
        "--min-mean-node-score", str(cfg["min_mean_node_score"]),
        "--min-visible-nodes", str(cfg["min_visible_nodes"]),
        "--overlapping-threshold", str(cfg["overlapping_threshold"]),
        "--window-size", str(cfg["window_size"]),
    ], log, env={**__import__('os').environ, "PYTHONPATH": str(REPO)})
    if rc != 0:
        return {"tag": tag, "stage": "retrack_failed", "error_rc": rc}
    print(f"[{tag}] retrack done in {dt:.0f}s", flush=True)

    # Step 2: aggregate
    print(f"[{tag}] aggregating to H5...", flush=True)
    rc, dt = run_cmd([
        str(SLEAP_NN), "scripts/sleap_nn/aggregate_to_h5.py",
        "--master", str(DEBUG_MASTER),
        "--retrack-dir", str(retrack_dir),
        "--out-dir", str(h5_dir),
    ], log, env={**__import__('os').environ, "PYTHONPATH": str(REPO)})
    if rc != 0:
        return {"tag": tag, "stage": "aggregate_failed"}

    # Step 3: build per-config master sheet pointing at this config's slps
    rc, dt = run_cmd([
        str(SLEAP_NN), "scripts/sleap_nn/build_master_sheet.py",
        "--src", str(DEBUG_MASTER),
        "--retrack-dir", str(retrack_dir),
        "--dst", str(msheet),
    ], log)
    if rc != 0:
        return {"tag": tag, "stage": "build_msheet_failed"}

    # Step 4: LUC3D no-PAF + tracklet_vote
    print(f"[{tag}] LUC3D no-PAF...", flush=True)
    rc, dt = run_cmd([
        str(PY), "scripts/run_luc3d_bench.py",
        "--master", str(msheet),
        "--pred-h5-dir", str(h5_dir),
        "--params", str(REPO / "outputs" / "PAF_3d_kalman" / "no_paf_params.json"),
        "--out-dir", str(luc3d_out),
        "--workers", str(workers),
    ], log)
    if rc != 0:
        return {"tag": tag, "stage": "luc3d_failed"}

    rc, dt = run_cmd([
        str(PY), "scripts/stitch_postprocess.py",
        "--in", str(luc3d_out),
        "--out", str(luc3d_tracklet),
        "--mode", "tracklet_vote", "--tracklet-gap", "30",
    ], log)
    if rc != 0:
        return {"tag": tag, "stage": "tracklet_failed"}

    rc, dt = run_cmd([
        str(PY), "scripts/evaluate.py",
        "--luc3d-dir", str(luc3d_tracklet),
        "--pred-h5-dir", str(h5_dir),
        "--out-csv", str(eval_luc3d),
        "--workers", str(workers),
    ], log)
    if rc != 0:
        return {"tag": tag, "stage": "eval_luc3d_failed"}

    # Step 5: Liezl on the same retracked .slps
    print(f"[{tag}] Liezl tracker...", flush=True)
    rc, dt = run_cmd([
        str(LIEZL_PY), "scripts/liezl/run_all_sessions.py",
        "--master-sheet", str(msheet),
        "--out-dir", str(liezl_out),
        "--workers", str(min(workers, 8)),  # liezl is heavier per session
        "--skip-existing",
    ], log)
    if rc != 0:
        # Don't bail — Liezl can crash on some sessions; capture what completed
        print(f"[{tag}] WARN: Liezl run failed rc={rc} (continuing)", flush=True)

    rc, dt = run_cmd([
        str(PY), "scripts/evaluate.py",
        "--luc3d-dir", str(liezl_out),
        "--pred-h5-dir", str(h5_dir),
        "--out-csv", str(eval_liezl),
        "--workers", str(workers),
    ], log)
    if rc != 0:
        return {"tag": tag, "stage": "eval_liezl_failed"}

    # Step 6: compute summary metrics
    out = {"tag": tag, "stage": "ok", "wall_s": time.time() - t_start, **cfg}
    for name, csv in [("luc3d_no_paf", eval_luc3d), ("liezl", eval_liezl)]:
        df = pd.read_csv(csv)
        sub = df[df["tracker"] == "luc3d"]  # both pipelines write tracker="luc3d"
        if len(sub) == 0:
            out[f"{name}_n"] = 0
            continue
        out[f"{name}_n"] = len(sub)
        out[f"{name}_switches"] = int(sub["num_switches"].sum())
        out[f"{name}_idf1"] = float(sub["idf1"].mean())
        out[f"{name}_idp"] = float(sub["idp"].mean())
        out[f"{name}_idr"] = float(sub["idr"].mean())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()

    SWEEP_DIR.mkdir(parents=True, exist_ok=True)
    results_csv = SWEEP_DIR / "sweep_results.csv"

    # Configurations: vary one parameter at a time around the current baseline.
    baseline = dict(
        min_instance_score=0.85,
        min_mean_node_score=0.55,
        min_visible_nodes=8,
        overlapping_threshold=0.50,
        window_size=15,
    )
    configs = [baseline]
    # Trimmed sweep focused on the most-impactful axis (min_mean_node_score)
    # plus a few promising combinations. Total ~8 configs × ~30 min each ≈
    # 4-5 hr on debug slice.
    # vary min_mean_node_score:
    for v in [0.35, 0.45, 0.65]:
        c = dict(baseline); c["min_mean_node_score"] = v; configs.append(c)
    # vary overlap threshold (looser = more close-by detections survive):
    for v in [0.70]:
        c = dict(baseline); c["overlapping_threshold"] = v; configs.append(c)
    # Promising combos (looser min_mean + looser overlap = preserve recall):
    combos = [
        dict(baseline, min_mean_node_score=0.45, overlapping_threshold=0.70),
        dict(baseline, min_mean_node_score=0.35, overlapping_threshold=0.70),
        dict(baseline, min_mean_node_score=0.35, min_visible_nodes=6),
    ]
    configs.extend(combos)
    # Dedupe by tag:
    seen = set()
    unique = []
    for c in configs:
        t = tag_of(c)
        if t in seen:
            continue
        seen.add(t)
        unique.append(c)
    configs = unique

    print(f"sweeping {len(configs)} configs on debug slice")

    # If sweep_results.csv exists, skip configs that already have a row.
    done_tags = set()
    if results_csv.exists():
        try:
            prior = pd.read_csv(results_csv)
            done_tags = set(prior["tag"].tolist())
            print(f"resuming: {len(done_tags)} configs already done")
        except Exception:
            pass

    for i, cfg in enumerate(configs, 1):
        tag = tag_of(cfg)
        if tag in done_tags:
            print(f"[{i}/{len(configs)}] {tag} — cached, skipping")
            continue
        print(f"\n[{i}/{len(configs)}] {tag}")
        try:
            res = run_one_config(cfg, args.workers)
        except Exception as e:
            print(f"  FAILED: {e}")
            res = {"tag": tag, "stage": f"exception:{e}", **cfg}
        # Append to results CSV immediately so partial runs aren't lost
        df_row = pd.DataFrame([res])
        if results_csv.exists():
            df_row.to_csv(results_csv, mode="a", header=False, index=False)
        else:
            df_row.to_csv(results_csv, index=False)
        print(f"  done in {res.get('wall_s', 0):.0f}s; "
              f"luc3d_sw={res.get('luc3d_no_paf_switches', '?')} "
              f"liezl_sw={res.get('liezl_switches', '?')} "
              f"liezl_idf1={res.get('liezl_idf1', '?')}")

    # Print final summary
    if results_csv.exists():
        df = pd.read_csv(results_csv)
        df = df[df["stage"] == "ok"].copy()
        print("\n=== SWEEP RESULTS (sorted by liezl_switches) ===")
        cols = ["tag", "min_mean_node_score", "overlapping_threshold",
                "min_visible_nodes", "min_instance_score",
                "luc3d_no_paf_switches", "luc3d_no_paf_idf1",
                "liezl_switches", "liezl_idf1"]
        avail = [c for c in cols if c in df.columns]
        print(df.sort_values("liezl_switches")[avail].to_string(index=False))


if __name__ == "__main__":
    main()
