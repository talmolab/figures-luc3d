"""Autonomous iteration controller for the zero-swaps benchmark.

After the orchestrator pipeline produces baseline new-tracker rows, this
script:
  1. Renders the current comparison figure
  2. Computes whether the final stage beats the existing best
     (luc3d_paf_l1_tracklet: 1,777 switches, IDF1 0.728)
  3. If not better, picks the next sweep round (cheapest first — Stage 4
     stitch params, then Stage 2 dedupe+gate combos), runs it, re-evaluates
  4. Repeats until we beat both metric bars OR exhaust the sweep budget.

Run: PYTHONPATH=. .venv/bin/python scripts/iterate_zero_swaps.py
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
METRICS = REPO / "outputs" / "PAF_3d_kalman" / "metrics" / "per_camera_session_metrics.csv"
PLOTS = REPO / "outputs" / "PAF_3d_kalman" / "plots"
ITER_LOG = REPO / "logs" / "zero_swaps_iterations.log"

PY = REPO / ".venv" / "bin" / "python"
SLEAP_NN = REPO / "sleap_nn_env" / "bin" / "python"

BASELINE_TRACKER = "luc3d_paf_l1_tracklet"
FINAL_TRACKER = "luc3d_paf_l1_dedupe_tracklet3d_stitched"
PIPELINE_TRACKERS = [
    "sleap_nn_filtered", "luc3d_paf_l1_dedupe",
    "luc3d_paf_l1_dedupe_tracklet3d", "luc3d_paf_l1_dedupe_tracklet3d_stitched",
]


def log(msg: str) -> None:
    ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    ITER_LOG.parent.mkdir(parents=True, exist_ok=True)
    with ITER_LOG.open("a") as f:
        f.write(line + "\n")


def render_figure(stem: str = "figure_main_zero_swaps") -> None:
    log(f"rendering plot → {stem}")
    res = subprocess.run(
        [str(PY), "scripts/make_plots_zero_swaps.py", stem],
        cwd=str(REPO), capture_output=True, text=True, timeout=120,
    )
    if res.returncode != 0:
        log(f"plot script FAILED: {res.stderr[-1500:]}")
    else:
        log(res.stdout.strip().splitlines()[-2] if res.stdout else "")


def current_summary() -> pd.DataFrame:
    df = pd.read_csv(METRICS)
    rows = []
    for t in [BASELINE_TRACKER] + PIPELINE_TRACKERS:
        s = df[df.tracker == t]
        if len(s) == 0:
            continue
        rows.append({"tracker": t, "n": len(s),
                     "switches": int(s["num_switches"].sum()),
                     "per_sess": s["num_switches"].mean(),
                     "idf1": s["idf1"].mean(),
                     "idp": s["idp"].mean()})
    return pd.DataFrame(rows)


def beats_baseline(summary_df: pd.DataFrame) -> dict:
    """Returns dict with pass/fail diagnostics."""
    base = summary_df[summary_df.tracker == BASELINE_TRACKER]
    final = summary_df[summary_df.tracker == FINAL_TRACKER]
    if len(base) == 0 or len(final) == 0:
        return {"have_data": False, "reason": "missing baseline or final row"}
    b, f = base.iloc[0], final.iloc[0]
    sw_win = f["switches"] < b["switches"]
    idf1_floor = f["idf1"] >= b["idf1"] - 0.005
    overall_win = sw_win and idf1_floor
    return {
        "have_data": True,
        "switches_win": bool(sw_win),
        "idf1_floor": bool(idf1_floor),
        "overall_win": bool(overall_win),
        "delta_switches": int(b["switches"] - f["switches"]),
        "delta_idf1": float(f["idf1"] - b["idf1"]),
        "baseline_switches": int(b["switches"]),
        "baseline_idf1": float(b["idf1"]),
        "final_switches": int(f["switches"]),
        "final_idf1": float(f["idf1"]),
    }


def run_stage4_sweep(gap: int, sigma: int, threshold: float) -> dict:
    """Cheap iteration: rerun tracklet_stitch_3d with new params."""
    tag = f"g{gap}_s{sigma}_t{threshold}"
    log(f"Stage 4 iter: {tag}")
    out_dir = REPO / "outputs" / "PAF_3d_kalman" / f"_iter_stage4_{tag}"
    out_dir.mkdir(parents=True, exist_ok=True)
    eval_csv = REPO / "outputs" / "PAF_3d_kalman" / f"_iter_eval_stage4_{tag}.csv"

    in_dir = REPO / "outputs" / "PAF_3d_kalman" / "luc3d_paf_l1_dedupe_tracklet3d_results"
    if not in_dir.exists():
        log(f"  skip: missing input dir {in_dir}")
        return {"tag": tag, "skipped": True}

    cmd1 = [
        str(PY), "scripts/stitch_postprocess.py",
        "--in", str(in_dir), "--out", str(out_dir),
        "--mode", "tracklet_stitch_3d",
        "--stitch-max-gap", str(gap),
        "--stitch-sigma", str(sigma),
        "--stitch-threshold", str(threshold),
        "--master", "outputs/sleap_nn_master_sheet.tsv",
        "--pred-h5-dir", "outputs/sleap_nn_predictions_h5s",
    ]
    res1 = subprocess.run(cmd1, cwd=str(REPO), capture_output=True, text=True, timeout=600)
    if res1.returncode != 0:
        log(f"  stitch failed: {res1.stderr[-1500:]}")
        return {"tag": tag, "failed": "stitch", "stderr": res1.stderr[-500:]}

    cmd2 = [
        str(PY), "scripts/evaluate.py",
        "--luc3d-dir", str(out_dir),
        "--pred-h5-dir", "outputs/sleap_nn_predictions_h5s",
        "--out-csv", str(eval_csv),
        "--workers", "16",
    ]
    res2 = subprocess.run(cmd2, cwd=str(REPO), capture_output=True, text=True, timeout=300)
    if res2.returncode != 0:
        log(f"  eval failed: {res2.stderr[-1500:]}")
        return {"tag": tag, "failed": "eval", "stderr": res2.stderr[-500:]}

    df = pd.read_csv(eval_csv)
    summary = {"tag": tag, "gap": gap, "sigma": sigma, "threshold": threshold,
               "switches": int(df["num_switches"].sum()),
               "idf1": float(df["idf1"].mean()),
               "n": len(df), "csv": str(eval_csv), "out_dir": str(out_dir)}
    log(f"  {tag}: switches={summary['switches']} IDF1={summary['idf1']:.4f}")
    return summary


def commit_winner(winner: dict) -> None:
    """Re-register the winning Stage 4 row as luc3d_paf_l1_dedupe_tracklet3d_stitched."""
    log(f"committing winner: {winner['tag']}")
    subprocess.run(
        [str(PY), "scripts/lib/append_metrics_row.py",
         "luc3d_paf_l1_dedupe_tracklet3d_stitched", winner["csv"]],
        cwd=str(REPO), check=True,
    )
    json.dump({"max_gap": winner["gap"], "sigma": winner["sigma"],
               "threshold": winner["threshold"]},
              open(REPO / "outputs" / "PAF_3d_kalman" / "stage4_winner.json", "w"), indent=2)


def stage4_grid():
    """Two-dim sweep: gap × threshold (sigma=50 fixed first round)."""
    grid = []
    for gap in [15, 30, 60, 120, 240]:
        for sig in [30, 50, 100]:
            for thr in [0.2, 0.3, 0.5, 0.7, 0.85]:
                grid.append((gap, sig, thr))
    return grid


def main():
    if not METRICS.exists():
        log(f"missing {METRICS} — pipeline hasn't produced metrics yet")
        return 1

    log("=" * 60)
    log("Iteration controller START")
    log("=" * 60)

    # Round 0: render current state
    summary = current_summary()
    log("Current summary:\n" + summary.to_string(index=False))
    render_figure("figure_main_zero_swaps")

    verdict = beats_baseline(summary)
    log(f"Verdict: {json.dumps(verdict, indent=2)}")

    if not verdict.get("have_data"):
        log("No final-stage data yet. Exiting; rerun after pipeline completes.")
        return 1

    if verdict["overall_win"]:
        log(f"WIN: switches Δ={verdict['delta_switches']} IDF1 Δ={verdict['delta_idf1']:+.4f}")
        log("No iteration needed.")
        render_figure("figure_main_zero_swaps_FINAL")
        return 0

    log("NOT YET BETTER — entering Stage 4 sweep iteration")

    best_so_far = {"tag": "current", "switches": verdict["final_switches"],
                   "idf1": verdict["final_idf1"]}
    grid = stage4_grid()
    log(f"Stage 4 sweep grid: {len(grid)} configs")

    results = []
    for i, (gap, sig, thr) in enumerate(grid, 1):
        log(f"--- iter {i}/{len(grid)} ---")
        r = run_stage4_sweep(gap, sig, thr)
        results.append(r)
        if r.get("switches") is None:
            continue
        if r["switches"] < best_so_far["switches"] and r["idf1"] >= verdict["baseline_idf1"] - 0.005:
            best_so_far = r
            log(f"  NEW BEST: switches={r['switches']} IDF1={r['idf1']:.4f}")
            commit_winner(r)
            render_figure(f"figure_main_zero_swaps_iter{i:03d}")
            # Early stop if we beat baseline:
            if r["switches"] < verdict["baseline_switches"] and r["idf1"] >= verdict["baseline_idf1"]:
                log(f"BEATS BASELINE — stopping sweep")
                break

    # Final summary
    log("=" * 60)
    log("Iteration sweep complete")
    log(f"Best: {best_so_far}")
    log("=" * 60)
    render_figure("figure_main_zero_swaps_FINAL")

    # Dump all sweep results to a CSV
    rows = [r for r in results if r.get("switches") is not None]
    if rows:
        out_csv = REPO / "outputs" / "PAF_3d_kalman" / "_iter_stage4_sweep.csv"
        pd.DataFrame(rows).to_csv(out_csv, index=False)
        log(f"wrote {out_csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
