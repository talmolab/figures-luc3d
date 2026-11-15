"""Score every LUC3D sweep config against ground truth and pick a winner.

Reads outputs/sweeps/{run_id}/configs/{cfg_id}/{session}.json (produced by
sweep_params.py), runs motmetrics LUC3D-only on each, and writes:

    outputs/sweeps/{run_id}/metrics.csv      — one row per (cfg_id, session, camera)
    outputs/sweeps/{run_id}/summary.csv      — one row per cfg_id (mean IDF1, MOTA, IDSW)
    outputs/sweeps/{run_id}/winner.json      — best cfg by mean IDF1, with its params

Re-uses scripts/evaluate.py:eval_camera but throws away the SLEAP/ByteTrack
columns since they're invariant across configs.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from lib.slap2m_paths import BENCH_CAMERAS
from evaluate import eval_camera, PRED_H5_DIR

REPO = Path(__file__).resolve().parents[1]
MASTER = REPO / "outputs" / "predictions_master_sheet.tsv"
SWEEPS_DIR = REPO / "outputs" / "sweeps"
SPLIT_PATH = REPO / "outputs" / "splits" / "sweep_test_split.json"


def score_one(cfg_id: str, cfg_dir: Path, sweep_df: pd.DataFrame, master_df: pd.DataFrame) -> list[dict]:
    rows = []
    for idx_in_master, row in sweep_df.iterrows():
        sid = str(row["session"])
        luc3d_json = cfg_dir / f"{sid}.json"
        if not luc3d_json.exists():
            continue
        for cam in BENCH_CAMERAS:
            pred_h5 = PRED_H5_DIR / f"{cam}_predictions.h5"
            gt_h5_str = row[f"{cam}_proofread_h5"]
            if not isinstance(gt_h5_str, str) or not gt_h5_str:
                continue
            gt_h5 = Path(gt_h5_str)
            try:
                metrics = eval_camera(pred_h5, gt_h5, luc3d_json, None,
                                      session_idx=int(idx_in_master))
            except Exception as e:
                print(f"  [{cfg_id}/{sid}/{cam}] eval failed: {e}", file=sys.stderr)
                continue
            m = metrics.get("luc3d", {})
            rows.append({
                "cfg_id": cfg_id,
                "session": sid,
                "camera": cam,
                "difficulty": int(row["difficulty"]),
                "animals": int(row["num_animals"]),
                "idf1": m.get("idf1"),
                "mota": m.get("mota"),
                "num_switches": m.get("num_switches"),
                "num_fragmentations": m.get("num_fragmentations"),
                "idp": m.get("idp"),
                "idr": m.get("idr"),
            })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True, help="sweep run id under outputs/sweeps/")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) // 2))
    args = ap.parse_args()

    run_dir = SWEEPS_DIR / args.run_id
    if not run_dir.exists():
        sys.exit(f"missing sweep dir: {run_dir}")

    master_df = pd.read_csv(MASTER, sep="\t", index_col=0)
    master_df["session"] = master_df["session"].astype(str)
    # Prefer a per-run splits file (sweep_switches.py writes one); fall back to
    # the global SPLIT_PATH; fall back to inferring sessions from the config-dir
    # filenames.
    run_split = REPO / "outputs" / "splits" / f"{args.run_id}.json"
    if run_split.exists():
        sweep_sessions = set(json.loads(run_split.read_text())["sweep_sessions"])
        source = str(run_split)
    elif SPLIT_PATH.exists():
        sweep_sessions = set(json.loads(SPLIT_PATH.read_text())["sweep_sessions"])
        source = str(SPLIT_PATH)
    else:
        # Infer from the per-config session JSONs.
        sweep_sessions = set()
        for d in (run_dir / "configs").iterdir():
            if d.is_dir():
                sweep_sessions.update(p.stem for p in d.glob("*.json")
                                      if p.name != "params.json")
        source = "inferred from config dirs"
    sweep_df = master_df[master_df["session"].isin(sweep_sessions)]
    print(f"sweep split: {len(sweep_df)} sessions ({source})")

    cfg_dirs = sorted([d for d in (run_dir / "configs").iterdir() if d.is_dir()])
    print(f"configs to evaluate: {len(cfg_dirs)}")

    all_rows: list[dict] = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(score_one, d.name, d, sweep_df, master_df): d.name for d in cfg_dirs}
        for fut in tqdm(as_completed(futs), total=len(futs), desc="configs"):
            try:
                all_rows.extend(fut.result())
            except Exception as e:
                print(f"  config {futs[fut]} failed: {e}", file=sys.stderr)

    df = pd.DataFrame(all_rows)
    metrics_path = run_dir / "metrics.csv"
    df.to_csv(metrics_path, index=False)
    print(f"wrote {metrics_path} ({len(df)} rows)")

    if df.empty:
        print("no rows produced; aborting summary")
        return

    summary = df.groupby("cfg_id").agg(
        mean_idf1=("idf1", "mean"),
        mean_mota=("mota", "mean"),
        mean_idsw=("num_switches", "mean"),
        mean_frag=("num_fragmentations", "mean"),
        n=("idf1", "count"),
    ).reset_index().sort_values("mean_idf1", ascending=False)
    summary_path = run_dir / "summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"wrote {summary_path}")
    print("\ntop 10 configs by mean IDF1:")
    print(summary.head(10).to_string(index=False))

    # Winner
    best_cfg_id = summary.iloc[0]["cfg_id"]
    best_params_path = run_dir / "configs" / best_cfg_id / "params.json"
    best_params = json.loads(best_params_path.read_text()) if best_params_path.exists() else {}
    winner = {
        "cfg_id": best_cfg_id,
        "mean_idf1": float(summary.iloc[0]["mean_idf1"]),
        "mean_mota": float(summary.iloc[0]["mean_mota"]),
        "mean_idsw": float(summary.iloc[0]["mean_idsw"]),
        "params": best_params,
    }
    (run_dir / "winner.json").write_text(json.dumps(winner, indent=2))
    print(f"\nwrote {run_dir / 'winner.json'} → cfg_id={best_cfg_id}")


if __name__ == "__main__":
    main()
