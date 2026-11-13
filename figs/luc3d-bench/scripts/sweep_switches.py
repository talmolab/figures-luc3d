"""Focused LUC3D sweep aimed at *reducing switches* on hard sessions.

The 2026-05-17 coarse sweep over wEpipolar/wReprojection/reprojSigma/
scoreThresh/continuityWeight was effectively a no-op: in 2-animal sessions,
Hungarian on a 2x2 cost matrix always picks both off-diagonal pairs regardless
of how the costs are reweighted. Empirically the per-frame assignments were
byte-identical across 81 configs in that run; only runtime differed.

This sweep instead varies knobs that *do* gate the final per-frame identity
output even with numAnimals=2:

- `uvMaxAcceptCost` — gate on accepting a reassignment in
  `verifyUnderCountedCameras`. Higher = more aggressive reassignment when
  one camera has fewer instances than `numAnimals`.
- `uvMinMargin` — relative cost margin required to flip an assignment
  (lower = more aggressive).
- `applyMotionVerify` + `viAbsThresh` / `viRelThresh` — motion-consistency
  veto on cross-view matches.
- `interpMaxGap` — gap-fill window for short detection dropouts.

Restricted to a small hand-picked set of difficulty-6/7 multi-animal
sessions where LUC3D currently has the most switches.
"""
from __future__ import annotations

import argparse
import itertools
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

# Hand-picked hard multi-animal sessions where LUC3D currently has the worst
# IDF1 / most switches. From outputs/metrics/worst_idf1_luc3d.csv filtered to
# difficulty>=6 and num_animals>=2.
HARD_SESSIONS = [
    "10072022162233",  # diff=7, anim=2 — worst IDF1
    "10072022183527",  # diff=7, anim=2
    "10072022190807",  # diff=7, anim=3
    "10212022181154",  # diff=6, anim=2
    "10212022175626",  # diff=6, anim=2
]

# Current winner = baseline. Any sweep config inherits all baseline values
# unless explicitly overridden.
BASELINE = {
    "applyUndercountVerifyClean": True,
    "applyInterpolation": True,
    "interpMaxGap": 5,
    "uvMaxAcceptCost": 120,
    "uvMinMargin": 1.2,
    "applyMotionVerify": False,
    "viAbsThresh": 60,
    "viRelThresh": 0.6,
}


def make_screening_configs():
    """1-D screening sweep: vary one knob at a time, others held at BASELINE.

    Yields tuples (cfg_id, cfg_dict). Includes the baseline once so the run
    serves as its own control.
    """
    yield "baseline", dict(BASELINE)

    # Wider/narrower undercount acceptance
    for v in [60, 200, 300]:
        cfg = dict(BASELINE); cfg["uvMaxAcceptCost"] = v
        yield f"uvMaxAcceptCost={v}", cfg

    for v in [1.0, 1.5, 2.0]:
        cfg = dict(BASELINE); cfg["uvMinMargin"] = v
        yield f"uvMinMargin={v}", cfg

    # Motion-verify on/off + thresholds
    for v in [40, 90, 120]:
        cfg = dict(BASELINE); cfg["applyMotionVerify"] = True; cfg["viAbsThresh"] = v
        yield f"motionVerify_viAbs={v}", cfg

    # Interpolation gap
    for v in [3, 10, 20]:
        cfg = dict(BASELINE); cfg["interpMaxGap"] = v
        yield f"interpMaxGap={v}", cfg

    # Combined "aggressive switch-reduction" — turn everything more conservative
    cfg = dict(BASELINE)
    cfg.update({"uvMaxAcceptCost": 200, "uvMinMargin": 1.0,
                "applyMotionVerify": True, "viAbsThresh": 60, "interpMaxGap": 10})
    yield "aggressive_combo", cfg


def _run_single(args_tuple):
    cdir_str, params_path_str, idx_in_master, sid, num_animals, calibration, max_frames = args_tuple
    cdir = Path(cdir_str)
    out_json = cdir / f"{sid}.json"
    if out_json.exists():
        return sid, True, "skip", str(out_json)
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
    return sid, proc.returncode == 0, f"{time.time()-t0:.1f}s", str(out_json)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", default=time.strftime("switches-%Y%m%d-%H%M%S"))
    ap.add_argument("--max-frames", type=int, default=5000)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) // 2))
    args = ap.parse_args()

    df = pd.read_csv(MASTER, sep="\t", index_col=0)
    # Filter to the hand-picked hard sessions
    df["session"] = df["session"].astype(str)
    sweep_df = df[df["session"].isin(HARD_SESSIONS)].copy()
    if len(sweep_df) == 0:
        sys.exit(f"no sessions matched {HARD_SESSIONS}")

    out_root = REPO / "outputs" / "sweeps" / args.run_id
    out_root.mkdir(parents=True, exist_ok=True)
    # Per-run splits file so evaluate_sweep.py picks up the right sessions.
    splits_path = REPO / "outputs" / "splits" / f"{args.run_id}.json"
    splits_path.parent.mkdir(parents=True, exist_ok=True)
    splits_path.write_text(json.dumps({
        "sweep_sessions": sweep_df["session"].tolist(),
        "test_sessions": [],
        "note": "hand-picked hard sessions for switch-reduction sweep",
    }, indent=2))
    print(f"sweep sessions ({len(sweep_df)}):", sweep_df["session"].tolist())

    configs = list(make_screening_configs())
    print(f"running {len(configs)} configs × {len(sweep_df)} sessions = "
          f"{len(configs) * len(sweep_df)} runs (workers={args.workers}, "
          f"max-frames={args.max_frames})")

    tasks = []
    for cid, cfg in configs:
        cdir = out_root / "configs" / cid
        cdir.mkdir(parents=True, exist_ok=True)
        params_path = cdir / "params.json"
        params_path.write_text(json.dumps(cfg, indent=2))
        for idx_in_master, row in sweep_df.iterrows():
            sid = str(row["session"])
            tasks.append((str(cdir), str(params_path), int(idx_in_master), sid,
                          int(row["num_animals"]), str(row["calibration_toml"]),
                          args.max_frames))

    failures = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(_run_single, t) for t in tasks]
        for fut in tqdm(as_completed(futs), total=len(futs), desc="runs"):
            sid, ok, info, _ = fut.result()
            if not ok:
                failures.append((sid, info))

    print(f"runs done → {out_root}/configs/")
    if failures:
        print(f"{len(failures)} failed runs (first 10):")
        for sid, info in failures[:10]:
            print(f"  {sid}: {info}")
    print(f"next: uv run python scripts/evaluate_sweep.py --run-id {out_root.name}")


if __name__ == "__main__":
    main()
