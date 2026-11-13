#!/usr/bin/env python
"""Run the REAL LUC3D tracker (talmolab/luc3d, pose/tracker.js + pose/cross-view-tracker.js,
via the repo's own headless driver scripts/bench/bench_crossview.mjs) on all BMimica
sessions that have a detection H5 built. Replaces the old python "Liezl" reimplementation
and the old broken flat-JS snapshot ("LUC3D (3D)") as the benchmark's multi-view-tracker bar.

bench_crossview.mjs's default thresholds already bake in corr3dWeight=6 (the champion
config), so no extra tuning flags are needed — this is a straight port of the on-pool
Liezl config into the production tracker.

Output: outputs/bmimica/results/luc3d_real/{session}.json (same schema the old Liezl/MuPPET
JSONs use: {frames:[{frame, assignments:[["cam:slot", global_id], ...]}], ...}).
Resumable + parallel (one node process per session; each session is single-camera-pooled
internally so this parallelizes across sessions, not within one).
"""
import os, glob, sys, subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed

REPO = "/root/vast/eric/luc3d-bench"
BM = f"{REPO}/outputs/bmimica"
DET = f"{BM}/det_h5"
OUT = f"{BM}/results/luc3d_real"
BMROOT = "/root/vast/eric/BMimica"
LUCID = "/root/vast/eric/sleap-3d-gui/scratch/repos/lucid"
DRIVER = f"{LUCID}/scripts/bench/bench_crossview.mjs"
SER = ["21241563", "21369048", "21372315", "21372316", "22085397"]


def job(sid):
    out = f"{OUT}/{sid}.json"
    if os.path.exists(out) and os.path.getsize(out) > 200:
        return (sid, "skip")
    if any(not os.path.exists(f"{DET}/{sid}/{c}_predictions.h5") for c in SER):
        return (sid, "no_det")
    calib = glob.glob(f"{BMROOT}/{sid}/calibration/*_calibration.toml")
    if not calib:
        return (sid, "no_calib")
    os.makedirs(OUT, exist_ok=True)
    cmd = ["node", DRIVER,
           "--session-idx", "0", "--num-animals", "2",
           "--calibration", calib[0],
           "--pred-h5-dir", f"{DET}/{sid}",
           "--out", out,
           "--cameras", ",".join(SER)]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
        if r.returncode != 0 or not os.path.exists(out):
            return (sid, "FAIL:" + r.stderr[-300:].replace("\n", " "))
        return (sid, "ok")
    except Exception as e:
        return (sid, "ERR:" + str(e)[:200])


def main():
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument("--workers", type=int, default=16); a = ap.parse_args()
    sids = sorted(os.path.basename(d) for d in glob.glob(f"{DET}/*") if os.path.isdir(d))
    print(f"luc3d_real: {len(sids)} sessions with det H5, {a.workers} workers", flush=True)
    ok = 0
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for f in as_completed([ex.submit(job, s) for s in sids]):
            sid, st = f.result()
            if st in ("ok", "skip"): ok += 1
            else: print(f"  {sid}: {st}", flush=True)
    print(f"DONE luc3d_real {ok}/{len(sids)}", flush=True)


if __name__ == "__main__":
    main()
