#!/usr/bin/env python
"""Run 3D-MuPPET (SORT + first-frame cross-view matching) on all BMimica sessions
that have a detection H5 built. Full sessions (no frame cap), num_animals=2, dmin=100.
Output: outputs/bmimica/results/muppet/{session}.json. Resumable + parallel."""
import os, glob, sys, subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed

REPO = "/root/vast/eric/luc3d-bench"
BM = f"{REPO}/outputs/bmimica"
DET = f"{BM}/det_h5"
OUT = f"{BM}/results/muppet"
BMROOT = "/root/vast/eric/BMimica"
MUPPY = "/root/vast/eric/muppet_env/bin/python"
SER = ["21241563", "21369048", "21372315", "21372316", "22085397"]


def job(sid):
    out = f"{OUT}/{sid}.json"
    if os.path.exists(out) and os.path.getsize(out) > 200:
        return (sid, "skip")
    # need all 5 det H5 for this session
    if any(not os.path.exists(f"{DET}/{sid}/{c}_predictions.h5") for c in SER):
        return (sid, "no_det")
    calib = glob.glob(f"{BMROOT}/{sid}/calibration/*_calibration.toml")
    if not calib:
        return (sid, "no_calib")
    os.makedirs(OUT, exist_ok=True)
    cmd = [MUPPY, f"{REPO}/scripts/bartul/muppet_run.py",
           "--det-dir", f"{DET}/{sid}", "--calib", calib[0],
           "--cameras", ",".join(SER), "--out", out,
           "--num-animals", "2", "--dmin", "100"]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
        if r.returncode != 0 or not os.path.exists(out):
            return (sid, "FAIL:" + r.stderr[-250:].replace("\n", " "))
        return (sid, "ok")
    except Exception as e:
        return (sid, "ERR:" + str(e)[:150])


def main():
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument("--workers", type=int, default=12); a = ap.parse_args()
    sids = sorted(os.path.basename(d) for d in glob.glob(f"{DET}/*") if os.path.isdir(d))
    print(f"muppet: {len(sids)} sessions with det H5, {a.workers} workers", flush=True)
    ok = 0
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for f in as_completed([ex.submit(job, s) for s in sids]):
            sid, st = f.result()
            if st in ("ok", "skip"): ok += 1
            else: print(f"  {sid}: {st}", flush=True)
    print(f"DONE muppet {ok}/{len(sids)}", flush=True)


if __name__ == "__main__":
    main()
