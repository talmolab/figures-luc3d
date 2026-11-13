#!/usr/bin/env python
"""Run the 3D-MuPPET + re-ID STEELMAN (muppet_run_reid.py) on all BMimica sessions.

Same harness as bmimica_muppet.py -- same detection pool, same calibration, same
num_animals=2 / dmin=100, resumable, parallel -- so the only difference from the
faithful port's run is the re-linking policy under test. Output:
outputs/bmimica/results/muppet_reid/{session}.json.

`--det-root` swaps the detection pool, which is the second steelman: pointing it at
outputs/bmimica/sleap_h5 feeds MuPPET SLEAP's OWN 2-track output instead of the
4-slot shared detection pool (Eric, 2026-08-17: "just force SLEAP to output the 2
tracks and then input that into 3D muppet"). `--runner` swaps the tracker, so the
2x2 (port | re-ID) x (pool | SLEAP-2track) is all reachable from here.

    /root/vast/eric/muppet_env/bin/python scripts/bartul/bmimica_muppet_reid.py
"""
import argparse
import glob
import os
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed

REPO = "/root/vast/eric/luc3d-bench"
BM = f"{REPO}/outputs/bmimica"
BMROOT = "/root/vast/eric/BMimica"
MUPPY = "/root/vast/eric/muppet_env/bin/python"
SER = ["21241563", "21369048", "21372315", "21372316", "22085397"]


def job(a, sid):
    out = f"{a.out_dir}/{sid}.json"
    if os.path.exists(out) and os.path.getsize(out) > 200:
        return (sid, "skip")
    if any(not os.path.exists(f"{a.det_root}/{sid}/{c}{a.det_suffix}.h5") for c in SER):
        return (sid, "no_det")
    calib = glob.glob(f"{BMROOT}/{sid}/calibration/*_calibration.toml")
    if not calib:
        return (sid, "no_calib")
    os.makedirs(a.out_dir, exist_ok=True)
    cmd = [MUPPY, f"{REPO}/scripts/bartul/{a.runner}",
           "--det-dir", f"{a.det_root}/{sid}", "--calib", calib[0],
           "--cameras", ",".join(SER), "--out", out,
           "--num-animals", "2", "--dmin", "100"]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=a.timeout)
        if r.returncode != 0 or not os.path.exists(out):
            return (sid, "FAIL:" + (r.stderr or "")[-250:].replace("\n", " "))
        tail = [ln for ln in r.stdout.strip().splitlines() if ln][-1:]
        return (sid, "ok " + (tail[0] if tail else ""))
    except Exception as e:
        return (sid, "ERR:" + str(e)[:150])


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--runner", default="muppet_run_reid.py",
                    choices=("muppet_run_reid.py", "muppet_run.py"))
    ap.add_argument("--det-root", default=f"{BM}/det_h5",
                    help="detection pool root (…/sleap_h5 for the 2-track arm)")
    ap.add_argument("--det-suffix", default="_predictions",
                    help="filename stem after the serial")
    ap.add_argument("--out-dir", default=f"{BM}/results/muppet_reid")
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--timeout", type=int, default=7200)
    a = ap.parse_args()

    sids = sorted(os.path.basename(d) for d in glob.glob(f"{a.det_root}/*")
                  if os.path.isdir(d))
    print(f"{a.runner} on {len(sids)} sessions from {a.det_root} "
          f"-> {a.out_dir}, {a.workers} workers", flush=True)
    ok = 0
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for f in as_completed([ex.submit(job, a, s) for s in sids]):
            sid, st = f.result()
            if st.startswith(("ok", "skip")):
                ok += 1
                print(f"  {sid}: {st}", flush=True)
            else:
                print(f"  {sid}: {st}", flush=True)
    print(f"DONE {ok}/{len(sids)}", flush=True)


if __name__ == "__main__":
    main()
