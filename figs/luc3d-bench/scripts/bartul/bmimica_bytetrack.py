#!/usr/bin/env python
"""Stage 4b: ByteTrack per (session, camera) on the per-session canonical det H5.
Output: outputs/bmimica/results/bytetrack/{session}/{cam}.h5. Resumable + parallel.
Run with the .venv python (has supervision)."""
import os, glob, sys
from concurrent.futures import ProcessPoolExecutor, as_completed
import h5py, numpy as np

sys.path.insert(0, "/root/vast/eric/luc3d-bench/scripts")
from run_bytetrack_bench import run_camera_session

BM = "/root/vast/eric/luc3d-bench/outputs/bmimica"
DET = f"{BM}/det_h5"
OUT = f"{BM}/results/bytetrack"
SER = ["21241563", "21369048", "21372315", "21372316", "22085397"]


def job(args):
    sid, cam = args
    src = f"{DET}/{sid}/{cam}_predictions.h5"
    dstdir = f"{OUT}/{sid}"; os.makedirs(dstdir, exist_ok=True)
    dst = f"{dstdir}/{cam}.h5"
    if os.path.exists(dst) and os.path.getsize(dst) > 500:
        return (sid, cam, "skip")
    if not os.path.exists(src):
        return (sid, cam, "no_src")
    try:
        arr = run_camera_session(src, 0)
        with h5py.File(dst, "w") as f:
            f.create_dataset("tracks", data=arr, dtype="float64", compression="gzip", compression_opts=4)
        return (sid, cam, "ok")
    except Exception as e:
        return (sid, cam, "ERR:" + str(e)[:150])


def main():
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument("--workers", type=int, default=12); a = ap.parse_args()
    sids = sorted(os.path.basename(d) for d in glob.glob(f"{DET}/*") if os.path.isdir(d))
    jobs = [(s, c) for s in sids for c in SER]
    print(f"bytetrack: {len(sids)} sessions, {len(jobs)} jobs", flush=True)
    ok = 0
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for f in as_completed([ex.submit(job, j) for j in jobs]):
            sid, cam, st = f.result()
            if st in ("ok", "skip"): ok += 1
            else: print(f"  {sid}/{cam}: {st}", flush=True)
    print(f"DONE bytetrack {ok}/{len(jobs)}", flush=True)


if __name__ == "__main__":
    main()
