#!/usr/bin/env python
"""Per-session native-SLEAP-track-slot H5, for the SLEAP-per-camera-2D reference bar
(BMimica's pipeline never built this — only the single Bartul smoke test did). Same
retracked filter+track .slp used to build det_h5, but sliced by SLEAP's OWN track
identity instead of the canonical (score-desc) detection slot, so it reflects what a
pure per-camera 2D tracker's identity looks like (camera-scoped, no cross-view link).

Output: outputs/bmimica/sleap_h5/{session}/{serial}_predictions.h5, shape (1, F, 2, 15, 2).
Resumable + parallel. Mirrors the `sleap` array in scripts/bartul/build_inputs.py.
"""
import os, glob, sys
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np, h5py, sleap_io as sio

sys.path.insert(0, "/root/vast/eric/luc3d-bench/scripts/lib")
from canonical_order import instance_xy

BM = "/root/vast/eric/luc3d-bench/outputs/bmimica"
RETR = f"{BM}/retracked"
OUT = f"{BM}/sleap_h5"
SER = ["21241563", "21369048", "21372315", "21372316", "22085397"]
N_NODES = 15
N_ANIMALS = 2


def build_cam(sid, cam):
    src = f"{RETR}/{sid}/{cam}.slp"
    dst_dir = f"{OUT}/{sid}"; os.makedirs(dst_dir, exist_ok=True)
    dst = f"{dst_dir}/{cam}_predictions.h5"
    if os.path.exists(dst) and os.path.getsize(dst) > 500:
        return "skip"
    if not (os.path.exists(src) and os.path.getsize(src) > 1000):
        return "no_src"
    L = sio.load_slp(src, open_videos=False)
    F = max((lf.frame_idx for lf in L.labeled_frames), default=-1) + 1
    if F <= 0:
        return "empty"
    track_index = {t: i for i, t in enumerate(L.tracks)}
    sleap = np.full((F, N_ANIMALS, N_NODES, 2), np.nan)
    for lf in L.labeled_frames:
        j = lf.frame_idx
        for inst in lf.instances:
            ti = track_index.get(inst.track, None)
            if ti is not None and ti < N_ANIMALS:
                sleap[j, ti] = instance_xy(inst)
    with h5py.File(dst, "w") as f:
        f.create_dataset("tracks", data=sleap[None], compression="gzip", compression_opts=1)
    return "ok"


def job(args):
    sid, cam = args
    try:
        return (sid, cam, build_cam(sid, cam))
    except Exception as e:
        return (sid, cam, "ERR:" + str(e)[:150])


def main():
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument("--workers", type=int, default=16); a = ap.parse_args()
    sids = sorted(os.path.basename(d) for d in glob.glob(f"{RETR}/*") if os.path.isdir(d))
    jobs = [(s, c) for s in sids for c in SER]
    print(f"build_sleap_ref: {len(sids)} sessions, {len(jobs)} jobs", flush=True)
    ok = 0
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for f in as_completed([ex.submit(job, j) for j in jobs]):
            sid, cam, st = f.result()
            if st in ("ok", "skip"): ok += 1
            else: print(f"  {sid}/{cam}: {st}", flush=True)
    print(f"DONE build_sleap_ref {ok}/{len(jobs)}", flush=True)


if __name__ == "__main__":
    main()
