"""Anipose end to end on one recording -- figs/fig7_calib_anipose.py, instrumented for the
apples-to-apples rerun. Detector, board and solver are UNCHANGED (stock aniposelib 0.8.0).

Differences from fig7_calib_anipose.py, all bookkeeping or budget, none algorithmic:
  * the process pool and OpenCV threads come from the CORE BUDGET the caller pins with
    taskset (CORES), not from a hard-coded per-rig number: POOL = min(cameras, CORES)
    processes x OPENCV_THREADS = max(1, CORES // cameras) threads, so detection fills the
    SAME cpuset calibrat3 gets, once;
  * `detection_s_cpu` is real CPU time (user+sys, per worker, from os.times()). The old
    field of that name was the SUM OF PER-PROCESS WALL TIMES of multi-threaded workers, and
    the published CSV labelled it "single core";
  * versions, affinity and load are written into the summary.

    CORES=16 python3 anipose_run_b.py <spec.json> <outdir> [tag]
Writes <outdir>/anipose_rows[_<tag>].pkl, calibration_anipose[_<tag>].toml,
       anipose_summary[_<tag>].json
"""
import sys, os, json, time, pickle, platform
from multiprocessing import Pool

import cv2
import numpy as np
import aniposelib
from aniposelib.boards import CharucoBoard
from aniposelib.cameras import CameraGroup

spec_path, out = sys.argv[1], sys.argv[2]
tag = ('_' + sys.argv[3]) if len(sys.argv) > 3 and sys.argv[3] else ''
spec = json.load(open(spec_path))
os.makedirs(out, exist_ok=True)
names = [c[0] for c in spec['cameras']]
vids = [c[1] for c in spec['cameras']]
B = spec['board']
CORES = int(os.environ.get('CORES') or len(os.sched_getaffinity(0)))
POOL = min(len(vids), CORES)
THREADS = int(os.environ.get('OPENCV_THREADS') or max(1, CORES // len(vids)))


def make_board():
    return CharucoBoard(B['board_x'], B['board_y'], B['square_length'], B['marker_length'],
                        marker_bits=B.get('marker_bits', 4), dict_size=B.get('dict_size', 1000))


def detect_one(arg):
    cv2.setNumThreads(THREADS)
    i, vid = arg
    c0, t0 = os.times(), time.time()
    rows = make_board().detect_video(vid, prefix=0, progress=False)
    c1 = os.times()
    return i, rows, time.time() - t0, (c1.user - c0.user) + (c1.system - c0.system)


if __name__ == '__main__':
    if cv2.__version__ != '4.13.0':
        raise SystemExit(f'cv2 {cv2.__version__}, not 4.13.0 -- REFUSING (calibrat3 is pinned to OpenCV.js 4.13.0)')
    print(f'cameras {names}; CORES {CORES} (affinity {sorted(os.sched_getaffinity(0))[:3]}..'
          f'{len(os.sched_getaffinity(0))} cpus); pool {POOL} x {THREADS} OpenCV threads', flush=True)
    load0 = open('/proc/loadavg').read().split()[:3]
    t0 = time.time()
    with Pool(POOL) as p:
        got = p.map(detect_one, list(enumerate(vids)), chunksize=1)
    t_det_wall = time.time() - t0
    got.sort(key=lambda g: g[0])
    rows = [g[1] for g in got]
    print(f'detection: {t_det_wall:.1f} s wall / {sum(g[3] for g in got):.0f} s CPU; '
          f'frames with board per camera: {[len(r) for r in rows]}', flush=True)
    with open(os.path.join(out, f'anipose_rows{tag}.pkl'), 'wb') as fh:
        pickle.dump({'names': names, 'rows': rows}, fh)

    cgroup = CameraGroup.from_names(names)
    for c in cgroup.cameras:
        c.set_size(tuple(spec['size']))
    t1, c1 = time.time(), os.times()
    err = cgroup.calibrate_rows(rows, make_board(), init_intrinsics=True, init_extrinsics=True, verbose=True)
    t_cal, c2 = time.time() - t1, os.times()
    print(f'anipose calibrate_rows error: {err}; time {t_cal:.1f} s', flush=True)
    cgroup.dump(os.path.join(out, f'calibration_anipose{tag}.toml'))
    json.dump({
        'names': names, 'cores': CORES, 'pool': POOL, 'opencv_threads': THREADS,
        'blas_threads': {k: os.environ.get(k) for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS')},
        'affinity': sorted(os.sched_getaffinity(0)),
        'detection_s_wall': t_det_wall,
        'detection_s_cpu': sum(g[3] for g in got),
        'detection_s_wall_per_camera': [g[2] for g in got],
        'calibration_s': t_cal,
        'calibration_s_cpu': (c2.user - c1.user) + (c2.system - c1.system),
        'error': float(err) if err is not None else None,
        'frames_per_cam': [len(r) for r in rows],
        'loadavg_start': load0, 'loadavg_end': open('/proc/loadavg').read().split()[:3],
        'versions': dict(cv2=cv2.__version__, aniposelib=getattr(aniposelib, '__version__', '0.8.0'),
                         numpy=np.__version__, python=platform.python_version()),
        'spec': os.path.abspath(spec_path), 'started_unix': t0,
    }, open(os.path.join(out, f'anipose_summary{tag}.json'), 'w'), indent=1)
    print('done', flush=True)
