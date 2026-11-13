#!/usr/bin/env bash
# md56 Q2c (2026-10-14): Fig 2G timing, ONE sitting, every solver genuinely single-threaded.
# Q2 and Q2b only pinned numba; NumPy/SciPy's BLAS/OpenMP pools ran ~190 threads on 64 cores
# (load up to 39), and Q2b also overlapped another job. Here: all thread pools = 1, start only
# once the 1-min load < 1, load and the timing process's thread count sampled every 5 s.
set -uo pipefail
cd /root/vast/eric/sleap-3d-gui/scratch/repos/lucid
L=figs/out/md56_logs
export PATH=/root/vast/eric/node26_env/bin:$PATH NODE=/root/vast/eric/node26_env/bin/node
export FIG2_SUFFIX=md56 POSE_DIR=/root/vast/eric/lucid-main-md56/pose
export NUMBA_NUM_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
       VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 UV_THREADPOOL_SIZE=1
AP=/root/vast/eric/luc3d-bench/anipose072_cv413_env/bin/python
t() { echo "[$(date +%T)] $* (load $(cut -d' ' -f1 /proc/loadavg))"; }
until awk '{exit !($1 < 1.0)}' /proc/loadavg; do sleep 15; done
t "idle; start"
( while :; do p=$(ps -eo pid,comm,args | awk '$2 ~ /python|node/ && /fig2g_retime|fig2_anipose_optim|fig2_solvers_time_luc3d/ {print $1}' | head -1)
    th=$([ -n "$p" ] && ls /proc/$p/task 2>/dev/null | wc -l || echo 0)
    echo "$(date +%T) $(cut -d' ' -f1-3 /proc/loadavg) threads=$th"; sleep 5; done ) > $L/Q2c_load.log &
LP=$!
$AP figs/fig2g_retime_all.py --rounds 5 > $L/Q2c_retime.log 2>&1; t "retime_all exit=$?"
$AP figs/fig2g_retime_extra.py > $L/Q2c_extra.log 2>&1; t "retime_extra exit=$?"
$AP figs/fig2_anipose_optim_retime.py > $L/Q2c_optim.log 2>&1; t "optim_retime exit=$?"
kill $LP
MAXL=$(awk '{print $2}' $L/Q2c_load.log | sort -n | tail -1); MAXT=$(sed 's/.*threads=//' $L/Q2c_load.log | sort -n | tail -1)
$AP - <<PY
import json; p = "figs/out/fig2g_retime_all_md56.json"; d = json.load(open(p))
d["sitting"] = {"max_1min_load": $MAXL, "max_threads_timing_process": $MAXT, "cores": 64,
    "thread_env": "NUMBA/OMP/OPENBLAS/MKL/VECLIB/NUMEXPR_NUM_THREADS=1, UV_THREADPOOL_SIZE=1",
    "code": "origin/main 2dc001fe", "note": "md56 Q2c, 2026-10-14: started at 1-min load < 1, nothing else running"}
json.dump(d, open(p, "w"), indent=1)
PY
t "max 1-min load $MAXL, max threads in timing process $MAXT"
t "Q2c DONE"
