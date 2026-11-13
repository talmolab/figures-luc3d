#!/usr/bin/env bash
# Q2d (2026-10-15): Fig 2G re-timed with EVERY solver on the same 200,010 keypoints
# (first 13,334 frames x 15 joints of fig4_input_md56). Same protocol as Q2c otherwise.
set -uo pipefail
cd /root/vast/eric/sleap-3d-gui/scratch/repos/lucid
L=figs/out/md56_logs
export PATH=/root/vast/eric/node26_env/bin:$PATH NODE=/root/vast/eric/node26_env/bin/node
export FIG2_SUFFIX=md56 POSE_DIR=/root/vast/eric/lucid-main-md56/pose FIG2G_FRAMES=13334
export NUMBA_NUM_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
       VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 UV_THREADPOOL_SIZE=1
AP=/root/vast/eric/luc3d-bench/anipose072_cv413_env/bin/python
t() { echo "[$(date +%T)] $* (load $(cut -d' ' -f1 /proc/loadavg))"; }
W=0; until awk '{exit !($1 < 1.0)}' /proc/loadavg || [ $W -ge 0 ]; do sleep 15; W=$((W+15)); done
START=$(cut -d' ' -f1 /proc/loadavg); t "start (waited ${W}s)"
( while :; do p=$(ps -eo pid,comm,args | awk '$2 ~ /python|node/ && /fig2g_retime|fig2_solvers_time_luc3d/ {print $1}' | head -1)
    th=$([ -n "$p" ] && ls /proc/$p/task 2>/dev/null | wc -l || echo 0)
    echo "$(date +%T) $(cut -d' ' -f1-3 /proc/loadavg) threads=$th"; sleep 5; done ) > $L/Q2d_load.log &
LP=$!
$AP figs/fig2g_retime_all.py --rounds 5 > $L/Q2d_retime.log 2>&1; t "retime_all exit=$?"
$AP figs/fig2g_retime_extra.py > $L/Q2d_extra.log 2>&1; t "retime_extra exit=$?"
kill $LP
MAXL=$(awk '{print $2}' $L/Q2d_load.log | sort -n | tail -1); MAXT=$(sed 's/.*threads=//' $L/Q2d_load.log | sort -n | tail -1)
$AP - <<PY
import json; p = "figs/out/fig2g_retime_all_md56.json"; d = json.load(open(p))
d["sitting"] = {"start_1min_load": $START, "max_1min_load": $MAXL, "max_threads_timing_process": $MAXT, "cores": 64,
    "thread_env": "NUMBA/OMP/OPENBLAS/MKL/VECLIB/NUMEXPR_NUM_THREADS=1, UV_THREADPOOL_SIZE=1",
    "code": "LUC3D v1.0.5 (2dc001fe)", "keypoints": "every solver on the same first 13,334 frames x 15 joints = 200,010",
    "note": "md56 Q2e (clean rerun of Q2d), 2026-10-15"}
json.dump(d, open(p, "w"), indent=1)
PY
t "max 1-min load $MAXL, max threads $MAXT"
t "Q2d DONE"
