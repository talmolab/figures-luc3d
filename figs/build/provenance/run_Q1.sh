#!/usr/bin/env bash
# md56 Q1 (2026-10-14): Fig 2F/G Anipose ACCURACY arm, all 56 sessions, aniposelib 0.7.2 +
# OpenCV 4.13.0. Not a timed number in the figure (Fig 2G reads fig2g_retime_all, phase Q2).
set -uo pipefail
cd /root/vast/eric/sleap-3d-gui/scratch/repos/lucid
L=figs/out/md56_logs
export PATH=/root/vast/eric/node26_env/bin:$PATH NODE=/root/vast/eric/node26_env/bin/node
export FIG2_SUFFIX=md56 POSE_DIR=/root/vast/eric/lucid-main-md56/pose NUMBA_NUM_THREADS=1
AP=/root/vast/eric/luc3d-bench/anipose072_cv413_env/bin/python
t() { echo "[$(date +%T)] $* (load $(cut -d' ' -f1 /proc/loadavg))"; }
until grep -q "measure exit" $L/run_F2a.out 2>/dev/null; do sleep 60; done   # needs fig4_md56.json
t "start"
$AP figs/fig2_solvers_anipose.py --jobs 12 --optim-jobs 12 \
  --optim-sweep 1000 2000 4000 8000 16000 23000 --out figs/out/fig4_anipose_md56.json > $L/Q1_anipose.log 2>&1
t "Q1 anipose exit=$?"
