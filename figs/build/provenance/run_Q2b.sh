#!/usr/bin/env bash
# md56 Q2b (2026-10-14): Fig 2G timing redone in ONE quiet sitting. The 11:00 attempt ran at
# load up to ~17 (LUC3D DLT rounds 1.17-2.15 us) and its two companions failed / were missing
# (fig2_anipose_optim_retime: missing `import os`; fig2g_retime_extra: not run). Kept as
# md56_logs/fig2g_retime_all_md56.loaded-attempt.json. Load sampled every 10 s.
set -uo pipefail
cd /root/vast/eric/sleap-3d-gui/scratch/repos/lucid
L=figs/out/md56_logs
export PATH=/root/vast/eric/node26_env/bin:$PATH NODE=/root/vast/eric/node26_env/bin/node
export FIG2_SUFFIX=md56 POSE_DIR=/root/vast/eric/lucid-main-md56/pose NUMBA_NUM_THREADS=1
AP=/root/vast/eric/luc3d-bench/anipose072_cv413_env/bin/python
t() { echo "[$(date +%T)] $* (load $(cut -d' ' -f1 /proc/loadavg))"; }
( while :; do echo "$(date +%T) $(cut -d' ' -f1-3 /proc/loadavg)"; sleep 10; done ) > $L/Q2b_load.log &
LP=$!
t start
$AP figs/fig2g_retime_all.py --rounds 5 > $L/Q2b_retime.log 2>&1; t "retime_all exit=$?"
$AP figs/fig2g_retime_extra.py > $L/Q2b_extra.log 2>&1; t "retime_extra exit=$?"
$AP figs/fig2_anipose_optim_retime.py > $L/Q2b_optim.log 2>&1; t "optim_retime exit=$?"
kill $LP
t "max 1-min load during sitting: $(awk '{print $2}' $L/Q2b_load.log | sort -n | tail -1)"
t "Q2b DONE"
