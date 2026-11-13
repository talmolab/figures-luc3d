#!/usr/bin/env bash
# T2 (CONSISTENCY-PLAN-2026-10-19): Fig 2G clean retime. Same protocol as C1 (every solver on
# the same first 13,334 frames x 15 joints = 200,010 keypoints, interleaved rounds, single-threaded,
# RANSAC on the same keypoints), run with NOTHING else of ours running. Host load from outside this
# container (~12 of 64 cores on 2026-10-19) cannot be removed; it is logged every 5 s and deposited.
set -uo pipefail
cd /root/vast/eric/sleap-3d-gui/scratch/repos/lucid
L=figs/out/md56_logs; O=figs/out
AP=/root/vast/eric/luc3d-bench/anipose072_cv413_env/bin/python
MAIN=/root/vast/eric/lucid-main-md56
export PATH=/root/vast/eric/node26_env/bin:$PATH
export NUMBA_NUM_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 UV_THREADPOOL_SIZE=1
[ -e $O/fig2g_retime_all_md56T2.json ] && { echo "T2 deposit exists"; exit 1; }
ln -sfn fig4_input_md56.json $O/fig4_input_md56T2.json
( while :; do echo "$(date -u +%T) $(cut -d' ' -f1-3 /proc/loadavg)"; sleep 5; done ) > $L/T2_load.log & LP=$!
FIG2_SUFFIX=md56T2 FIG2G_FRAMES=13334 POSE_DIR=$MAIN/pose NODE=/root/vast/eric/node26_env/bin/node \
  $AP figs/fig2g_retime_all.py --rounds 5 > $L/T2_retime.log 2>&1; rc1=$?
FIG2_SUFFIX=md56T2 FIG2G_FRAMES=13334 $AP figs/fig2g_retime_extra.py > $L/T2_extra.log 2>&1; rc2=$?
kill $LP
[ $rc1 -eq 0 ] && [ $rc2 -eq 0 ] || { echo "T2 exit $rc1/$rc2"; exit 1; }
MAXL=$(awk '{print $2}' $L/T2_load.log | sort -n | tail -1); MEDL=$(awk '{print $2}' $L/T2_load.log | sort -n | awk '{a[NR]=$1} END{print a[int((NR+1)/2)]}')
$AP - <<PY
import json; p = "$O/fig2g_retime_all_md56T2.json"; d = json.load(open(p))
d["sitting"] = {"id": "T2 (CONSISTENCY-PLAN-2026-10-19)", "max_1min_load": $MAXL, "median_1min_load": $MEDL, "cores": 64,
  "idle_rule": "nothing of ours running; host load from outside this container logged, not removable",
  "load_log": "md56_logs/T2_load.log",
  "thread_env": "NUMBA/OMP/OPENBLAS/MKL/VECLIB/NUMEXPR_NUM_THREADS=1, UV_THREADPOOL_SIZE=1",
  "code": "LUC3D v1.0.5 (2dc001fe)", "keypoints": "every solver on the same first 13,334 frames x 15 joints = 200,010"}
json.dump(d, open(p, "w"), indent=1)
PY
echo "T2 DONE max load $MAXL median $MEDL"
