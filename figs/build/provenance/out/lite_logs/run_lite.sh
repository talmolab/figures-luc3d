#!/usr/bin/env bash
# Light-filter sweep, parallel: each variant builds its pool then runs LUC3D on the 42
# multi-animal sessions. Outputs tagged; no base deposit is touched.
set -uo pipefail
cd /root/vast/eric/sleap-3d-gui/scratch/repos/lucid
export PATH=/root/vast/eric/node22_env/bin:$PATH
export PYTHONPATH=/tmp/claude-0/-root-vast-eric-sleap-3d-gui-scratch-repos-lucid/44adbb8d-7377-41ea-ab2c-5dae7a320d17/scratchpad/cv2h
L=figs/out/lite_logs
one() {
  v=$1
  # vis4 was started by the first chain; wait for that process instead of racing it
  while pgrep -f "build_lite_pools.py $v\$" >/dev/null; do sleep 10; done
  /root/vast/eric/luc3d-bench/lp3d_env/bin/python figs/build_lite_pools.py $v > $L/build_$v.log 2>&1
  echo "[$(date +%T)] built $v"
  /root/vast/eric/luc3d-bench/liezl_env/bin/python figs/fig4_slap2m.py --pool lite_$v --configs sync_stale20_dist25 --multi-only --tag lite --workers 10 > $L/track_$v.log 2>&1
  echo "[$(date +%T)] tracked $v exit=$?"
}
for v in vis4 vis6 score07 vis4_or_score07 vis6_and_score08 vis8_and_score085; do one $v & done
wait
echo "[$(date +%T)] all done"
