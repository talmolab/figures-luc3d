#!/usr/bin/env bash
# Body-only keypoint-count sweep (tail never counted): build, track (42 multi-animal),
# then score with SCORE_NOTAIL=1. Tagged outputs only.
set -uo pipefail
cd /root/vast/eric/sleap-3d-gui/scratch/repos/lucid
export PATH=/root/vast/eric/node22_env/bin:$PATH
export PYTHONPATH=/tmp/claude-0/-root-vast-eric-sleap-3d-gui-scratch-repos-lucid/44adbb8d-7377-41ea-ab2c-5dae7a320d17/scratchpad/cv2h
L=figs/out/lite_logs
one() {
  v=$1
  /root/vast/eric/luc3d-bench/lp3d_env/bin/python figs/build_lite_pools.py $v > $L/build_$v.log 2>&1
  echo "[$(date +%T)] built $v"
  SCORE_NOTAIL=1 /root/vast/eric/luc3d-bench/liezl_env/bin/python figs/fig4_slap2m.py --pool lite_$v --configs sync_stale20_dist25 --multi-only --tag lite --workers 14 > $L/track_$v.log 2>&1
  echo "[$(date +%T)] tracked+scored $v exit=$? failed=$(grep -c FAILED $L/track_$v.log)"
}
for v in bvis4 bvis5 bvis6 bvis7; do one $v & done
wait
echo "[$(date +%T)] all done"
