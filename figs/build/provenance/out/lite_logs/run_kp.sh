#!/usr/bin/env bash
# Missing keypoints (tail excluded) and keypoints seen by <=1 camera, per detection pool.
set -uo pipefail
cd /root/vast/eric/sleap-3d-gui/scratch/repos/lucid
export PYTHONPATH=/tmp/claude-0/-root-vast-eric-sleap-3d-gui-scratch-repos-lucid/44adbb8d-7377-41ea-ab2c-5dae7a320d17/scratchpad/cv2h
PY=/root/vast/eric/luc3d-bench/lp3d_env/bin/python
L=figs/out/lite_logs; T=figs/out/tmp
declare -A POOL=( [predslp]=$PWD/$T/predslp_pool [keeptrack]=/root/vast/eric/luc3d-bench/outputs/keeptrack_h5s
  [bvis4]=$PWD/$T/lite_pools/bvis4 [bvis5]=$PWD/$T/lite_pools/bvis5 [bvis6]=$PWD/$T/lite_pools/bvis6 [bvis7]=$PWD/$T/lite_pools/bvis7 )
one() {
  n=$1; p=${POOL[$n]}
  FIG4_POOL=$p $PY figs/fig4_detections.py --jobs 6 --exclude-nodes TailTip,Tail_0,Tail_1,Tail_2 \
     --out figs/out/kp_detections_${n}_notail.json > $L/kp_det_$n.log 2>&1
  FIG4_POOL=$p FIG4_REC_TAG=${n}_kp $PY figs/fig4_recovery_pernode.py --workers 6 > $L/kp_rec_$n.log 2>&1
  echo "[$(date +%T)] $n done"
}
for n in predslp keeptrack bvis4 bvis5 bvis6 bvis7; do one $n & done
wait; echo "[$(date +%T)] all done"
