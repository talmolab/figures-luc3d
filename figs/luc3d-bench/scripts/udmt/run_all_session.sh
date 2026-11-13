#!/usr/bin/env bash
# Run the FULL UDMT pipeline on all 6 bench cameras of one session, 2 at a time
# (one per A40). Round 1: back,backL  Round 2: mid,midL  Round 3: top,topL.
# Usage: run_all_session.sh <session> <session_path>   e.g. 10072022145420 2022-10-07/10072022145420
set -uo pipefail
SESSION="$1"; SPATH="$2"
ROOT=/root/talmolab-smb/eric/slap_2m/$SPATH
LOGDIR=/root/vast/eric/luc3d-bench/logs
HERE=$(dirname "$0")
CAMS=(back backL mid midL top topL)

resolve_video(){ local cam="$1"; ls "$ROOT/$cam/$cam-"*_h265_CRF12_denoised.mp4 2>/dev/null | head -1; }

run_cam(){
  local cam="$1" gpu="$2"
  local vid; vid=$(resolve_video "$cam")
  if [ -z "$vid" ]; then echo "NO VIDEO for $cam" >> "$LOGDIR/udmt_${SESSION}_${cam}.log"; return 1; fi
  echo "[$(date)] START $cam on GPU$gpu : $vid" >> "$LOGDIR/udmt_${SESSION}_${cam}.log"
  bash "$HERE/run_udmt_camera.sh" "$SESSION" "$cam" "$vid" "$gpu" \
    >> "$LOGDIR/udmt_${SESSION}_${cam}.log" 2>&1
  echo "[$(date)] DONE $cam rc=$?" >> "$LOGDIR/udmt_${SESSION}_${cam}.log"
}

for pair in "back backL" "mid midL" "top topL"; do
  set -- $pair
  run_cam "$1" 0 &
  P1=$!
  run_cam "$2" 1 &
  P2=$!
  wait $P1 $P2
  echo "[$(date)] round done: $pair"
done
echo "[$(date)] ALL CAMERAS DONE for $SESSION"
