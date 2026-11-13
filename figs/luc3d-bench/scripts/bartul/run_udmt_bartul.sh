#!/usr/bin/env bash
# Run UDMT on all 5 Bartul cameras, 9000-frame window, 2-animal, split across 2 GPUs.
set -uo pipefail
SESSION=20250827_141755
SD=/root/vast/eric/bartul-data/$SESSION
PROJ=/root/vast/eric/luc3d-bench/outputs/bartul_smoke/UDMT_runs_ep20
LOG=/root/vast/eric/luc3d-bench/outputs/bartul_smoke/results/udmt_logs
mkdir -p "$PROJ" "$LOG"
RUN=/root/vast/eric/luc3d-bench/scripts/udmt/run_udmt_camera.sh

# GPU 0 gets cams 0,2,4 ; GPU 1 gets cams 1,3
CAMS=(21241563 21369048 21372315 21372316 22085397)

run_group () {  # $1=gpu  $2..=cams
  local gpu=$1; shift
  for cam in "$@"; do
    local vid; vid=$(ls "$SD/$cam"/*.mp4)
    echo "[$(date +%T)] GPU$gpu START $cam"
    "$RUN" "$SESSION" "$cam" "$vid" "$gpu" \
      --proj-root "$PROJ" --num-animals 2 --max-frames 9000 --epochs 20 --batch 8 \
      > "$LOG/$cam.log" 2>&1
    echo "[$(date +%T)] GPU$gpu DONE  $cam (exit $?)"
  done
}

run_group 0 "${CAMS[0]}" "${CAMS[2]}" "${CAMS[4]}" &
P0=$!
run_group 1 "${CAMS[1]}" "${CAMS[3]}" &
P1=$!
wait $P0 $P1
echo "[$(date +%T)] ALL UDMT CAMERAS DONE"
