#!/bin/bash
# Parallel inference: 2 sessions at a time, one per A40 GPU.
set -euo pipefail

REPO=/root/vast/eric/luc3d-bench
PY=$REPO/lp3d_env/bin/python
MODEL_DIR=$REPO/lp3d_models/slap2m_mvt
OUT_DIR=$REPO/outputs/lp3d_csv_preds
LOG_DIR=$REPO/logs/lp3d_inference

# Inference sessions
SESSIONS=(
  "10072022124146" "10072022125259"
  "10072022172040" "10072022175009"
  "10072022194521" "10072022200754"
  "10072022184618" "10202022163211"
  "10072022161055" "10072022162233"
)

mkdir -p "$OUT_DIR" "$LOG_DIR"

# Run pairs on alternating GPUs
NUM=${#SESSIONS[@]}
i=0
while [ $i -lt $NUM ]; do
  # GPU 0 worker
  sess0=${SESSIONS[$i]}
  gpu0_log="$LOG_DIR/${sess0}.log"
  CUDA_VISIBLE_DEVICES=0 $PY $REPO/scripts/lp3d/run_inference_one.py \
    --model-dir "$MODEL_DIR" \
    --session-id "$sess0" \
    --out-csv-root "$OUT_DIR" \
    --skip-existing > "$gpu0_log" 2>&1 &
  pid0=$!

  # GPU 1 worker (if any sessions remain)
  pid1=""
  if [ $((i+1)) -lt $NUM ]; then
    sess1=${SESSIONS[$((i+1))]}
    gpu1_log="$LOG_DIR/${sess1}.log"
    CUDA_VISIBLE_DEVICES=1 $PY $REPO/scripts/lp3d/run_inference_one.py \
      --model-dir "$MODEL_DIR" \
      --session-id "$sess1" \
      --out-csv-root "$OUT_DIR" \
      --skip-existing > "$gpu1_log" 2>&1 &
    pid1=$!
  fi

  # Wait for both
  wait $pid0
  rc0=$?
  if [ -n "$pid1" ]; then
    wait $pid1
    rc1=$?
  else
    rc1=0
  fi
  echo "[$(date +%H:%M:%S)] pair done: $sess0 (rc=$rc0)${pid1:+, $sess1 (rc=$rc1)}"
  i=$((i+2))
done

echo "All inference done."
