#!/usr/bin/env bash
# Wrapper that sets the full headless UDMT env, then runs the camera driver.
# Usage: run_udmt_camera.sh <session> <cam> <video> [gpu] [extra args...]
set -euo pipefail
SESSION="$1"; CAM="$2"; VIDEO="$3"; GPU="${4:-0}"; shift 4

export CUDA_HOME=/root/miniforge3/envs/udmt
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib:${LD_LIBRARY_PATH:-}
export QT_QPA_PLATFORM=offscreen
export PYTHONPATH=/root/vast/eric/UDMT
export TORCH_CUDA_ARCH_LIST=8.6
export CUDA_VISIBLE_DEVICES=$GPU
export PYTHONUNBUFFERED=1
export XDG_RUNTIME_DIR=/tmp/runtime-udmt-$GPU
mkdir -p "$XDG_RUNTIME_DIR"; chmod 700 "$XDG_RUNTIME_DIR"

exec /root/miniforge3/envs/udmt/bin/python \
  /root/vast/eric/luc3d-bench/scripts/udmt/run_udmt_camera.py \
  --session "$SESSION" --cam "$CAM" --video "$VIDEO" "$@"
