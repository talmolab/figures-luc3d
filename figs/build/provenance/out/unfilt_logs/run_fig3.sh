#!/usr/bin/env bash
# Fig 3b: head-to-head + quality on the UNFILTERED SLAP-2M pool, no match gate (eric/figs has none).
set -uo pipefail
cd /root/vast/eric/sleap-3d-gui/scratch/repos/lucid
export PATH=/root/vast/eric/node26_env/bin:$PATH
export FIG3_VARIANT=unfilt HH_JOBS=32
export PYTHONPATH=/tmp/claude-0/-root-vast-eric-sleap-3d-gui-scratch-repos-lucid/44adbb8d-7377-41ea-ab2c-5dae7a320d17/scratchpad/cv2h
PY=/root/vast/eric/luc3d-bench/liezl_env/bin/python
L=figs/out/unfilt_logs
echo "[$(date +%T)] headtohead start"; $PY figs/fig3_headtohead.py > $L/fig3_headtohead.log 2>&1; echo "[$(date +%T)] headtohead exit=$?"
$PY figs/fig3_quality.py > $L/fig3_quality.log 2>&1; echo "[$(date +%T)] quality exit=$?"
