#!/usr/bin/env bash
# Fig 3G/H sweep frame-matched rescoring (2026-10-15). Waits for both timing runs (Q2d, F3 same-frames).
set -uo pipefail
cd /root/vast/eric/sleap-3d-gui/scratch/repos/lucid
L=figs/out/md56_logs
until grep -q "F3SAMEFR DONE" $L/run_F3samefr.out 2>/dev/null; do sleep 30; done
export FIG3_VARIANT=md56 SCORE_NOTAIL=1 SFM_JOBS=40
echo "[$(date +%T)] start (load $(cut -d' ' -f1 /proc/loadavg))"
/root/vast/eric/luc3d-bench/liezl_env/bin/python figs/fig3_sweep_framematched.py > $L/G3sfm.log 2>&1
echo "[$(date +%T)] sfm exit=$?"
echo "G3SFM DONE"
