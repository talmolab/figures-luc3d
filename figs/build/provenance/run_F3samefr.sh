#!/usr/bin/env bash
# Fig 3F on identical frames (2026-10-15). Waits for the Fig 2G sitting (Q2d) so the two timings don't overlap.
set -uo pipefail
cd /root/vast/eric/sleap-3d-gui/scratch/repos/lucid
L=figs/out/md56_logs
until grep -qE "Q2d DONE|exit=[1-9]" $L/run_Q2d.out 2>/dev/null; do sleep 20; done
export PATH=/root/vast/eric/node26_env/bin:$PATH LUCID_ROOT=/root/vast/eric/lucid-main-md56 FIG3_VARIANT=md56 SCORE_NOTAIL=1 HH_JOBS=16
echo "[$(date +%T)] start (load $(cut -d' ' -f1 /proc/loadavg))"
/root/vast/eric/luc3d-bench/liezl_env/bin/python figs/fig3_samefr_runtime.py > $L/F3samefr.log 2>&1
echo "[$(date +%T)] samefr exit=$? (load $(cut -d' ' -f1 /proc/loadavg))"
echo "F3SAMEFR DONE"
