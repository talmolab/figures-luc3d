#!/usr/bin/env bash
# Waits for the C6 SLEAP b3 track+verify, then scores it tail-free (camera-scoped), 56 sessions.
set -uo pipefail
cd /root/vast/eric/sleap-3d-gui/scratch/repos/lucid
ST=figs/figures/drafts/RUN-STATUS-2026-10-15.md; L=figs/out/md56_logs
PY=/root/vast/eric/luc3d-bench/liezl_env/bin/python
until grep -qE "C6SLEAP DONE|FAILED" $L/run_C6_sleap_b3.out 2>/dev/null; do sleep 120; done
grep -q "C6SLEAP DONE" $L/run_C6_sleap_b3.out || exit 1
SCORE_NOTAIL=1 $PY figs/fig6_sleap_scoped.py --retracked /root/vast/eric/luc3d-bench/outputs/bmimica/retracked_max2_b3 \
  --out fig7_sleap_max2_c6b3.json --workers 20 > $L/C6_sleap_score.log 2>&1
echo "- $(date -u +%H:%M) | ericmimicc | C6 SLEAP b3 | score exit=$? -> figs/out/fig7_sleap_max2_c6b3.json ($(tail -1 $L/C6_sleap_score.log))" >> $ST
