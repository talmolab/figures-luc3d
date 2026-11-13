#!/usr/bin/env bash
# md56 phase S2 (2026-10-14): the Fig 3G/H sweep REDONE on origin/main (talmolab/luc3d) for
# all 56 sessions, nothing reused. Replaces run_S.sh, whose tracker outputs came from the
# eric/figs branch's pose/ + xv_experimental.js. Tracks with the main checkout's real pose/
# through figs/fig3-bench/fig3_bench.mjs (SWEEP50_LUCID_ROOT), cache tmp/sweep50_<commit>/,
# deposits *_md56main.json. Prints "PHASE S2 DONE" only if both arms exit 0.
#   PY=<md56_sweep_env python>  WORKERS=<n>  run_S2.sh
set -uo pipefail
cd /root/vast/eric/sleap-3d-gui/scratch/repos/lucid
: "${PY:?set PY}"
W=${WORKERS:-$(( $(nproc) - 2 ))}
export PATH=/root/vast/eric/node26_env/bin:$PATH SCORE_NOTAIL=1 SWEEP50_SUFFIX=md56main \
       FIG8M_SUFFIX=md56 SWEEP50_LUCID_ROOT=/root/vast/eric/lucid-main-md56
L=figs/out/md56_logs; G="--grid-corr3d 0,0.25,0.5,1,2,3,4,6,8,12,16,24 --grid-corr2d 1"
t() { echo "[$(date +%T) $(hostname -s)] $* (load $(cut -d' ' -f1 /proc/loadavg))"; }
t "start, main $(git -C $SWEEP50_LUCID_ROOT rev-parse --short HEAD), node $(node --version), workers $W"
# Current tracker = main's defaults (distanceThreshold 25, stale 20), stated explicitly.
$PY figs/fig3_sweep50.py $G --thresholds '{"distanceThreshold":25,"stale":20}' \
  --workers $(( W / 2 )) > $L/S2_sweep_current.log 2>&1 &
P1=$!
# Previous settings on main: distanceThreshold 50, no stale eviction. Main's association is
# always frame-synchronous, so this is NOT the pre-#210 tracker, only its settings.
$PY figs/fig3_sweep50.py $G --thresholds '{"distanceThreshold":50,"stale":0}' \
  --workers $(( W - W / 2 )) > $L/S2_sweep_prevsettings.log 2>&1 &
P2=$!
wait $P1; E1=$?; t "S2 current exit=$E1"
wait $P2; E2=$?; t "S2 prev-settings exit=$E2"
ls -la figs/out/fig3_sweep50*md56main*.json
if [ $E1 -eq 0 ] && [ $E2 -eq 0 ]; then t "PHASE S2 DONE"; else t "PHASE S2 FAILED"; fi
