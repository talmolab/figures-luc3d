#!/usr/bin/env bash
# md56 phase S (2026-10-14): the two Fig 3G/H sweep arms on all 56 Mouse-Dyad-10M sessions.
# OFFLOADED to a second machine (see figs/figures/drafts/HANDOFF-MD56-SWEEPS.md). Accuracy
# only, nothing timed. Shares the VAST mount with the main box; writes only
# figs/out/tmp/sweep50/<tag>/ caches and figs/out/fig3_sweep50*_md56.json.
#   PY=<python 3.12 that sees liezl_env's site-packages>  WORKERS=<n>  run_S.sh
set -uo pipefail
cd /root/vast/eric/sleap-3d-gui/scratch/repos/lucid
: "${PY:?set PY to the python built in the handoff}"
W=${WORKERS:-$(( $(nproc) - 2 ))}
export PATH=/root/vast/eric/node26_env/bin:$PATH SCORE_NOTAIL=1 SWEEP50_SUFFIX=md56
L=figs/out/md56_logs; G="--grid-corr3d 0,0.25,0.5,1,2,3,4,6,8,12,16,24 --grid-corr2d 1"
t() { echo "[$(date +%T) $(hostname -s)] $* (load $(cut -d' ' -f1 /proc/loadavg))"; }
until grep -q "U4 inputs exit=0" $L/run_phaseU.out 2>/dev/null; do sleep 60; done
n=$(ls /root/vast/eric/luc3d-bench/outputs/bmimica/det_h5 | wc -l)
t "start, det_h5 sessions $n, node $(node --version), workers $W"
[ "$n" -eq 56 ] || { t "ABORT: det_h5 has $n sessions, expected 56"; exit 1; }
$PY figs/fig3_sweep50.py $G --method '{"sync":true,"stale":20}' --thresholds '{"distanceThreshold":25}' \
  --workers $(( W / 2 )) > $L/S_sweep_fresh.log 2>&1 &
P1=$!
$PY figs/fig3_sweep50.py $G --workers $(( W - W / 2 )) > $L/S_sweep_shipped.log 2>&1 &
P2=$!
wait $P1; t "S1 sweep fresh exit=$?"
wait $P2; t "S2 sweep shipped exit=$?"
ls -la figs/out/fig3_sweep50*md56*.json
t "PHASE S DONE"
