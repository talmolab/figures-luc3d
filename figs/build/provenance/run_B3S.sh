#!/usr/bin/env bash
# B3S (2026-10-15, Eric): Fig 3G/H sweep lines on Mouse-Dyad config b3
# (det_h5_bvis5 body>=5 pool, tail IN tracking, tail-free scoring). Writes only *_b3 deposits
# and tmp/sweep50_2dc001fe_b3/. Same grid/thresholds/env as run_S2.sh (md56main).
set -uo pipefail
cd /root/vast/eric/sleap-3d-gui/scratch/repos/lucid
PY=/root/vast/eric/envs/md56_score_env/bin/python
W=${WORKERS:-40}
export PATH=/root/vast/eric/node26_env/bin:$PATH SCORE_NOTAIL=1 MD_CFG=b3 SWEEP50_SUFFIX=b3 \
       FIG8M_SUFFIX=b3 SWEEP50_LUCID_ROOT=/root/vast/eric/lucid-main-md56
L=figs/out/md56_logs; G="--grid-corr3d 0,0.25,0.5,1,2,3,4,6,8,12,16,24 --grid-corr2d 1"
ST=figs/figures/drafts/RUN-STATUS-2026-10-15.md
t() { echo "[$(date -u +%T) $(hostname -s)] $* (load $(cut -d' ' -f1 /proc/loadavg))"; }
st() { echo "- $(date -u '+%F %T') UTC $(hostname -s) [main box] $*" >> $ST; t "$*"; }
st "B3S start: Fig 3G/H sweeps on b3 (det_h5_bvis5, tail in tracking), main $(git -C $SWEEP50_LUCID_ROOT rev-parse --short HEAD), node $(node --version), workers $W"
$PY figs/fig3_sweep50.py $G --thresholds '{"distanceThreshold":25,"stale":20}' --workers $(( W / 2 )) > $L/B3S_sweep_current.log 2>&1 & P1=$!
$PY figs/fig3_sweep50.py $G --thresholds '{"distanceThreshold":50,"stale":0}'  --workers $(( W - W / 2 )) > $L/B3S_sweep_prevsettings.log 2>&1 & P2=$!
wait $P1; E1=$?; t "current exit=$E1"
wait $P2; E2=$?; t "prev-settings exit=$E2"
ls -la figs/out/fig3_sweep50*_b3.json
if [ $E1 -eq 0 ] && [ $E2 -eq 0 ]; then st "B3S DONE: figs/out/fig3_sweep50__*_b3.json"; else st "B3S FAILED (exit $E1/$E2)"; fi
