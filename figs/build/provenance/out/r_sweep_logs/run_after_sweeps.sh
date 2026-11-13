#!/bin/bash
# 2026-10-18: render the SLAP-2M staleness figure when stale_sweep_slap2m.py finishes,
# and log both sweeps' results to RUN-STATUS. Detached so it survives Eric disconnecting.
cd /root/vast/eric/sleap-3d-gui/scratch/repos/lucid/figs
ST=figures/drafts/RUN-STATUS-2026-10-15.md
log() { echo "- $(date -u '+%Y-%m-%d %H:%M:%S') UTC eric-mimic2 [main box] $*" >> $ST; }
while pgrep -f stale_sweep_slap2m.py >/dev/null; do sleep 60; done
if [ -f out/stale_sweep_slap2m.json ]; then
  cp figures/fig6/fig6f_axes.json /tmp/fig6f_axes.keep.json
  FIG_TAG=slapstale FIG6F_NOEVICT=stale0_dist25 .venv/bin/python panels/fig6_07_pr_switches.py \
    --deposit stale_sweep_slap2m.json > out/r_sweep_logs/slapstale_render.out 2>&1
  rc=$?
  cp /tmp/fig6f_axes.keep.json figures/fig6/fig6f_axes.json   # panel writes this; keep main copy
  log "STALESWEEP DONE (render exit $rc -> figures/fig6slapstale/): $(grep '^\[stale_sweep\] s' out/r_sweep_logs/stale_sweep_slap2m.out | tr '\n' ';')"
else
  log "STALESWEEP FAILED: $(tail -2 out/r_sweep_logs/stale_sweep_slap2m.out | tr '\n' ' ')"
fi
while pgrep -f 'r_sweep_slap2m.py --workers 24 --stale 20' >/dev/null; do sleep 60; done
log "RSWEEP stale20 finished: $(grep -E 'r[0-9.]+ +all' out/r_sweep_logs/r_sweep_slap2m_stale20.out | tr -s ' ' | tr '\n' ';')"
