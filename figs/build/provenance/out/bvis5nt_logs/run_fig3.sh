#!/usr/bin/env bash
# bvis5nt (2026-10-12): Fig 3 head-to-head with the EXHAUSTIVE arm tail-free, so both arms
# ignore the four tail nodes. Greedy arms are unchanged (always tail weight 0) and are
# copied from the bvis5 caches; only exhaustive.json and quality.json are recomputed.
# Tagged outputs only; nothing bvis5 or base is touched. Run on a quiet machine (Fig 3F
# times come from these exhaustive runs).
set -uo pipefail
cd /root/vast/eric/sleap-3d-gui/scratch/repos/lucid
export PATH=/root/vast/eric/node26_env/bin:$PATH SCORE_NOTAIL=1 FIG3_VARIANT=bvis5nt HH_JOBS=16
export PYTHONPATH=/tmp/claude-0/-root-vast-eric-sleap-3d-gui-scratch-repos-lucid/44adbb8d-7377-41ea-ab2c-5dae7a320d17/scratchpad/cv2h
PY=/root/vast/eric/luc3d-bench/liezl_env/bin/python; L=figs/out/bvis5nt_logs; T=distanceThreshold25-stale20-sync_e508a7ab
O=figs/out/tmp
# seed greedy arms (production head-to-head and fresh-anchor) from bvis5
for src in headtohead_bvis5 headtohead_var_bvis5/$T; do
  dst=${src/bvis5/bvis5nt}
  ( cd $O/$src && find . -name greedy.json ) | while read f; do
    mkdir -p "$O/$dst/$(dirname "$f")"; cp -n "$O/$src/$f" "$O/$dst/$f"; done
done
echo "[$(date +%T)] seeded $(find $O/headtohead_bvis5nt -name greedy.json | wc -l) + $(find $O/headtohead_var_bvis5nt -name greedy.json | wc -l) greedy cells (load $(cut -d' ' -f1 /proc/loadavg))"
$PY figs/fig3_headtohead.py > $L/fig3_headtohead.log 2>&1; echo "[$(date +%T)] headtohead exit=$? failed=$(grep -c FAILED $L/fig3_headtohead.log)"
echo "[$(date +%T)] exhaustive cells tail-free: $(grep -l '"excludeTail": true' -r $O/headtohead_bvis5nt --include=exhaustive.json | wc -l)"
$PY figs/fig3_quality.py > $L/fig3_quality.log 2>&1; echo "[$(date +%T)] quality exit=$?"
$PY figs/fig3_hh_freshanchor.py --run --jobs 16 > $L/fig3_freshanchor.log 2>&1; echo "[$(date +%T)] freshanchor exit=$?"
ls -la figs/out/fig3_headtohead_bvis5nt.json figs/out/fig3_quality_bvis5nt.json figs/out/fig3_quality__${T}_bvis5nt.json 2>&1
echo "[$(date +%T)] FIG3 DONE"
