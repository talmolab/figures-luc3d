#!/usr/bin/env bash
# md56 F4 (2026-10-14): Fig 4D (LUC3D cross-view IDF1 on SLAP-2M, by difficulty) re-tracked from
# scratch, exactly as the md56 Fig 6 LUC3D arm: fig6_bench.mjs driver, config
# sync_stale20_dist25 ({"sync":true,"stale":20} + distanceThreshold 25), lite_bvis5 pool,
# tail-free scoring. --tag md56 = fresh cache out/tmp/fig9slap_lite_bvis5_md56 (no reuse).
set -uo pipefail
cd /root/vast/eric/sleap-3d-gui/scratch/repos/lucid
L=figs/out/md56_logs
export PATH=/root/vast/eric/node26_env/bin:$PATH SCORE_NOTAIL=1
export PYTHONPATH=/tmp/claude-0/-root-vast-eric-sleap-3d-gui-scratch-repos-lucid/44adbb8d-7377-41ea-ab2c-5dae7a320d17/scratchpad/cv2h
PY=/root/vast/eric/luc3d-bench/liezl_env/bin/python
t() { echo "[$(date +%T)] $* (load $(cut -d' ' -f1 /proc/loadavg))"; }
t start
$PY figs/fig4_slap2m.py --pool lite_bvis5 --configs sync_stale20_dist25 --tag md56 --workers 24 > $L/F4_slap2m.log 2>&1
t "fig4_slap2m exit=$?"
mv figs/out/fig9_slap2m_lite_bvis5_md56_notailscore.json figs/out/fig9_slap2m_lite_bvis5_notailscore_md56.json && t "renamed -> fig9_slap2m_lite_bvis5_notailscore_md56.json"
t "F4 DONE"
