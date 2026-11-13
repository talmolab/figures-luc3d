#!/usr/bin/env bash
# Fig 3c re-time on a quiet machine: SLAP-2M head-to-head cells from scratch (same sampled
# frames -- deterministic -- so 3e's accuracy numbers must not move), then quality,
# fresh-anchor, the greedy (LUC3D) timing run, and the 3c rebuild. Old cells kept.
set -uo pipefail
cd /root/vast/eric/sleap-3d-gui/scratch/repos/lucid
export PATH=/root/vast/eric/node26_env/bin:$PATH SCORE_NOTAIL=1 FIG3_VARIANT=bvis5 HH_JOBS=16 PYTHONPATH=/tmp/claude-0/-root-vast-eric-sleap-3d-gui-scratch-repos-lucid/44adbb8d-7377-41ea-ab2c-5dae7a320d17/scratchpad/cv2h
PY=/root/vast/eric/luc3d-bench/liezl_env/bin/python; L=figs/out/bvis5_logs; H=figs/out/tmp/headtohead_bvis5
mkdir -p figs/out/tmp/headtohead_bvis5_loaded
for c in A2_C6_slap2m A3_C5_slap2m A4_C3_slap2m; do mv $H/$c figs/out/tmp/headtohead_bvis5_loaded/$c; done
cp figs/out/fig3_headtohead_bvis5.json figs/out/fig3_headtohead_bvis5.loaded.json
cp figs/out/fig3_quality_bvis5.json figs/out/fig3_quality_bvis5.loaded.json
echo "[$(date +%T)] headtohead start (load $(cut -d' ' -f1 /proc/loadavg))"
$PY figs/fig3_headtohead.py > $L/retime_headtohead.log 2>&1; echo "[$(date +%T)] headtohead exit=$? failed=$(grep -c FAILED $L/retime_headtohead.log)"
$PY figs/fig3_quality.py > $L/retime_quality.log 2>&1; echo "[$(date +%T)] quality exit=$?"
find figs/out/tmp/headtohead_var_bvis5 -path "*slap2m*" -name quality.json -delete
$PY figs/fig3_hh_freshanchor.py --run --jobs 16 > $L/retime_freshanchor.log 2>&1; echo "[$(date +%T)] freshanchor exit=$?"
$PY figs/fig3_scale_runtime.py > $L/retime_scale.log 2>&1; echo "[$(date +%T)] greedy timing exit=$?"
FIG_TAG=c FIG_VARIANT=bvis5 figs/.venv/bin/python figs/make_figures.py 3 > $L/retime_make3c.log 2>&1; echo "[$(date +%T)] make 3c exit=$?"
grep -h assembled $L/retime_make3c.log; echo "[$(date +%T)] RETIME DONE"
