#!/usr/bin/env bash
# Fig 3b on body>=5, tail-free GT matching. Tagged outputs only.
set -uo pipefail
cd /root/vast/eric/sleap-3d-gui/scratch/repos/lucid
export PATH=/root/vast/eric/node26_env/bin:$PATH SCORE_NOTAIL=1 FIG3_VARIANT=bvis5 HH_JOBS=16
export PYTHONPATH=/tmp/claude-0/-root-vast-eric-sleap-3d-gui-scratch-repos-lucid/44adbb8d-7377-41ea-ab2c-5dae7a320d17/scratchpad/cv2h
PY=/root/vast/eric/luc3d-bench/liezl_env/bin/python; L=figs/out/bvis5_logs; T=distanceThreshold25-stale20-sync_e508a7ab
# Mouse-Dyad-10M has no filter: reuse its tracker/exhaustive caches, but NOT quality.json
# (that holds tail-counted GT matching, which must be recomputed tail-free)
mkdir -p figs/out/tmp/headtohead_bvis5 figs/out/tmp/headtohead_var_bvis5/$T
cp -a figs/out/tmp/headtohead/A2_C5_bmimica figs/out/tmp/headtohead_bvis5/
cp -a figs/out/tmp/headtohead_var/$T/A2_C5_bmimica figs/out/tmp/headtohead_var_bvis5/$T/
find figs/out/tmp/headtohead_bvis5/A2_C5_bmimica figs/out/tmp/headtohead_var_bvis5/$T/A2_C5_bmimica -name quality.json -delete
echo "[$(date +%T)] headtohead start"
$PY figs/fig3_headtohead.py > $L/fig3_headtohead.log 2>&1; echo "[$(date +%T)] headtohead exit=$? failed=$(grep -c FAILED $L/fig3_headtohead.log)"
Q_FORCE= $PY figs/fig3_quality.py > $L/fig3_quality.log 2>&1; echo "[$(date +%T)] quality exit=$?"
$PY figs/fig3_hh_freshanchor.py --run --jobs 32 > $L/fig3_freshanchor.log 2>&1; r=$?
ls figs/out/fig3_quality__${T}_bvis5.json >/dev/null 2>&1 && echo "[$(date +%T)] freshanchor ok" || echo "[$(date +%T)] freshanchor FAILED exit=$r"
