#!/usr/bin/env bash
# bvis4: build Fig 4b + 6b as soon as the Fig 6 deposit is installed; Fig 3b after a quiet
# 3f timing run. Refuses to build anything whose deposits are missing.
set -uo pipefail
cd /root/vast/eric/sleap-3d-gui/scratch/repos/lucid
L=figs/out/bvis4_logs; P=figs/.venv/bin/python
need() { for f in "$@"; do [ -s "$f" ] || { echo "[final] REFUSING: missing $f"; return 1; }; done; }
while pgrep -f "bvis4_logs/run_fig6.sh" >/dev/null || [ ! -e $L/run_fig6.out ]; do sleep 60; done
cat $L/run_fig6.out
if need figs/out/fig7_variant_best_bvis4.json figs/out/fig9_slap2m_lite_bvis4_notailscore.json; then
  for d in fig4b fig6b; do [ -e figs/figures/${d}_unfilt ] || [ ! -e figs/figures/$d ] || cp -a figs/figures/$d figs/figures/${d}_unfilt; done
  FIG_TAG=b FIG_VARIANT=bvis4 $P figs/make_figures.py 6 > $L/make_6b.log 2>&1; echo "[$(date +%T)] make 6b exit=$?"
  ( export FIG_TAG=b FIG_VARIANT=proofread_notail
    for p in fig4_13_enrichment_grid fig4_14_occlusion fig4_06_detection_quality fig4_11_idf1_by_difficulty; do $P figs/panels/$p.py; done
    $P figs/assemble.py 4 ) > $L/make_4b.log 2>&1; echo "[$(date +%T)] make 4b exit=$?"
  grep -h "assembled" $L/make_6b.log $L/make_4b.log
  echo "[$(date +%T)] 4b 6b DONE"
fi
# 3b: needs its head-to-head chain, then a quiet machine for the 3f timing
while pgrep -f "bvis4_logs/run_fig3.sh" >/dev/null || [ ! -e $L/run_fig3.out ]; do sleep 60; done
while pgrep -f "sleap_nn.cli|fig6_slap2m_fair.py|fig3_headtohead.py|fig3_quality.py|fig3_hh_freshanchor.py|fig4_slap2m.py|fig3_scale_runtime.py" >/dev/null; do sleep 60; done
export PATH=/root/vast/eric/node26_env/bin:$PATH PYTHONPATH=/tmp/claude-0/-root-vast-eric-sleap-3d-gui-scratch-repos-lucid/44adbb8d-7377-41ea-ab2c-5dae7a320d17/scratchpad/cv2h
echo "[$(date +%T)] scale runtime start (load $(cut -d' ' -f1 /proc/loadavg))"
FIG3_VARIANT=bvis4 /root/vast/eric/luc3d-bench/liezl_env/bin/python figs/fig3_scale_runtime.py > $L/fig3_scale.log 2>&1; echo "[$(date +%T)] scale exit=$?"
if need figs/out/fig3_headtohead_bvis4.json figs/out/fig3_quality_bvis4.json figs/out/fig3_runtime_bvis4.json figs/out/fig3_quality__distanceThreshold25-stale20-sync_e508a7ab_bvis4.json; then
  [ -e figs/figures/fig3b_unfilt ] || [ ! -e figs/figures/fig3b ] || cp -a figs/figures/fig3b figs/figures/fig3b_unfilt
  FIG_TAG=b FIG_VARIANT=bvis4 $P figs/make_figures.py 3 > $L/make_3b.log 2>&1; echo "[$(date +%T)] make 3b exit=$?"
  grep -h "assembled" $L/make_3b.log
fi
echo "[$(date +%T)] FINAL DONE"
