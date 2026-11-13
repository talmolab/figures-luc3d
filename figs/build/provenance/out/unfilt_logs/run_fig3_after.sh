#!/usr/bin/env bash
# Fig 3b, part 2: fresh-anchor greedy arm on the unfiltered pool, then the 3f timing
# benchmark on a quiet machine. Tagged outputs only (FIG3_VARIANT=unfilt).
set -uo pipefail
cd /root/vast/eric/sleap-3d-gui/scratch/repos/lucid
export PATH=/root/vast/eric/node26_env/bin:$PATH
export FIG3_VARIANT=unfilt
export PYTHONPATH=/tmp/claude-0/-root-vast-eric-sleap-3d-gui-scratch-repos-lucid/44adbb8d-7377-41ea-ab2c-5dae7a320d17/scratchpad/cv2h
PY=/root/vast/eric/luc3d-bench/liezl_env/bin/python
L=figs/out/unfilt_logs
while pgrep -f "unfilt_logs/run_fig3.sh" >/dev/null; do sleep 60; done
grep -q "quality exit=0" $L/run_fig3.out || { echo "[after] head-to-head/quality did not finish cleanly; stopping"; exit 2; }
echo "[$(date +%T)] freshanchor start"
$PY figs/fig3_hh_freshanchor.py --run --jobs 32 > $L/fig3_freshanchor.log 2>&1; echo "[$(date +%T)] freshanchor exit=$?"
# timing needs a quiet box: wait for the Fig 6 chain and any tracker/sleap jobs
while pgrep -f "unfilt_logs/run_fig6.sh|sleap_nn.cli|fig6_slap2m_fair.py" >/dev/null; do sleep 120; done
echo "[$(date +%T)] scale runtime start (load $(cut -d' ' -f1 /proc/loadavg))"
$PY figs/fig3_scale_runtime.py > $L/fig3_scale.log 2>&1; echo "[$(date +%T)] scale exit=$?"
