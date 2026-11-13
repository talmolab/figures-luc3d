#!/usr/bin/env bash
# Fig 6b: SLEAP + ByteTrack on the UNFILTERED pool, scored with the gate-free LUC3D runs.
set -uo pipefail
cd /root/vast/eric/sleap-3d-gui/scratch/repos/lucid
export FAIR_VARIANT=unfilt FAIR_SLEAP_PY=/root/vast/eric/sleap-3d-gui/scratch/repos/lucid/figs/out/tmp/snenv_unfilt/bin/python
export PYTHONPATH=/tmp/claude-0/-root-vast-eric-sleap-3d-gui-scratch-repos-lucid/44adbb8d-7377-41ea-ab2c-5dae7a320d17/scratchpad/cv2h
B=/root/vast/eric/luc3d-bench; L=figs/out/unfilt_logs
echo "[$(date +%T)] sleap start"
$B/eks_env/bin/python figs/fig6_slap2m_fair.py --stage sleap --workers 28 > $L/fair_sleap.log 2>&1
$B/eks_env/bin/python figs/fig6_slap2m_fair.py --stage sleap --workers 28 >> $L/fair_sleap.log 2>&1   # retry pass
n=$(find figs/out/tmp/slap2m_fair_unfilt/sleap_cap -name '*.slp' ! -name '*.partial.slp' | wc -l); echo "[$(date +%T)] sleap outputs $n / 444"
$FAIR_SLEAP_PY figs/fig6_slap2m_fair.py --stage verify > $L/fair_verify.log 2>&1; v=$?; tail -3 $L/fair_verify.log
if [ $v -ne 0 ] || [ "$n" -ne 444 ]; then echo "[chain] REFUSING to continue (verify=$v, n=$n)"; exit 2; fi
$B/eks_env/bin/python figs/fig6_slap2m_fair.py --stage bytetrack --workers 28 > $L/fair_byte.log 2>&1; echo "[$(date +%T)] bytetrack exit=$?"
$B/liezl_env/bin/python figs/fig6_slap2m_fair.py --stage score --workers 32 > $L/fair_score.log 2>&1; echo "[$(date +%T)] score exit=$?"
$B/liezl_env/bin/python figs/fig6_slap2m_fair.py --stage install > $L/fair_install.log 2>&1; echo "[$(date +%T)] install exit=$?"
tail -12 $L/fair_install.log
