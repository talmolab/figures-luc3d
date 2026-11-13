#!/usr/bin/env bash
# Fig 6b / 4b on body>=4, tail-free scoring. Tagged outputs only.
set -uo pipefail
cd /root/vast/eric/sleap-3d-gui/scratch/repos/lucid
export PATH=/root/vast/eric/node22_env/bin:$PATH SCORE_NOTAIL=1 FAIR_VARIANT=bvis4 FAIR_SLEAP_PY=/root/vast/eric/sleap-3d-gui/scratch/repos/lucid/figs/out/tmp/snenv_unfilt/bin/python
export PYTHONPATH=/tmp/claude-0/-root-vast-eric-sleap-3d-gui-scratch-repos-lucid/44adbb8d-7377-41ea-ab2c-5dae7a320d17/scratchpad/cv2h
B=/root/vast/eric/luc3d-bench; L=figs/out/bvis4_logs; SN=figs/out/tmp/snenv_unfilt/bin/python
# 2-3. SLEAP on body>=4-filtered copies of the raw predictions, no filter flags
$B/eks_env/bin/python figs/fig6_slap2m_fair.py --stage sleap --workers 28 > $L/sleap.log 2>&1
$B/eks_env/bin/python figs/fig6_slap2m_fair.py --stage sleap --workers 28 >> $L/sleap.log 2>&1
n=$(find figs/out/tmp/slap2m_fair_bvis4/sleap_cap -name '*.slp' ! -name '*.partial.slp' | wc -l); echo "[$(date +%T)] sleap $n/444"
$SN figs/fig6_slap2m_fair.py --stage verify > $L/verify.log 2>&1; v=$?; tail -1 $L/verify.log
if [ $v -ne 0 ] || [ "$n" -ne 444 ]; then echo "[chain] REFUSING (verify=$v n=$n)"; exit 2; fi
$B/eks_env/bin/python figs/fig6_slap2m_fair.py --stage bytetrack --workers 28 > $L/byte.log 2>&1; echo "[$(date +%T)] bytetrack exit=$?"
$B/liezl_env/bin/python figs/fig6_slap2m_fair.py --stage score --workers 32 > $L/score.log 2>&1; echo "[$(date +%T)] score exit=$?"; tail -3 $L/score.log
$B/liezl_env/bin/python figs/fig6_slap2m_fair.py --stage install > $L/install.log 2>&1; echo "[$(date +%T)] install exit=$?"; tail -8 $L/install.log
