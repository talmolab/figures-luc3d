#!/usr/bin/env bash
# Fig 6b / 4b on body>=5, tail-free scoring. Tagged outputs only.
set -uo pipefail
cd /root/vast/eric/sleap-3d-gui/scratch/repos/lucid
export PATH=/root/vast/eric/node22_env/bin:$PATH SCORE_NOTAIL=1 FAIR_VARIANT=bvis5 FAIR_SLEAP_PY=/root/vast/eric/sleap-3d-gui/scratch/repos/lucid/figs/out/tmp/snenv_unfilt/bin/python
export PYTHONPATH=/tmp/claude-0/-root-vast-eric-sleap-3d-gui-scratch-repos-lucid/44adbb8d-7377-41ea-ab2c-5dae7a320d17/scratchpad/cv2h
B=/root/vast/eric/luc3d-bench; L=figs/out/bvis5_logs; SN=figs/out/tmp/snenv_unfilt/bin/python
while pgrep -f "bvis4_logs/run_fig6.sh" >/dev/null; do sleep 60; done
while pgrep -f "bvis5_logs/run_light.sh" >/dev/null; do sleep 60; done
SCORE_NOTAIL=1 $B/liezl_env/bin/python figs/fig4_slap2m.py --pool lite_bvis5 --configs sync_stale20_dist25 --workers 10 > $L/luc3d74_retry.log 2>&1; echo "[$(date +%T)] luc3d74 retry failed=$(grep -c FAILED $L/luc3d74_retry.log)"
# 2-3. SLEAP on body>=5-filtered copies of the raw predictions, no filter flags
$B/eks_env/bin/python figs/fig6_slap2m_fair.py --stage sleap --workers 28 > $L/sleap.log 2>&1
$B/eks_env/bin/python figs/fig6_slap2m_fair.py --stage sleap --workers 28 >> $L/sleap.log 2>&1
n=$(find figs/out/tmp/slap2m_fair_bvis5/sleap_cap -name '*.slp' ! -name '*.partial.slp' | wc -l); echo "[$(date +%T)] sleap $n/444"
$SN figs/fig6_slap2m_fair.py --stage verify > $L/verify.log 2>&1; v=$?; tail -1 $L/verify.log
if [ $v -ne 0 ] || [ "$n" -ne 444 ]; then echo "[chain] REFUSING (verify=$v n=$n)"; exit 2; fi
$B/eks_env/bin/python figs/fig6_slap2m_fair.py --stage bytetrack --workers 28 > $L/byte.log 2>&1; echo "[$(date +%T)] bytetrack exit=$?"
$B/liezl_env/bin/python figs/fig6_slap2m_fair.py --stage score --workers 32 > $L/score.log 2>&1; echo "[$(date +%T)] score exit=$?"; tail -3 $L/score.log
$B/liezl_env/bin/python figs/fig6_slap2m_fair.py --stage install > $L/install.log 2>&1; echo "[$(date +%T)] install exit=$?"; tail -8 $L/install.log
