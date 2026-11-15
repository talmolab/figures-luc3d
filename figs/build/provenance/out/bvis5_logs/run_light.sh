#!/usr/bin/env bash
# Body>=5 (Figs 3c/4c/6c), light part now: LUC3D 74 sessions + SLEAP's filtered inputs.
set -uo pipefail
cd /root/vast/eric/sleap-3d-gui/scratch/repos/lucid
export PATH=/root/vast/eric/node22_env/bin:$PATH SCORE_NOTAIL=1 FAIR_VARIANT=bvis5 PYTHONPATH=/tmp/claude-0/-root-vast-eric-sleap-3d-gui-scratch-repos-lucid/44adbb8d-7377-41ea-ab2c-5dae7a320d17/scratchpad/cv2h
B=/root/vast/eric/luc3d-bench; L=figs/out/bvis5_logs
mkdir -p figs/out/tmp/fig9slap_lite_bvis5/sync_stale20_dist25
cp --update=none figs/out/tmp/fig9slap_lite_bvis5_lite/sync_stale20_dist25/*.json figs/out/tmp/fig9slap_lite_bvis5/sync_stale20_dist25/
$B/liezl_env/bin/python figs/fig4_slap2m.py --pool lite_bvis5 --configs sync_stale20_dist25 --workers 10 > $L/luc3d74.log 2>&1
echo "[$(date +%T)] luc3d74 exit=$? failed=$(grep -c FAILED $L/luc3d74.log)"
figs/out/tmp/snenv_unfilt/bin/python figs/fig6_slap2m_fair.py --stage sleapsrc --workers 8 > $L/sleapsrc.log 2>&1; tail -1 $L/sleapsrc.log
