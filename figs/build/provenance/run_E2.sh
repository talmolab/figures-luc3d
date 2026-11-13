#!/usr/bin/env bash
# md56 phase E2 (2026-10-14): the rest of Fig 3 on origin/main 2dc001fe, unattended.
#   1. Fig 3E greedy arm + quality rescore on main (fig3_hh_freshanchor.py, FRESHANCHOR_LUCID_ROOT;
#      fig3_rescore_frames.mjs now honours LUCID_ROOT). Deposits -> figs/out/md56main/.
#   2. Gate + diff of (1) against what fig3b drew (compare_e2.py).
#   3. Waits for the Fig 3G/H sweep (run_S2.sh, "PHASE S2 DONE"), then compare_s2.py.
# Everything lands in md56_logs/REPORT_md56main.txt. Writes no *_md56 / base deposit.
set -uo pipefail
cd /root/vast/eric/sleap-3d-gui/scratch/repos/lucid
L=figs/out/md56_logs; R=$L/REPORT_md56main.txt
PY=$HOME/md56_sweep_env/bin/python
export PATH=/root/vast/eric/node26_env/bin:$PATH FIG3_VARIANT=md56 SCORE_NOTAIL=1 \
       FRESHANCHOR_LUCID_ROOT=/root/vast/eric/lucid-main-md56
t() { echo "[$(date -u +%F' '%T) UTC $(hostname -s)] $* (load $(cut -d' ' -f1 /proc/loadavg))"; }
{
t "E2 start: Fig 3E on main $(git -C $FRESHANCHOR_LUCID_ROOT rev-parse --short HEAD)"
$PY figs/fig3_hh_freshanchor.py --run --jobs 8 > $L/E2_freshanchor.log 2>&1
E=$?; t "E2 freshanchor exit=$E (log $L/E2_freshanchor.log)"
grep -E "FAILED|Error|refusing" $L/E2_freshanchor.log | head -20
ls -la figs/out/md56main/ 2>&1
echo; echo "=== Fig 3E: main vs what fig3b drew ==="
$PY $L/compare_e2.py 2>&1
echo
t "waiting for the Fig 3G/H sweep (run_S2.out)"
until grep -qE "PHASE S2 (DONE|FAILED)" $L/run_S2.out 2>/dev/null; do sleep 120; done
cat $L/run_S2.out
grep -hE "SCORE-FAILED|FAILED c2" $L/S2_sweep_*.log | head
echo; echo "=== Fig 3G/H: main vs eric/figs ==="
$PY $L/compare_s2.py 2>&1
t "E2 DONE"
} > $R 2>&1
