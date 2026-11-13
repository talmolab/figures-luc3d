#!/usr/bin/env bash
# F5FIX (2026-10-15 audit): Fig 5 fixes. Additive; writes only new tagged names.
#   1. fig5_controls.py with FIG5_MIN_DISPLAYS=6 (the text's >=6 / 23 sessions; the script
#      default 5 gives 24). Deposit -> data/fig5f5fix/fig5s_controls.csv (FIG_TAG=f5fix),
#      NOT the published data/fig5/fig5s_controls.csv.
#   2. fig5_rear_coupling.py --per-direction --slap-animals 2 --skip-slap
#      -> out/fig5_rear_coupling_2animal_perdir.json (far + circular-shift null per
#      direction, on the same sessions as each direction's near curve).
#   3. Fig 5G panel with FIG5G_PERDIR=1 FIG_TAG=f5fix -> figures/fig5f5fix/fig5f5fixg_rear_coupling.{pdf,png}
#      + data/fig5f5fix/fig5g_rear_coupling_perdir.csv. Published fig5 untouched.
#   4. compare_f5fix.py: old vs new for every Fig 5 control / Fig 5G number in the tex.
# Pure NumPy/pandas on the BMimica proofread points3d files -- no tracker involved.
# Expected runtime ~1-2 min (14 + 12 worker processes, ~1 s and ~0.5 s per session).
set -uo pipefail
cd /root/vast/eric/sleap-3d-gui/scratch/repos/lucid
L=figs/out/md56_logs; PY=figs/.venv/bin/python
t() { echo "[$(date -u +%F' '%T) UTC $(hostname -s)] $* (load $(cut -d' ' -f1 /proc/loadavg))"; }
export FIG_TAG=f5fix
unset FIG_VARIANT FIG_VARIANT_REQUIRE

t "F5FIX 1/4 controls at >=6 displays"
FIG5_MIN_DISPLAYS=6 $PY figs/fig5_controls.py > $L/F5FIX_controls.log 2>&1
t "controls exit=$?"; cat $L/F5FIX_controls.log

t "F5FIX 2/4 rear coupling, per direction"
if [ -e figs/out/fig5_rear_coupling_2animal_perdir.json ]; then
  t "REFUSING: figs/out/fig5_rear_coupling_2animal_perdir.json exists (never overwrite deposits)"
else
  $PY figs/fig5_rear_coupling.py --per-direction --slap-animals 2 --skip-slap --jobs 12 \
      --out figs/out/fig5_rear_coupling_2animal_perdir.json > $L/F5FIX_coupling.log 2>&1
  t "coupling exit=$?"
fi
cat $L/F5FIX_coupling.log

t "F5FIX 3/4 Fig 5G panel (per-direction far/null), FIG_TAG=$FIG_TAG"
FIG5G_PERDIR=1 $PY figs/panels/fig5_12_coupling.py > $L/F5FIX_panel.log 2>&1
t "panel exit=$?"; cat $L/F5FIX_panel.log
ls -la figs/figures/fig5f5fix/ figs/data/fig5f5fix/ 2>&1

t "F5FIX 4/4 old vs new"
$PY $L/compare_f5fix.py 2>&1 | tee $L/F5FIX_compare.txt
echo "F5FIX DONE"
