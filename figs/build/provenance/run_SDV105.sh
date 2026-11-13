#!/usr/bin/env bash
# SDV105 (prepared 2026-10-15): Supplementary Fig 6G-I (social-DANNCE, 41 rat sessions:
# TRIADS 6, BEDDING 6, SCN2A 29 = soc1 14 + soc3 15) re-run on LUC3D v1.0.5
# (pose/ at /root/vast/eric/lucid-main-md56, commit 2dc001fe) instead of the
# lucid-switch-correct worktree (c9b2a325) the deposit was made with.
#
# Same hyperparameters as the deposit: no --params, so the tracker runs on the hooks'
# THRESHOLD_DEFAULTS (sync association via runCrossViewTracker, stale 20,
# distanceThreshold 25, corr3dWeight 6, corr2dWeight 1, velocityThreshold 10,
# timePenalty 0.1, maxTargets = #animals, all 23 node weights 1; v1.0.5 adds
# matchGate 0 = off). Same per-cell prep seeds (fig10_run.cell_seed), same cells:
#   C1_sigma0 (6G, 6I, all text numbers) + C7c_cams{2,3,4,5}_sigma0 (6H; k=6 is C1).
# Controls: the SAME single sample as the deposit -- SCN2A soc1 2022_09_22_M3_M4,
# first 20,000 frames, prep seed 99 (fig10_controls.py defaults).
#
# Everything new lands in figs/fig10-bench/results_v105/ (FIG10_RESULTS) and
# figs/fig10-bench/work_v105/ (scratch); panels in figures/fig6sdv105/ + data/fig6sdv105/
# (FIG_TAG=sdv105, FIG6_SDANNCE_AGG). The deposited results/ and figures/fig6 are not touched.
# Resumable: fig10_run.py skips cells whose JSON already exists under results_v105/.
set -uo pipefail
cd /root/vast/eric/sleap-3d-gui/scratch/repos/lucid/figs
L=$PWD/out/md56_logs; B=$PWD/fig10-bench
PY=$PWD/.venv/bin/python
JOBS=${JOBS:-12}
export PATH=/root/vast/eric/node26_env/bin:$PATH
export LUCID_SC_ROOT=/root/vast/eric/lucid-main-md56
export FIG10_RESULTS=$B/results_v105 FIG10_WORK=$B/work_v105
t() { echo "[$(date -u +%F' '%T) UTC] $* (load $(cut -d' ' -f1 /proc/loadavg))"; }

REV=$(git -C $LUCID_SC_ROOT rev-parse --short=8 HEAD)
[ "$REV" = "2dc001fe" ] || { t "LUCID_SC_ROOT is at $REV, not v1.0.5 2dc001fe -- refusing"; echo "SDV105 DONE (FAILED)"; exit 1; }
# fig10-bench/node_modules was a dangling link to the renamed fig8-bench; it now points
# at ../fig6-bench/node_modules (h5wasm 0.7.9). Check it resolves before burning compute.
[ -e $B/node_modules/h5wasm/package.json ] || { t "fig10-bench/node_modules/h5wasm missing"; echo "SDV105 DONE (FAILED)"; exit 1; }
mkdir -p $FIG10_RESULTS/agg/swapdiag $FIG10_WORK
t "SDV105 start: pose=$LUCID_SC_ROOT@$REV results=$FIG10_RESULTS jobs=$JOBS node=$(node --version)"

# 1. C0 audit on v1.0.5: one C1_sigma0 cell per family, effective config read back.
(cd $B && $PY fig10_c0_audit.py --jobs 4) > $L/SDV105_c0audit.log 2>&1
t "c0 audit exit=$? $(grep -E '\"(triads|bedding|soc1|soc3)\"' $L/SDV105_c0audit.log | tr -d ' \n')"

# 2. The cells: 41 sessions x 5 cells = 205.
CELLS=C1_sigma0,C7c_cams2_sigma0,C7c_cams3_sigma0,C7c_cams4_sigma0,C7c_cams5_sigma0
(cd $B && $PY fig10_run.py --datasets triads,bedding,soc1,soc3 --cells $CELLS --jobs $JOBS) > $L/SDV105_run.log 2>&1
t "run exit=$? ok=$(grep -c '] OK ' $L/SDV105_run.log) skip=$(grep -c '] SKIP ' $L/SDV105_run.log) fail=$(grep -cE 'FAIL' $L/SDV105_run.log)"
grep -E "FAIL" $L/SDV105_run.log | head

# 3. Aggregate (missing.txt will list the noisy/dropout cells, which were not re-run).
(cd $B && $PY fig10_aggregate.py) > $L/SDV105_aggregate.log 2>&1; t "aggregate exit=$?"

# 4. Swap diagnostics for every C1_sigma0 session with a switch (feeds 6I labels and the
#    "within 6 mm" / "45 mm" / swap-frame numbers). Same prep seed as the scored cell.
$PY - > $FIG10_WORK/swap_list.txt <<'EOF'
import glob, json, os, sys
sys.path.insert(0, 'fig10-bench')
from fig10_run import DATASETS, cell_seed
R = os.environ['FIG10_RESULTS']
for p in sorted(glob.glob(f'{R}/*/*/C1_sigma0.json')):
    ds, sess = p.split('/')[-3], p.split('/')[-2]
    if json.load(open(p))['switches'] > 0:
        print(ds, sess, os.path.join(DATASETS[ds], sess), cell_seed(ds, sess, 'C1_sigma0'), p)
EOF
t "swap sessions: $(wc -l < $FIG10_WORK/swap_list.txt)"
rm -f $FIG10_RESULTS/agg/swapdiag/*.json
xargs -P 6 -L 1 bash -c '
  W=$FIG10_WORK/swapdiag__$0__$1; rm -rf $W
  (cd '"$B"' && '"$PY"' fig10_swapdiag.py --session $2 --work $W --seed $3 \
     --out $FIG10_RESULTS/agg/swapdiag/$0__$1.json --scored $4) > '"$L"'/SDV105_swapdiag__$0__$1.log 2>&1
  echo "swapdiag $0/$1 exit=$?"; rm -rf $W' < $FIG10_WORK/swap_list.txt
t "swapdiag done: $(ls $FIG10_RESULTS/agg/swapdiag | wc -l) files"

# 5. Panel data (10f merge joined with the new C1_sigma0 scores; 10b residuals are
#    tracker-independent and simply re-derived).
(cd $B && $PY fig10_paneldata.py) > $L/SDV105_paneldata.log 2>&1; t "paneldata exit=$? $(tail -n 1 $L/SDV105_paneldata.log)"

# 6. Verification controls on v1.0.5, same single sample as the deposit.
(cd $B && $PY fig10_controls.py --session /root/vast/eric/s-DANNCE-data/s-DANNCE-SCN2A_SOC1/2022_09_22_M3_M4 --frames 20000) > $L/SDV105_controls.log 2>&1
t "controls exit=$?"; grep -E "^(A|B|C) |baseline" $L/SDV105_controls.log

# 7. Panels 6G/H/I under the new tag.
for p in fig6_09_sdannce_survival fig6_10_sdannce_cameras fig6_11_sdannce_contact; do
  FIG6_SDANNCE_AGG=$FIG10_RESULTS/agg FIG_TAG=sdv105 $PY panels/$p.py > $L/SDV105_$p.log 2>&1
  t "$p exit=$? $(grep -E 'wrote|note' $L/SDV105_$p.log | tr '\n' ' ' | cut -c1-300)"
done
rm -rf $FIG10_WORK/controls

# 8. Old vs new, every s-DANNCE number in the tex.
$PY $L/compare_sdv105.py $B/results/agg $FIG10_RESULTS/agg > $L/SDV105_compare.txt 2>&1
t "compare exit=$? -> $L/SDV105_compare.txt"; cat $L/SDV105_compare.txt
t "SDV105 DONE"
echo "SDV105 DONE"
