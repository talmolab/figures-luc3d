#!/usr/bin/env bash
# md56 phase T (2026-10-14): Mouse-Dyad-10M tracking results appended to all 56 sessions.
# Accuracy only, nothing timed. Mirrors bvis5nt's run_mousedyad.sh with md56 tags. The
# per-session caches are shared and keyed by file, so the 50 cached sessions are reused
# and only the six 2025-09-08 sessions are tracked. The `shipped` (pre-#210 "previous
# default") arm is pinned to distanceThreshold 50 in fig6_methods.py / fig3_sweep50.py,
# gated bit-identical on 20250827_141755. Fig 3 head-to-head is phase Q (timed).
set -uo pipefail
cd /root/vast/eric/sleap-3d-gui/scratch/repos/lucid
export PATH=/root/vast/eric/node26_env/bin:$PATH SCORE_NOTAIL=1
export SWEEP50_SUFFIX=md56 FIG8M_SUFFIX=md56 CAMS_SUFFIX=md56 FAIR_SUFFIX=md56
export CAMS_HH_CACHE=/root/vast/eric/sleap-3d-gui/scratch/repos/lucid/figs/out/tmp/headtohead_md56/A2_C5_bmimica
export BYTE_B1_DIR=/root/vast/eric/luc3d-bench/outputs/bmimica/results/bytetrack_noretire_notail
export PYTHONPATH=/tmp/claude-0/-root-vast-eric-sleap-3d-gui-scratch-repos-lucid/44adbb8d-7377-41ea-ab2c-5dae7a320d17/scratchpad/cv2h
PY=/root/vast/eric/luc3d-bench/liezl_env/bin/python; EKS=/root/vast/eric/luc3d-bench/eks_env/bin/python
L=figs/out/md56_logs; G="--grid-corr3d 0,0.25,0.5,1,2,3,4,6,8,12,16,24 --grid-corr2d 1"
t() { echo "[$(date +%T)] $*"; }
until grep -q "U4 inputs exit=0" $L/run_phaseU.out 2>/dev/null; do sleep 60; done
n=$(ls /root/vast/eric/luc3d-bench/outputs/bmimica/det_h5 | wc -l)
t "det_h5 sessions: $n"; [ "$n" -eq 56 ] || { t "ABORT: det_h5 has $n sessions"; exit 1; }

# cams needs fig8m's deposit for its k = 5 gate.
laneA() {
  FIG8M_WORKERS=20 $PY figs/fig6_methods.py --all-sessions --configs shipped,sync_stale1_dist25,sync_stale10_dist25,sync_stale20_dist25,sync_stale30_dist25,dist25_corr36 > $L/T_fig8m.log 2>&1
  t "A fig8m exit=$?"
  $PY figs/fig6_report50.py > $L/T_report50.log 2>&1; t "A report50 exit=$?"
  $PY figs/fig2_cams_identity.py --stage track --workers 20 > $L/T_cams_track.log 2>&1; t "A cams track exit=$?"
  $PY figs/fig2_cams_identity.py --stage score --workers 20 > $L/T_cams_score.log 2>&1; t "A cams score exit=$?"
}
# The two Fig 3G/H sweep arms are OFFLOADED to ericmimicc (run_S.sh, HANDOFF-MD56-SWEEPS.md).
laneB() {
  until grep -q "PHASE U DONE" $L/run_phaseU.out 2>/dev/null; do sleep 60; done
  grep -q "U5 verify exit=0" $L/run_phaseU.out || t "WARNING: U5 verify did not exit 0 -- check $L/U5_verify.log"
  $PY figs/fig6_sleap_scoped.py --retracked /root/vast/eric/luc3d-bench/outputs/bmimica/retracked_max2 --out fig7_sleap_max2_md56.json --workers 12 > $L/T_sleap.log 2>&1
  t "B sleap exit=$?"
  $PY figs/fig6_bytetrack_max2.py --stage score --arms B1,B1s --out fig7_bytetrack_max2_md56.json --workers 12 > $L/T_byte_score.log 2>&1
  t "B byte score exit=$?"
}
laneA & laneB & wait
$PY figs/fig6_fair_baselines.py > $L/T_fair.log 2>&1; t "fair exit=$?"
$PY figs/fig6_variant_best_rebuild.py --tag md56 --base bvis5nt > $L/T_variant.log 2>&1; t "variant rebuild exit=$? $(tail -n 1 $L/T_variant.log)"
ls -la figs/out/*_md56.json
t "PHASE T DONE"
