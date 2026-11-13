#!/usr/bin/env bash
# md56 phase U (2026-10-14): bench inputs for the six 20250908 Mouse-Dyad-10M sessions.
# Every step is resumable and skips the 50 sessions whose outputs already exist, so this only
# ADDS files. bmimica_master.tsv was backed up to bmimica_master.tsv.pre-md56.bak.
# Plan: figs/figures/drafts/MOUSEDYAD-56-RERUN-PLAN.md
set -uo pipefail
cd /root/vast/eric/luc3d-bench
L=/root/vast/eric/sleap-3d-gui/scratch/repos/lucid/figs/out/md56_logs
FIGS=/root/vast/eric/sleap-3d-gui/scratch/repos/lucid/figs
export SLEAP_PY=/root/vast/eric/sleap_nn_030_fix/bin/python      # sleap-nn 0.3.0, sleap-io 0.8.0
export PYTHONPATH=/tmp/claude-0/-root-vast-eric-sleap-3d-gui-scratch-repos-lucid/44adbb8d-7377-41ea-ab2c-5dae7a320d17/scratchpad/cv2h   # OpenCV 4.13.0 headless
LP=/root/vast/eric/luc3d-bench/lp3d_env/bin/python
EKS=/root/vast/eric/luc3d-bench/eks_env/bin/python
CTRL=/tmp/claude-0/-root-vast-eric-sleap-3d-gui-scratch-repos-lucid/0158ef19-7a81-4590-9f2e-38e6639c9e8b/scratchpad/ctrl_max2
NEW="20250908_152229,20250908_154926,20250908_161812,20250908_164528,20250908_171240,20250908_173928"
t() { echo "[$(date +%T)] $*"; }

t "start (load $(cut -d' ' -f1 /proc/loadavg))"
# Control: one OLD session through the repaired env, to compare with the deposited retracked_max2.
( $SLEAP_PY -c "import sleap_nn;assert sleap_nn.__version__=='0.3.0'" &&
  $LP $FIGS/fig6_sleap_max2_retrack.py --out-dir $CTRL --only-session 20250827_141755 --workers 5 > $L/ctrl_max2.log 2>&1
  t "control max2 exit=$?" ) &

$LP scripts/bartul/bmimica_build_gt.py --workers 6 > $L/U2_gt.log 2>&1;        t "U2 gt exit=$?"
$LP scripts/bartul/bmimica_build_master.py > $L/U3_master.log 2>&1;             t "U3 master exit=$? rows=$(wc -l < outputs/bmimica/bmimica_master.tsv)"
$LP scripts/bartul/bmimica_retrack_all.py --workers 16 > $L/U1_retrack.log 2>&1; t "U1 retrack exit=$?"
$LP scripts/bartul/bmimica_build_inputs.py --workers 16 > $L/U4_inputs.log 2>&1; t "U4 inputs exit=$?"

( cd /root/vast/eric/sleap-3d-gui/scratch/repos/lucid
  $LP figs/fig6_sleap_max2_retrack.py --workers 12 --sessions $NEW > $L/U5_max2.log 2>&1; t "U5 max2 exit=$?"
  $LP figs/fig6_sleap_max2_retrack.py --verify > $L/U5_verify.log 2>&1;             t "U5 verify exit=$?" ) &
( cd /root/vast/eric/sleap-3d-gui/scratch/repos/lucid
  SCORE_NOTAIL=1 BYTE_B1_DIR=/root/vast/eric/luc3d-bench/outputs/bmimica/results/bytetrack_noretire_notail \
    $EKS figs/fig6_bytetrack_max2.py --stage track --workers 10 --sessions $NEW > $L/U6_byte.log 2>&1; t "U6 byte exit=$?" ) &
wait
t "PHASE U DONE"
