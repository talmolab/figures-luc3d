#!/usr/bin/env bash
# B1 on the main box (moved from ericmimicc at 06:40 UTC, 2026-10-15; that machine lacks libGL).
# Mouse-Dyad-10M SLEAP tail-free re-track for Supp Fig 6A. Same steps and gates as
# HANDOFF-OTHER-JOB-2026-10-15.md section 3. Each step must pass before the next.
set -uo pipefail
cd /root/vast/eric/sleap-3d-gui/scratch/repos/lucid/figs
L=out/md56_logs; ST=figures/drafts/RUN-STATUS-2026-10-15.md
SN=/root/vast/eric/sleap_nn_030_env/bin/python        # sleap-nn 0.3.0, sleap-io 0.8.0, py 3.11.14 (uv)
B=/root/vast/eric/luc3d-bench/outputs/bmimica
W=26
export SLEAP_PY=$SN
st() { echo "- $(date -u +'%F %T') UTC $(hostname -s) [main box] $*" | tee -a $ST; }
fail() { st "B1 FAILED: $*"; echo "B1 FAILED"; exit 1; }

$SN -c "import sys,sleap_nn,sleap_io,numpy,torch; print(sys.version.split()[0],sleap_nn.__version__,sleap_io.__version__,numpy.__version__,torch.__version__)" > $L/B1_versions.log 2>&1
grep -q "^3.11.14 0.3.0 0.8.0 2.4.6 2.12.1" $L/B1_versions.log || fail "env versions: $(cat $L/B1_versions.log)"
st "B1 start (main box): versions $(cat $L/B1_versions.log)"

# Gate 3.1: reproduce the deposited tail-on run on 20250827_141755 exactly.
CTRL=$B/ctrl_max2_mainbox_20251015
ST1=--single-thread
$SN fig6_sleap_max2_retrack.py --out-dir $CTRL --only-session 20250827_141755 --workers 5 $ST1 > $L/B1_ctrl.log 2>&1
if ! $SN fig6_sleap_max2_retrack.py --out-dir $CTRL --compare-to $B/retracked_max2 --only-session 20250827_141755 >> $L/B1_ctrl.log 2>&1; then
  st "B1 gate: differs (or failed) with --single-thread: $(grep -E '^compare|done:' $L/B1_ctrl.log | tr '\n' ' '); retrying without"
  CTRL=${CTRL}_nosinglethread; ST1=""
  $SN fig6_sleap_max2_retrack.py --out-dir $CTRL --only-session 20250827_141755 --workers 5 > $L/B1_ctrl_nst.log 2>&1
  $SN fig6_sleap_max2_retrack.py --out-dir $CTRL --compare-to $B/retracked_max2 --only-session 20250827_141755 >> $L/B1_ctrl_nst.log 2>&1 \
    || fail "gate 3.1 differs both with and without --single-thread: $(grep -E '^compare|done:' $L/B1_ctrl_nst.log | tr '\n' ' ')"
fi
st "B1 gate 3.1 PASS ($([ -n "$ST1" ] && echo with || echo without) --single-thread): $(grep '^compare' $L/B1_ctrl*.log | tail -1)"

# 3.2 prep + check
$SN fig6_sleap_max2_retrack.py --notail --prep --workers $W > $L/B1_prep.log 2>&1 || fail "prep"
$SN fig6_sleap_max2_retrack.py --notail --check-prep --workers $W > $L/B1_checkprep.log 2>&1 || fail "check-prep: $(tail -3 $L/B1_checkprep.log | tr '\n' ' ')"
grep -q "check-prep: all 280 OK" $L/B1_checkprep.log || fail "check-prep count: $(tail -1 $L/B1_checkprep.log)"
st "B1 prep checked: $(tail -1 $L/B1_checkprep.log)"

# 3.3 track
$SN fig6_sleap_max2_retrack.py --notail $ST1 --workers $W > $L/B1_track.log 2>&1 || fail "track"
tail -1 $L/B1_track.log | grep -q "^done: 280 ok, 0 failed" || fail "track: $(tail -1 $L/B1_track.log)"

# 3.4 verify (<=2 tracks, no tail in any output)
$SN fig6_sleap_max2_retrack.py --notail --verify > $L/B1_verify.log 2>&1 || fail "verify: $(tail -2 $L/B1_verify.log | tr '\n' ' ')"
grep -q "280 outputs at <= 2 tracks, 0 ABOVE the cap" $L/B1_verify.log || fail "verify count: $(tail -2 $L/B1_verify.log | tr '\n' ' ')"
grep -q "TAIL PRESENT" $L/B1_verify.log && fail "tail present in outputs"
# run_OVERNIGHT.sh waits on this filename (it was written for the other machine).
ln -sfn B1_verify.log $L/OTHER_sleapnt_verify.log
st "B1 DONE (main box): $(grep 'ABOVE the cap' $L/B1_verify.log) -> $B/retracked_max2_notail (scored by run_OVERNIGHT.sh)"
echo "B1 DONE"
