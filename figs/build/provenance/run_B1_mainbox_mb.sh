#!/usr/bin/env bash
# B1 on the main box, PRIVATE dirs (FIG6_SLEAPNT_DIRTAG=mb): retracked_notail_src_mb/ and
# retracked_max2_notail_mb/. 07:18 UTC: both machines prepped the shared retracked_notail_src/
# at once (HDF5 lock collision); the shared dirs are abandoned by the main box.
# Gate 3.1 already PASSED on this box at 07:07 (B1_ctrl.log: compare 5 identical, 0 different,
# --single-thread, same env and flags), so it is not rerun.
set -uo pipefail
cd /root/vast/eric/sleap-3d-gui/scratch/repos/lucid/figs
L=out/md56_logs; ST=figures/drafts/RUN-STATUS-2026-10-15.md
SN=/root/vast/eric/sleap_nn_030_env/bin/python
export SLEAP_PY=$SN FIG6_SLEAPNT_DIRTAG=mb
W=26
st() { echo "- $(date -u +'%F %T') UTC $(hostname -s) [main box] $*" | tee -a $ST; }
fail() { st "B1(mb) FAILED: $*"; echo "B1MB FAILED"; exit 1; }
grep -q "compare: 5 identical, 0 different" $L/B1_ctrl.log || fail "gate 3.1 evidence missing"
st "B1(mb) start: private dirs retracked_notail_src_mb / retracked_max2_notail_mb"
$SN fig6_sleap_max2_retrack.py --notail --prep --workers $W > $L/B1mb_prep.log 2>&1 || fail "prep: $(tail -1 $L/B1mb_prep.log)"
$SN fig6_sleap_max2_retrack.py --notail --check-prep --workers $W > $L/B1mb_checkprep.log 2>&1 || fail "check-prep: $(tail -2 $L/B1mb_checkprep.log | tr '\n' ' ')"
grep -q "check-prep: all 280 OK" $L/B1mb_checkprep.log || fail "check-prep count: $(tail -1 $L/B1mb_checkprep.log)"
st "B1(mb) prep checked: $(tail -1 $L/B1mb_checkprep.log)"
$SN fig6_sleap_max2_retrack.py --notail --single-thread --workers $W > $L/B1mb_track.log 2>&1 || fail "track"
tail -1 $L/B1mb_track.log | grep -q "^done: 280 ok, 0 failed" || fail "track: $(tail -1 $L/B1mb_track.log)"
$SN fig6_sleap_max2_retrack.py --notail --verify > $L/B1mb_verify.log 2>&1 || fail "verify: $(tail -2 $L/B1mb_verify.log | tr '\n' ' ')"
grep -q "280 outputs at <= 2 tracks, 0 ABOVE the cap" $L/B1mb_verify.log || fail "verify count"
grep -q "TAIL PRESENT" $L/B1mb_verify.log && fail "tail present"
st "B1(mb) DONE: $(grep 'ABOVE the cap' $L/B1mb_verify.log) -> /root/vast/eric/luc3d-bench/outputs/bmimica/retracked_max2_notail_mb"
echo "B1MB DONE"
