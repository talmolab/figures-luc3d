#!/usr/bin/env bash
# Fig 6c SLEAP arm under the b3 rules (body >= 5 pool, tail in tracking), ericmimicc, 2026-10-15.
# Writes only retracked_b3_src/ and retracked_max2_b3/ (new dirs) + C6_* logs.
set -uo pipefail
cd /root/vast/eric/sleap-3d-gui/scratch/repos/lucid/figs
L=out/md56_logs; ST=figures/drafts/RUN-STATUS-2026-10-15.md
SN=/root/vast/eric/sleap_nn_030_env/bin/python
export SLEAP_PY=$SN HDF5_USE_FILE_LOCKING=FALSE
W=58
st() { echo "- $(date -u +'%H:%M') | ericmimicc | C6 SLEAP b3 | $*" >> $ST; echo "$*"; }
fail() { st "FAILED: $*"; exit 1; }
$SN fig6_sleap_max2_retrack.py --b3 --prep --workers 40 > $L/C6_sleap_prep.log 2>&1 || fail "prep: $(tail -1 $L/C6_sleap_prep.log)"
$SN fig6_sleap_max2_retrack.py --b3 --check-prep --workers 40 > $L/C6_sleap_checkprep.log 2>&1 || fail "check-prep: $(tail -1 $L/C6_sleap_checkprep.log)"
grep -q "check-prep: 280 of 280 OK" $L/C6_sleap_checkprep.log || fail "check-prep count"
st "input gate PASS: 280/280 SLEAP inputs equal det_h5_bvis5 exactly; tracking started"
$SN fig6_sleap_max2_retrack.py --b3 --single-thread --workers $W > $L/C6_sleap_track.log 2>&1 || fail "track"
tail -1 $L/C6_sleap_track.log | grep -q "^done: 280 ok, 0 failed" || fail "track: $(tail -1 $L/C6_sleap_track.log)"
$SN fig6_sleap_max2_retrack.py --b3 --verify > $L/C6_sleap_verify.log 2>&1 || fail "verify: $(tail -2 $L/C6_sleap_verify.log | tr '\n' ' ')"
grep -q "280 outputs at <= 2 tracks, 0 ABOVE the cap" $L/C6_sleap_verify.log || fail "verify count"
st "tracked + verified: $(grep 'ABOVE the cap' $L/C6_sleap_verify.log) -> retracked_max2_b3"
echo C6SLEAP DONE
