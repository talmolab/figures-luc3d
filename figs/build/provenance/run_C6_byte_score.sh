#!/usr/bin/env bash
# Waits for C6 ByteTrack b3 tracking, then: pilot score (1 session), then full score (56). Tail-free scoring.
set -uo pipefail
cd /root/vast/eric/sleap-3d-gui/scratch/repos/lucid
ST=figs/figures/drafts/RUN-STATUS-2026-10-15.md; L=figs/out/md56_logs
PY=/root/vast/eric/luc3d-bench/liezl_env/bin/python
until grep -q "track done" $L/C6_byte_track.log 2>/dev/null; do sleep 120; done
export SCORE_NOTAIL=1 STITCH_DEDUPE=1 BYTE_B3=1
$PY figs/fig6_bytetrack_max2.py --stage score --sessions 20250827_141755 --arms B1,B1s --out tmp/c6_byte_pilot_score.json --workers 1 > $L/C6_byte_score_pilot.log 2>&1 \
  || { echo "- $(date -u +%H:%M) | ericmimicc | C6 ByteTrack b3 | PILOT SCORE FAILED, see C6_byte_score_pilot.log" >> $ST; exit 1; }
$PY figs/fig6_bytetrack_max2.py --stage score --arms B1,B1s --out fig7_bytetrack_max2_c6b3.json --workers 20 > $L/C6_byte_score.log 2>&1
echo "- $(date -u +%H:%M) | ericmimicc | C6 ByteTrack b3 | score exit=$? -> figs/out/fig7_bytetrack_max2_c6b3.json ($(grep 'B1s ' $L/C6_byte_score.log | tail -1))" >> $ST
