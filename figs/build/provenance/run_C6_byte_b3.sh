#!/usr/bin/env bash
# Fig 6c ByteTrack arm, b3 rules: track only (score later). ericmimicc.
set -uo pipefail
cd /root/vast/eric/sleap-3d-gui/scratch/repos/lucid
ST=figs/figures/drafts/RUN-STATUS-2026-10-15.md
env -u SCORE_NOTAIL BYTE_B3=1 /root/vast/eric/luc3d-bench/eks_env/bin/python figs/fig6_bytetrack_max2.py --stage track --workers 6 > figs/out/md56_logs/C6_byte_track.log 2>&1
echo "- $(date -u +%H:%M) | ericmimicc | C6 ByteTrack b3 | track exit=$? $(tail -1 figs/out/md56_logs/C6_byte_track.log) -> results/bytetrack_noretire_b3 (scoring not yet run)" >> $ST
