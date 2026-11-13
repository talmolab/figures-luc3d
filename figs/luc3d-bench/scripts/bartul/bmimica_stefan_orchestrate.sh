#!/usr/bin/env bash
# Stefan-model BMimica rerun orchestrator: launches Stage 0 (real inference,
# 2 GPU workers) in the background, then repeatedly runs Stage 1 (tail-blind,
# uncapped tracking, CPU-only) to pick up whatever Stage 0 has finished so
# far, until Stage 0 completes and one final Stage 1 pass has swept up the
# stragglers. Both stages are resumable (skip files that already exist), so
# this is safe to re-run if interrupted.
#
# Scoped (2026-07-22) to the 3 hardest BMimica sessions by prior-benchmark
# swap count / worst LUC3D identity (see bmimica_crossview_all_eval.csv):
#   20250907_174343 (most LUC3D switches, 220)
#   20250829_171625 (2nd-most switches 180, 3rd-worst idf1 0.501)
#   20250907_151901 (worst LUC3D idf1, 0.489)
# 3 sessions x 5 cams = 15 videos, ~19h at ~40fps aggregate on 2 GPUs.
set -uo pipefail
cd /root/vast/eric/luc3d-bench
LOG_DIR=/root/vast/eric/stefan-bartul-model
LOG=$LOG_DIR/orchestrate.log
SESSIONS="20250907_174343 20250829_171625 20250907_151901"
say(){ echo "[orch $(date +%H:%M:%S)] $*" >> "$LOG"; }

say "stage0: launching real inference (2 GPU workers, cuda:0 + cuda:1) on sessions: $SESSIONS"
/root/vast/eric/sleap_nn_030_env/bin/python scripts/bartul/bmimica_infer_stefan.py \
  --workers 2 --gpus cuda:0 cuda:1 --only-sessions $SESSIONS >> "$LOG_DIR/infer_stefan.log" 2>&1 &
STAGE0_PID=$!
say "stage0 pid=$STAGE0_PID"

while kill -0 $STAGE0_PID 2>/dev/null; do
  say "stage1: sweeping tail-blind tracking over whatever stage0 has finished"
  /root/vast/eric/sleap_nn_030_env/bin/python scripts/bartul/bmimica_track_stefan_notail.py \
    --workers 12 --only-sessions $SESSIONS >> "$LOG_DIR/track_stefan_notail.log" 2>&1
  sleep 1800
done
wait $STAGE0_PID
say "stage0 done"

say "stage1: final sweep after stage0 completion"
/root/vast/eric/sleap_nn_030_env/bin/python scripts/bartul/bmimica_track_stefan_notail.py \
  --workers 12 --only-sessions $SESSIONS >> "$LOG_DIR/track_stefan_notail.log" 2>&1
say "PIPELINE COMPLETE — tracked output at $LOG_DIR/BMimica_tracked/"
