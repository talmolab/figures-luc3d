#!/usr/bin/env bash
# BMimica full pipeline orchestrator: waits for retrack + GT, then builds inputs,
# runs Liezl x6 + ByteTrack, evaluates, and plots. Resumable (each stage skips done work).
set -uo pipefail
cd /root/vast/eric/luc3d-bench
BM=/root/vast/eric/luc3d-bench/outputs/bmimica
LOG=$BM/orchestrate.log
export LIEZL_INPUT_CAMERAS="21241563,21369048,21372315,21372316,22085397"
say(){ echo "[orch $(date +%H:%M:%S)] $*" >> "$LOG"; }

say "waiting for retrack to finish..."
until grep -q "^DONE ok=" $BM/retrack_all.log 2>/dev/null; do sleep 120; done
say "retrack done: $(grep '^DONE ok=' $BM/retrack_all.log | tail -1)"
until grep -q "^DONE GT" $BM/build_gt.log 2>/dev/null; do sleep 30; done
say "gt done"

say "stage3: build detection H5s"
lp3d_env/bin/python scripts/bartul/bmimica_build_inputs.py --workers 16 >> $BM/build_inputs.log 2>&1
say "inputs done: $(tail -1 $BM/build_inputs.log)"

say "stage4: Liezl x6 + ByteTrack (parallel)"
liezl_env/bin/python scripts/liezl/run_all_sessions.py \
  --master-sheet $BM/bmimica_master.tsv --out-dir $BM/results/liezl_c6 \
  --corr3d 6 --workers 16 --skip-existing >> $BM/liezl_run.log 2>&1 &
LPID=$!
.venv/bin/python scripts/bartul/bmimica_bytetrack.py --workers 12 >> $BM/bytetrack_run.log 2>&1 &
BPID=$!
wait $LPID; say "liezl done"
wait $BPID; say "bytetrack done"

say "stage5: eval + figure"
.venv/bin/python scripts/bartul/bmimica_eval.py --workers 12 >> $BM/eval_run.log 2>&1
lp3d_env/bin/python scripts/bartul/bmimica_plot.py >> $BM/eval_run.log 2>&1
say "PIPELINE COMPLETE — figure at $BM/figure_bmimica_liezl_vs_bytetrack.png"
