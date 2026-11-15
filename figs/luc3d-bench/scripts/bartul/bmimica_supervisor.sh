#!/usr/bin/env bash
# Fully-detached (setsid) supervisor for the whole 50-session BMimica pipeline.
# Resumable at every stage; relaunches the retrack if its master dies. Owns:
#   retrack -> build detection H5s -> (Liezl x6 + ByteTrack + 3D-MuPPET) -> eval -> figure
set -uo pipefail
cd /root/vast/eric/luc3d-bench
BM=/root/vast/eric/luc3d-bench/outputs/bmimica
LOG=$BM/supervisor.log
export LIEZL_INPUT_CAMERAS="21241563,21369048,21372315,21372316,22085397"
say(){ echo "[sup $(date +%m-%d_%H:%M:%S)] $*" >> "$LOG"; }
say "supervisor up (pid $$)"

# ---- Stage 1: ensure retrack completes (resumable; relaunch master if it dies) ----
while [ "$(find $BM/retracked -name '*.slp' 2>/dev/null | wc -l)" -lt 250 ] && \
      ! grep -q '^DONE ok=' $BM/retrack_all.log 2>/dev/null; do
  if ! pgrep -f bmimica_retrack_all >/dev/null; then
    say "retrack master not running -> relaunching (resumable)"
    lp3d_env/bin/python scripts/bartul/bmimica_retrack_all.py --workers 16 >> $BM/retrack_all.log 2>&1 &
  fi
  sleep 120
done
say "retrack complete ($(find $BM/retracked -name '*.slp' | wc -l) slp)"
until grep -q '^DONE GT' $BM/build_gt.log 2>/dev/null; do sleep 20; done

# ---- Stage 2: build detection H5s ----
say "build detection H5s"
lp3d_env/bin/python scripts/bartul/bmimica_build_inputs.py --workers 16 >> $BM/build_inputs.log 2>&1
say "inputs: $(tail -1 $BM/build_inputs.log)"

# ---- Stage 3: trackers in parallel (Liezl x6, ByteTrack, 3D-MuPPET) ----
say "trackers: Liezl x6 + ByteTrack + 3D-MuPPET"
liezl_env/bin/python scripts/liezl/run_all_sessions.py --master-sheet $BM/bmimica_master.tsv \
  --out-dir $BM/results/liezl_c6 --corr3d 6 --workers 14 --skip-existing >> $BM/liezl_run.log 2>&1 &
P1=$!
.venv/bin/python scripts/bartul/bmimica_bytetrack.py --workers 8 >> $BM/bytetrack_run.log 2>&1 &
P2=$!
/root/vast/eric/muppet_env/bin/python scripts/bartul/bmimica_muppet.py --workers 8 >> $BM/muppet_run.log 2>&1 &
P3=$!
wait $P1; say "liezl done"
wait $P2; say "bytetrack done"
wait $P3; say "muppet done"

# ---- Stage 4: eval + aggregate figure (Liezl vs 3D-MuPPET vs ByteTrack) ----
say "eval + figure"
.venv/bin/python scripts/bartul/bmimica_eval.py --workers 12 >> $BM/eval_run.log 2>&1
lp3d_env/bin/python scripts/bartul/bmimica_plot.py >> $BM/eval_run.log 2>&1
say "PIPELINE COMPLETE -> $BM/figure_bmimica_liezl_vs_bytetrack.png"
