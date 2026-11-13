#!/usr/bin/env bash
# Master overnight job: full UDMT on all 6 cameras of the hardest 4-animal session
# (2 at a time across the 2 A40s), then score each vs proofread GT, then build figures.
# Designed to be launched via the Bash tool with run_in_background:true so completion
# notifies the agent. Robust: per-camera/per-stage failures are logged, not fatal.
set -uo pipefail
SESSION=10072022145420
SPATH=2022-10-07/10072022145420
ROOT=/root/talmolab-smb/eric/slap_2m/$SPATH
BENCH=/root/vast/eric/luc3d-bench
LOGDIR=$BENCH/logs
HERE=$BENCH/scripts/udmt
VENV=$BENCH/.venv/bin/python
OUTCSV=$BENCH/outputs/PAF_3d_kalman/_eval_udmt_hardsession.csv
CAMS=(back backL mid midL top topL)
rm -f "$OUTCSV"
echo "[$(date)] === UDMT overnight master START ==="

resolve_video(){ ls "$ROOT/$1/$1-"*_h265_CRF12_denoised.mp4 2>/dev/null | head -1; }
proj_dir(){ ls -d /root/vast/eric/UDMT_runs/${SESSION}_$1-* 2>/dev/null | head -1; }

run_cam(){
  local cam="$1" gpu="$2"; local vid; vid=$(resolve_video "$cam")
  local L="$LOGDIR/udmt_${SESSION}_${cam}.log"
  if [ -z "$vid" ]; then echo "NO VIDEO $cam" >> "$L"; return 1; fi
  echo "[$(date)] START $cam GPU$gpu" >> "$L"
  bash "$HERE/run_udmt_camera.sh" "$SESSION" "$cam" "$vid" "$gpu" >> "$L" 2>&1
  echo "[$(date)] DONE $cam rc=$?" >> "$L"
}

# ---- Stage 1: run all 6 cameras, 2 concurrent (one per GPU) ----
for pair in "mid midL" "backL top" "topL back"; do
  set -- $pair
  run_cam "$1" 0 & P1=$!
  run_cam "$2" 1 & P2=$!
  wait $P1 $P2
  echo "[$(date)] round done: $pair"
done
echo "[$(date)] === all camera pipelines finished ==="

# ---- Stage 2: score each camera vs proofread GT (bench .venv) ----
for cam in "${CAMS[@]}"; do
  vid=$(resolve_video "$cam"); stem=$(basename "$vid" .mp4)
  proj=$(proj_dir "$cam")
  if [ -z "$proj" ] || ! ls "$proj"/tracking-results/*/*-whole-filter*.npy >/dev/null 2>&1; then
    echo "[$(date)] SCORE skip $cam (no tracking results)"; continue
  fi
  echo "[$(date)] SCORE $cam"
  PYTHONPATH=$BENCH $VENV "$HERE/score_udmt.py" \
    --session "$SESSION" --session-path "$SPATH" --cam "$cam" \
    --proj "$proj" --vid "$stem" --out-csv "$OUTCSV" \
    >> "$LOGDIR/udmt_score.log" 2>&1 || echo "score failed for $cam" >> "$LOGDIR/udmt_score.log"
done

# ---- Stage 3: figures ----
echo "[$(date)] building figures"
# single-session figures ONLY — do NOT touch the 252-session aggregate (figure_method_comparison.png
# stays UDMT-free until all sessions are run; per Eric 2026-06-30).
$VENV "$HERE/make_plots_udmt_hardsession.py" >> "$LOGDIR/udmt_figs.log" 2>&1 || echo "hardsession percam fig failed" >> "$LOGDIR/udmt_figs.log"
$VENV "$BENCH/scripts/make_plots_compare_small.py" >> "$LOGDIR/udmt_figs.log" 2>&1 || echo "small aggregate fig failed" >> "$LOGDIR/udmt_figs.log"
echo "[$(date)] === UDMT overnight master DONE ==="
