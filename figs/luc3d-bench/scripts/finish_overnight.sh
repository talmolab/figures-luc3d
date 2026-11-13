#!/bin/bash
# Run-everything-and-finish-cleanly wrapper. Designed to be launched via
#   nohup setsid bash scripts/finish_overnight.sh > /tmp/finish_overnight.log 2>&1 &
# so it survives the Claude Code harness exiting.
#
# What it does:
#   1. Wait for Track A's 12-config sweep + eval to complete
#   2. Wait for Track B's CrossViewTracker run to finish on remaining sessions
#   3. Wait for Track C's full-bench eval CSV
#   4. Merge all CSVs into the metrics CSV
#   5. Regenerate make_plots_paf.py
#   6. Write done marker at /tmp/finish_overnight.DONE
#
# Safe to call multiple times — uses skip-existing patterns.

set -uo pipefail
cd /root/vast/eric/luc3d-bench
source .venv/bin/activate

LOG=/tmp/finish_overnight.log
DONE=/tmp/finish_overnight.DONE
rm -f "$DONE"
echo "[$(date)] start finish_overnight" >> "$LOG"

# ----------------------------------------------------------------------------
# 1. Track A — make sure all 12 configs are stitched + evaluated.
# ----------------------------------------------------------------------------
for wg in 0.5 1.0 2.0 5.0; do
  for wv in 0.0 0.5 1.0; do
    cfg="wg${wg}_wv${wv}"
    out_dir="outputs/PAF_3d_kalman/luc3d_l1_tracklet_3d_results/$cfg"
    eval_csv="outputs/PAF_3d_kalman/_eval_tracklet_3d_${cfg}.csv"
    # Stitch if missing.
    if [ ! -d "$out_dir" ] || [ "$(ls "$out_dir" 2>/dev/null | wc -l)" -lt 74 ]; then
      echo "[$(date)] Track A: stitching $cfg" >> "$LOG"
      python scripts/stitch_postprocess.py \
        --in outputs/PAF_3d_kalman/luc3d_l1_only_results \
        --out "$out_dir" \
        --mode tracklet_vote_3d --tracklet-gap 30 \
        --w-geom "$wg" --w-velo "$wv" --sigma-geom 50 --sigma-velo 20 \
        --pred-h5-dir outputs/predictions_h5s \
        --master-sheet outputs/predictions_master_sheet.tsv \
        --progress-log "$LOG" >> "$LOG" 2>&1
    fi
    # Eval if missing.
    if [ ! -f "$eval_csv" ]; then
      echo "[$(date)] Track A: eval $cfg" >> "$LOG"
      python scripts/evaluate.py \
        --luc3d-dir "$out_dir" \
        --out-csv "$eval_csv" \
        --workers 4 >> "$LOG" 2>&1
    fi
  done
done

# ----------------------------------------------------------------------------
# 2. Track B — finish remaining CrossViewTracker sessions if any.
# ----------------------------------------------------------------------------
n_b=$(ls outputs/PAF_3d_kalman/liezl_cross_view_results/*.json 2>/dev/null | wc -l)
if [ "$n_b" -lt 74 ] && [ -f "/root/vast/eric/luc3d-bench/scripts/liezl/run_all_sessions.py" ]; then
  echo "[$(date)] Track B: continuing — currently $n_b/74 sessions" >> "$LOG"
  /root/vast/eric/luc3d-bench/liezl_env/bin/python scripts/liezl/run_all_sessions.py \
    --workers 8 --skip-existing >> "$LOG" 2>&1 || true
fi
# Eval Track B once at least some sessions are done.
if [ ! -f "outputs/PAF_3d_kalman/_eval_liezl.csv" ] && [ "$n_b" -gt 0 ]; then
  echo "[$(date)] Track B: eval" >> "$LOG"
  python scripts/evaluate.py \
    --luc3d-dir outputs/PAF_3d_kalman/liezl_cross_view_results \
    --out-csv outputs/PAF_3d_kalman/_eval_liezl.csv \
    --workers 8 >> "$LOG" 2>&1 || true
fi

# ----------------------------------------------------------------------------
# 3. Track C — full-bench ReID eval CSV.
# ----------------------------------------------------------------------------
if [ ! -f "outputs/PAF_3d_kalman/_eval_tracklet_reid_full.csv" ]; then
  echo "[$(date)] Track C: full ReID eval" >> "$LOG"
  python scripts/evaluate.py \
    --luc3d-dir outputs/PAF_3d_kalman/luc3d_l1_tracklet_reid_results \
    --out-csv outputs/PAF_3d_kalman/_eval_tracklet_reid_full.csv \
    --workers 8 >> "$LOG" 2>&1
fi

# ----------------------------------------------------------------------------
# 4. Merge + plots.
# ----------------------------------------------------------------------------
echo "[$(date)] merging variants" >> "$LOG"
python scripts/merge_all_variants.py >> "$LOG" 2>&1 || true
echo "[$(date)] regenerating plots" >> "$LOG"
python scripts/make_plots_paf.py >> "$LOG" 2>&1 || true

echo "[$(date)] DONE" >> "$LOG"
touch "$DONE"
