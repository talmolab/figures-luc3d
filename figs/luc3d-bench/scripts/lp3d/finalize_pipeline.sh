#!/bin/bash
# Final phases: re-aggregate, run LUC3D, evaluate.
set -euo pipefail

REPO=/root/vast/eric/luc3d-bench
PY=$REPO/lp3d_env/bin/python
LP3D_CSV_ROOT=$REPO/outputs/lp3d_csv_preds
LP3D_H5_DIR=$REPO/outputs/lp3d_predictions_h5s
LUC3D_OUT=$REPO/outputs/luc3d_lp3d_results
METRICS_DIR=$REPO/outputs/metrics/lp3d_swap

mkdir -p "$LP3D_H5_DIR" "$LUC3D_OUT" "$METRICS_DIR"

echo "=== Step 1: Re-aggregate LP3D CSV preds into per-cam H5s ==="
$PY $REPO/scripts/lp3d/aggregate_lp3d_h5.py \
  --csv-root "$LP3D_CSV_ROOT" \
  --out-dir "$LP3D_H5_DIR" \
  --conf-thresh 0.0

echo
echo "=== Step 2: Run LUC3D node tracker on LP3D H5s ==="
SESSIONS=(
  "10072022124146" "10072022125259"
  "10072022172040" "10072022175009"
  "10072022194521" "10072022200754"
  "10072022184618" "10202022163211"
  "10072022161055" "10072022162233"
)
LOG_DIR=$REPO/logs/luc3d_lp3d
mkdir -p "$LOG_DIR"

for sess in "${SESSIONS[@]}"; do
  out_path="$LUC3D_OUT/$sess.json"
  if [ -f "$out_path" ]; then
    echo "[skip] $sess"
    continue
  fi
  read idx num_animals calib < <($PY -c "
import pandas as pd
df = pd.read_csv('$REPO/outputs/predictions_master_sheet.tsv', sep='\t', index_col=0)
r = df[df['session'].astype(str)=='$sess'].iloc[0]
print(int(df[df['session'].astype(str)=='$sess'].index[0]), int(r['num_animals']), r['calibration_toml'])
")
  echo "[luc3d] $sess idx=$idx num=$num_animals"
  node $REPO/scripts/luc3d_track_all.mjs \
    --session-idx "$idx" \
    --num-animals "$num_animals" \
    --calibration "$calib" \
    --pred-h5-dir "$LP3D_H5_DIR" \
    --out "$out_path" \
    > "$LOG_DIR/$sess.log" 2>&1 || echo "  FAIL $sess"
done

echo
echo "=== Step 3: Evaluate ==="
$PY $REPO/scripts/evaluate.py \
  --luc3d-dir "$LUC3D_OUT" \
  --out-csv "$METRICS_DIR/per_camera_session_metrics.csv"

echo
echo "=== Step 4: Compare vs SLEAP baseline ==="
$PY $REPO/scripts/lp3d/compare_results.py \
  --lp3d-csv "$METRICS_DIR/per_camera_session_metrics.csv" \
  --out "$METRICS_DIR/comparison.csv"

echo
echo "Done. Report at $REPO/outputs/metrics/lp3d_swap_report.md"
