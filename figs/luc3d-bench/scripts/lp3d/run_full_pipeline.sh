#!/bin/bash
# End-to-end pipeline runner for LP3D detection swap benchmark.
# Run after training completes (lp3d_models/slap2m_mvt/.../*.ckpt exists).
set -euo pipefail

REPO=/root/vast/eric/luc3d-bench
PY=$REPO/lp3d_env/bin/python
LITPOSE=$REPO/lp3d_env/bin/litpose

# Inference sessions (disjoint from training)
INFER_SESSIONS=(
  "10072022124146" "10072022125259"    # d=1
  "10072022172040" "10072022175009"    # d=3
  "10072022194521" "10072022200754"    # d=5
  "10072022184618" "10202022163211"    # d=6
  "10072022161055" "10072022162233"    # d=7
)

VAL_SESSION="10202022145302"  # d=3, single-animal, held-out

MODEL_DIR=$1
if [ -z "$MODEL_DIR" ]; then
  echo "Usage: $0 <model_dir>"
  exit 1
fi
LP3D_CSV_ROOT=$REPO/outputs/lp3d_csv_preds
LP3D_H5_DIR=$REPO/outputs/lp3d_predictions_h5s
LUC3D_OUT=$REPO/outputs/luc3d_lp3d_results
METRICS_DIR=$REPO/outputs/metrics/lp3d_swap

mkdir -p "$LP3D_CSV_ROOT" "$LP3D_H5_DIR" "$LUC3D_OUT" "$METRICS_DIR"

echo "=== Phase 2.5: Validation on held-out session $VAL_SESSION ==="
$PY $REPO/scripts/lp3d/validate_model.py \
  --model-dir "$MODEL_DIR" \
  --session-id "$VAL_SESSION" \
  --out-dir "$REPO/outputs/lp3d_val/$VAL_SESSION" \
  --max-frames 2000

echo
echo "=== Phase 3: Inference on 10 sessions ==="
$PY $REPO/scripts/lp3d/run_lp3d_inference.py \
  --model-dir "$MODEL_DIR" \
  --sessions "${INFER_SESSIONS[@]}" \
  --out-csv-root "$LP3D_CSV_ROOT" \
  --skip-existing

echo
echo "=== Phase 3.5: Aggregate CSVs into per-cam H5 files ==="
$PY $REPO/scripts/lp3d/aggregate_lp3d_h5.py \
  --csv-root "$LP3D_CSV_ROOT" \
  --out-dir "$LP3D_H5_DIR" \
  --conf-thresh 0.0

echo
echo "=== Phase 4: Run LUC3D tracker with LP3D detections ==="
# Make a master sheet pointer (it's the same — we override --pred-h5-dir)
$PY $REPO/scripts/lp3d/make_lp3d_master_sheet.py
# Run LUC3D with LP3D detections — but the run script doesn't support overriding pred h5 dir;
# we hack it via env var. Actually the python wrapper hardcodes it. Let me invoke node directly.
for sess in "${INFER_SESSIONS[@]}"; do
  # find session_idx from master sheet
  idx=$($PY -c "
import pandas as pd
df=pd.read_csv('$REPO/outputs/predictions_master_sheet.tsv', sep='\t', index_col=0)
i=df[df['session'].astype(str)=='$sess'].index[0]
print(i)
")
  num_animals=$($PY -c "
import pandas as pd
df=pd.read_csv('$REPO/outputs/predictions_master_sheet.tsv', sep='\t', index_col=0)
r=df[df['session'].astype(str)=='$sess'].iloc[0]
print(int(r['num_animals']))
")
  calib=$($PY -c "
import pandas as pd
df=pd.read_csv('$REPO/outputs/predictions_master_sheet.tsv', sep='\t', index_col=0)
r=df[df['session'].astype(str)=='$sess'].iloc[0]
print(r['calibration_toml'])
")
  out_path="$LUC3D_OUT/$sess.json"
  if [ -f "$out_path" ]; then
    echo "[skip] $sess (output exists)"
    continue
  fi
  echo "[luc3d] $sess (idx=$idx, num_animals=$num_animals)"
  node $REPO/scripts/luc3d_track_all.mjs \
    --session-idx "$idx" \
    --num-animals "$num_animals" \
    --calibration "$calib" \
    --pred-h5-dir "$LP3D_H5_DIR" \
    --out "$out_path" \
    2> "$REPO/logs/luc3d_lp3d_${sess}.log" || echo "  FAIL $sess (see log)"
done

echo
echo "=== Phase 5: Evaluate ==="
$PY $REPO/scripts/evaluate.py \
  --luc3d-dir "$LUC3D_OUT" \
  --out-csv "$METRICS_DIR/per_camera_session_metrics.csv"

echo
echo "Done. Metrics at $METRICS_DIR/per_camera_session_metrics.csv"
