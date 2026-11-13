#!/bin/bash
# Run the LUC3D node driver on LP3D-aggregated H5 detections.
# Assumes /root/vast/eric/luc3d-bench/outputs/lp3d_predictions_h5s/{cam}_predictions.h5 exists.
set -euo pipefail

REPO=/root/vast/eric/luc3d-bench
PY=$REPO/lp3d_env/bin/python
LP3D_H5_DIR=$REPO/outputs/lp3d_predictions_h5s
LUC3D_OUT=$REPO/outputs/luc3d_lp3d_results
LOG_DIR=$REPO/logs/luc3d_lp3d

# The same 10 inference sessions
SESSIONS=(
  "10072022124146" "10072022125259"
  "10072022172040" "10072022175009"
  "10072022194521" "10072022200754"
  "10072022184618" "10202022163211"
  "10072022161055" "10072022162233"
)

mkdir -p "$LUC3D_OUT" "$LOG_DIR"

for sess in "${SESSIONS[@]}"; do
  out_path="$LUC3D_OUT/$sess.json"
  if [ -f "$out_path" ]; then
    echo "[skip] $sess"
    continue
  fi
  # Look up session_idx, num_animals, calibration from master sheet
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
    > "$LOG_DIR/$sess.log" 2>&1 || echo "  FAIL $sess (see $LOG_DIR/$sess.log)"
done
echo "Done."
