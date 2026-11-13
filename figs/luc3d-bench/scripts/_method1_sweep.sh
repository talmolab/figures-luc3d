#!/usr/bin/env bash
# Method 1: push Liezl 3D cross-view weight harder + combine with the clean tracked pool.
set -uo pipefail
cd /root/vast/eric/luc3d-bench
export PATH="/usr/local/bin:$PATH"
SESS=$(cat /tmp/multi_sessions.txt)
PYL=liezl_env/bin/python
PY=.venv/bin/python
B=outputs/PAF_3d_kalman/_liezlsweep
mkdir -p "$B" logs

# tag  master  pred_h5  corr3d  vel  dist  tpen
CONFIGS=(
  "E_detok_3d6      outputs/detections_only_master_sheet.tsv outputs/detections_only_h5s 6  10 50 0.1"
  "F_keeptrack_3d3  outputs/sleap_nn_master_sheet.tsv        outputs/keeptrack_h5s       3  10 50 0.1"
  "G_keeptrack_3d6  outputs/sleap_nn_master_sheet.tsv        outputs/keeptrack_h5s       6  10 50 0.1"
)
for cfg in "${CONFIGS[@]}"; do
  read -r TAG M H5 C3 VEL DIST TPEN <<< "$cfg"
  OUT="$B/$TAG"; CSV="$B/eval_$TAG.csv"
  echo "[$(date +%H:%M:%S)] $TAG (master=$(basename $M) corr3d=$C3)"
  $PYL scripts/liezl/run_all_sessions.py --master-sheet "$M" --out-dir "$OUT" \
    --only-sessions $SESS --workers 12 --corr3d "$C3" \
    --vel-thresh "$VEL" --dist-thresh "$DIST" --time-penalty "$TPEN" \
    > "logs/m1_$TAG.log" 2>&1
  $PY scripts/evaluate.py --luc3d-dir "$OUT" --pred-h5-dir "$H5" --master "$M" \
    --out-csv "$CSV" --no-sleap-baseline --no-bytetrack --workers 16 > /dev/null 2>&1
  PYTHONPATH=. $PY scripts/switch_breakdown.py "$OUT" "$H5" --label "$TAG" --min-run 30 --workers 12 2>&1 | tail -1
  $PY -c "
import pandas as pd
df=pd.read_csv('$CSV');df=df[(df.tracker=='luc3d')&(df.animals>=2)]
print(f'  $TAG: raw={int(df.num_switches.sum())} idf1={df.idf1.mean():.4f} idp={df.idp.mean():.4f}')
"
done
echo "=== METHOD1 DONE $(date +%H:%M:%S) ==="
