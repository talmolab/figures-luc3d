#!/usr/bin/env bash
# Keep-all-tracklets with sleap-nn FILTERS: Liezl cross-view on the sleap-nn
# filtered+TRACKED detections (capped to num_animals/frame, FP-free), fair eval.
set -uo pipefail
cd /root/vast/eric/luc3d-bench
export PATH="/usr/local/bin:$PATH"
SESS=$(cat /tmp/multi_sessions.txt)
PYL=liezl_env/bin/python
PY=.venv/bin/python
M=outputs/sleap_nn_master_sheet.tsv
H5=outputs/keeptrack_h5s
OUT=outputs/PAF_3d_kalman/keeptrack_liezl_results
mkdir -p logs

# 1. Aggregate filtered+tracked .slp -> canonical wide H5 (background)
echo "[$(date +%H:%M:%S)] aggregate tracked detections -> $H5"
PYTHONPATH=. sleap_nn_env/bin/python scripts/sleap_nn/aggregate_detections_h5.py \
  --detections-dir outputs/sleap_nn_retracked --out-dir "$H5" \
  --master "$M" --n-dets 5 --log logs/aggregate_keeptrack.log \
  > logs/aggregate_keeptrack.stdout 2>&1 &
AGG_PID=$!

# 2. Liezl cross-view on the filtered+tracked detections (multi-animal only)
echo "[$(date +%H:%M:%S)] Liezl on filtered+tracked detections (42 multi-animal sessions)"
$PYL scripts/liezl/run_all_sessions.py --master-sheet "$M" --out-dir "$OUT" \
  --only-sessions $SESS --workers 12 > logs/keeptrack_liezl.log 2>&1

wait $AGG_PID
echo "[$(date +%H:%M:%S)] aggregate done; H5:"
ls -la "$H5"/*.h5 2>/dev/null | head -1

# 3. Eval (fair, canonical slots) + raw switches
echo "[$(date +%H:%M:%S)] eval"
$PY scripts/evaluate.py --luc3d-dir "$OUT" --pred-h5-dir "$H5" --master "$M" \
  --out-csv outputs/PAF_3d_kalman/_eval_keeptrack_liezl.csv \
  --no-sleap-baseline --no-bytetrack --workers 16 > logs/keeptrack_eval.log 2>&1

# 4. Sustained-switch breakdown
PYTHONPATH=. $PY scripts/switch_breakdown.py "$OUT" "$H5" \
  --label keeptrack_liezl --min-run 30 --workers 12 2>&1 | tail -1

echo "=== KEEPTRACK RESULT (multi-animal) ==="
$PY -c "
import pandas as pd
df=pd.read_csv('outputs/PAF_3d_kalman/_eval_keeptrack_liezl.csv');df=df[(df.tracker=='luc3d')&(df.animals>=2)]
print(f'keeptrack Liezl: n={len(df)} raw_switches={int(df.num_switches.sum())} idf1={df.idf1.mean():.4f} idp={df.idp.mean():.4f} idr={df.idr.mean():.4f}')
print('detok_liezl (filter-only): raw_switches=4447 idf1=0.6795 idp=0.8473 idr=0.5880 ; sustained=1590')
"
echo "DONE $(date +%H:%M:%S)"
