#!/usr/bin/env bash
# Method 1: larger sleap-nn tracking window (30 vs 15) -> recover more across gaps.
# Full pipeline on the 42 multi-animal sessions, champion Liezl config (corr3d=6).
set -uo pipefail
cd /root/vast/eric/luc3d-bench
export PATH="/usr/local/bin:$PATH"
SESS=$(cat /tmp/multi_sessions.txt)
PYSN="sleap_nn_env/bin/python"; PYL="liezl_env/bin/python"; PY=".venv/bin/python"
W=30
RAWM=outputs/_multi_raw_master.tsv
W30M=outputs/_multi_w30_master.tsv
RETRACK=outputs/sleap_nn_retracked_w30
H5=outputs/keeptrack_w30_h5s
OUT=outputs/PAF_3d_kalman/keeptrack_w30_3d6_results
mkdir -p logs

# raw-predictions multi-animal master for the retrack source
$PY - <<'PY'
import pandas as pd
m=pd.read_csv('outputs/predictions_master_sheet.tsv',sep='\t',index_col=0)
m['session']=m['session'].astype(str)
mm=pd.read_csv('outputs/_multi_w30_master.tsv',sep='\t')
keep=set(mm['session'].astype(str))
m[m['session'].isin(keep)].reset_index(drop=True).to_csv('outputs/_multi_raw_master.tsv',sep='\t')
PY

echo "[$(date +%H:%M:%S)] 1. retrack window=$W (42 sessions x6 cams)"
PYTHONPATH=. $PYSN scripts/sleap_nn/retrack_all.py --master "$RAWM" --out-dir "$RETRACK" \
  --window-size $W --workers 14 > logs/m1ws_retrack.log 2>&1
echo "[$(date +%H:%M:%S)] retrack done: $(ls $RETRACK/*/back.slp 2>/dev/null|wc -l) sessions"

echo "[$(date +%H:%M:%S)] 2. aggregate -> $H5"
PYTHONPATH=. $PYSN scripts/sleap_nn/aggregate_detections_h5.py \
  --detections-dir "$RETRACK" --out-dir "$H5" --master "$W30M" --n-dets 5 \
  --log logs/m1ws_aggregate.log > logs/m1ws_aggregate.stdout 2>&1

echo "[$(date +%H:%M:%S)] 3. Liezl corr3d=6"
$PYL scripts/liezl/run_all_sessions.py --master-sheet "$W30M" --out-dir "$OUT" \
  --only-sessions $SESS --workers 12 --corr3d 6 > logs/m1ws_liezl.log 2>&1

echo "[$(date +%H:%M:%S)] 4. eval + sustained"
$PY scripts/evaluate.py --luc3d-dir "$OUT" --pred-h5-dir "$H5" --master "$W30M" \
  --out-csv outputs/PAF_3d_kalman/_eval_keeptrack_w30_3d6.csv --no-sleap-baseline --no-bytetrack --workers 16 \
  > logs/m1ws_eval.log 2>&1
PYTHONPATH=. $PY scripts/switch_breakdown.py "$OUT" "$H5" --label keeptrack_w30_3d6 --min-run 30 --workers 12 2>&1 | tail -1

echo "=== METHOD1-WS RESULT ==="
$PY -c "
import pandas as pd
df=pd.read_csv('outputs/PAF_3d_kalman/_eval_keeptrack_w30_3d6.csv');df=df[(df.tracker=='luc3d')&(df.animals>=2)]
print(f'keeptrack window=30 + corr3d6: raw={int(df.num_switches.sum())} idf1={df.idf1.mean():.4f} idr={df.idr.mean():.4f}')
print('vs window=15 + corr3d6      : raw=3594 sustained=1005 idf1=0.8327')
"
echo "M1WS DONE $(date +%H:%M:%S)"
