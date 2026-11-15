#!/usr/bin/env bash
# Guided Liezl temporal-param sweep (round 2): gradient says go TIGHTER / weight 3D more.
set -uo pipefail
cd /root/vast/eric/luc3d-bench
export PATH="/usr/local/bin:$PATH"
SESS=$(cat /tmp/multi_sessions.txt)
PY_L=liezl_env/bin/python
PY=.venv/bin/python
M=outputs/detections_only_master_sheet.tsv
H5=outputs/detections_only_h5s
BASE=outputs/PAF_3d_kalman/_liezlsweep
mkdir -p "$BASE" logs

# tag  vel  dist  tpen  corr2d corr3d
CONFIGS=(
  "B_tight   5  30  0.2  1.0 1.0"
  "C_3dheavy 10 50  0.1  1.0 3.0"
  "D_tight3d 5  30  0.2  1.0 3.0"
)
echo "tag,switches,sw_per,idf1,idp,frag" > /tmp/liezl_sweep2.csv
for cfg in "${CONFIGS[@]}"; do
  read -r TAG VEL DIST TPEN C2 C3 <<< "$cfg"
  OUT="$BASE/$TAG"
  CSV="$BASE/eval_$TAG.csv"
  echo "[$(date +%H:%M:%S)] running $TAG (vel$VEL dist$DIST tpen$TPEN c3=$C3)"
  $PY_L scripts/liezl/run_all_sessions.py --master-sheet "$M" --out-dir "$OUT" \
    --only-sessions $SESS --workers 12 \
    --vel-thresh "$VEL" --dist-thresh "$DIST" --time-penalty "$TPEN" \
    --corr2d "$C2" --corr3d "$C3" > "logs/liezlsweep_$TAG.log" 2>&1
  $PY scripts/evaluate.py --luc3d-dir "$OUT" --pred-h5-dir "$H5" --master "$M" \
    --out-csv "$CSV" --no-sleap-baseline --no-bytetrack --workers 16 > /dev/null 2>&1
  $PY -c "
import pandas as pd
df=pd.read_csv('$CSV');df=df[(df.tracker=='luc3d')&(df.animals>=2)]
print(f'$TAG,{int(df.num_switches.sum())},{df.num_switches.sum()/len(df):.2f},{df.idf1.mean():.4f},{df.idp.mean():.4f},{int(df.num_fragmentations.sum())}')
" | tee -a /tmp/liezl_sweep2.csv
done
echo "=== SWEEP2 DONE ==="
echo "baseline,4447,17.65,0.6795,0.8473,225434" >> /tmp/liezl_sweep2.csv
column -t -s, /tmp/liezl_sweep2.csv
