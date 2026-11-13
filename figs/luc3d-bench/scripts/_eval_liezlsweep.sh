#!/usr/bin/env bash
# Eval one Liezl-temporal-sweep config dir on multi-animal sessions; print vs baseline.
set -uo pipefail
cd /root/vast/eric/luc3d-bench
DIR="$1"; TAG="$2"
CSV="outputs/PAF_3d_kalman/_liezlsweep/eval_${TAG}.csv"
.venv/bin/python scripts/evaluate.py --luc3d-dir "$DIR" \
  --pred-h5-dir outputs/detections_only_h5s --master outputs/detections_only_master_sheet.tsv \
  --out-csv "$CSV" --no-sleap-baseline --no-bytetrack --workers 16 > /dev/null 2>&1
.venv/bin/python -c "
import pandas as pd
df=pd.read_csv('$CSV'); df=df[(df.tracker=='luc3d')&(df.animals>=2)]
print(f'$TAG : n={len(df)} switches={int(df.num_switches.sum())} sw/sess={df.num_switches.sum()/len(df):.2f} idf1={df.idf1.mean():.4f} idp={df.idp.mean():.4f} frag={int(df.num_fragmentations.sum())}')
print('baseline detok_liezl (v10 d50 t0.1 c1): switches=4447 sw/sess=17.65 idf1=0.6795 frag=225434')
"
