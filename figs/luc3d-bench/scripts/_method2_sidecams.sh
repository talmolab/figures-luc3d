#!/usr/bin/env bash
# Method 2: add side/sideL as extra triangulation views to Liezl (eval still on 6 GT cams).
# Test on the diff-7 sessions (where swaps concentrate).
set -uo pipefail
cd /root/vast/eric/luc3d-bench
export PATH="/usr/local/bin:$PATH"
D7=$(cat /tmp/diff7_sessions.txt)
PYSN="sleap_nn_env/bin/python"; PYL="liezl_env/bin/python"; PY=".venv/bin/python"
SIDEDIR=outputs/side_detections
M8=outputs/_sidecam_master.tsv
OUT=outputs/PAF_3d_kalman/sidecam_liezl_results
mkdir -p "$SIDEDIR" logs

echo "[$(date +%H:%M:%S)] 1. filter side/sideL detections (diff-7 sessions)"
PYTHONPATH=. $PYSN - <<PY > logs/m2_sidefilter.log 2>&1
import pandas as pd
from pathlib import Path
from scripts.lib.slap2m_paths import raw_pred_slp
from scripts.sleap_nn.filter_only import filter_one
m=pd.read_csv('outputs/sleap_nn_master_sheet.tsv',sep='\t')
m['session']=m['session'].astype(str)
d7="$D7".split()
for sid in d7:
    row=m[m['session']==sid].iloc[0]; sp=row['session_path']
    for cam in ('side','sideL'):
        src=raw_pred_slp(sp,cam)
        if src is None: print('NO RAW',sid,cam); continue
        dst=Path('$SIDEDIR')/sid/f'{cam}.slp'; dst.parent.mkdir(parents=True,exist_ok=True)
        r=filter_one(src,dst)
        print(sid,cam,'kept',r.get('n_kept'),'max/frame',r.get('max_dets_per_frame'))
PY
echo "  filtered: $(ls $SIDEDIR/*/side.slp 2>/dev/null|wc -l) side, $(ls $SIDEDIR/*/sideL.slp 2>/dev/null|wc -l) sideL"

echo "[$(date +%H:%M:%S)] 2. build 8-cam master"
$PY - <<PY
import pandas as pd
from pathlib import Path
m=pd.read_csv('outputs/sleap_nn_master_sheet.tsv',sep='\t',index_col=0); m['session']=m['session'].astype(str)
for cam in ('side','sideL'):
    m[f'{cam}_raw_pred_slp']=m['session'].map(lambda s: str(Path('$SIDEDIR')/s/f'{cam}.slp') if (Path('$SIDEDIR')/s/f'{cam}.slp').exists() else '')
m.to_csv('$M8',sep='\t')
print('wrote 8-cam master; side cols filled:',int((m['side_raw_pred_slp']!='').sum()))
PY

echo "[$(date +%H:%M:%S)] 3. Liezl 8-cam (corr3d=6) on diff-7"
LIEZL_INPUT_CAMERAS="back,backL,mid,midL,top,topL,side,sideL" \
  $PYL scripts/liezl/run_all_sessions.py --master-sheet "$M8" --out-dir "$OUT" \
  --only-sessions $D7 --workers 10 --corr3d 6 > logs/m2_liezl.log 2>&1

echo "[$(date +%H:%M:%S)] 4. eval on 6 GT cams (vs keeptrack baseline)"
$PY scripts/evaluate.py --luc3d-dir "$OUT" --pred-h5-dir outputs/keeptrack_h5s --master "$M8" \
  --out-csv outputs/PAF_3d_kalman/_eval_sidecam.csv --no-sleap-baseline --no-bytetrack --workers 12 \
  > logs/m2_eval.log 2>&1
echo "=== METHOD2 (side cams) RESULT on diff-7 ==="
$PY -c "
import pandas as pd
sub='$D7'.split()
v=pd.read_csv('outputs/PAF_3d_kalman/_eval_sidecam.csv'); v=v[v.tracker=='luc3d']
b=pd.read_csv('outputs/PAF_3d_kalman/_eval_keeptrack_liezl.csv'); b=b[(b.tracker=='luc3d')&(b.session.astype(str).isin(sub))]
print(f'keeptrack 6-cam (diff-7): switches {int(b.num_switches.sum())} idf1 {b.idf1.mean():.4f} idr {b.idr.mean():.4f}')
print(f'+side/sideL 8-cam(diff-7): switches {int(v.num_switches.sum())} idf1 {v.idf1.mean():.4f} idr {v.idr.mean():.4f}')
"
echo "M2 DONE $(date +%H:%M:%S)"
