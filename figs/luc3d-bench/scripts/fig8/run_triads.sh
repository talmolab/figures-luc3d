#!/bin/bash
# Fig 8d (real detections): LUC3D's tracker on s-DANNCE's own per-camera COM output.
set -u
REPO=/root/vast/eric/luc3d-bench
PY=$REPO/eks_env/bin/python
OUT=$REPO/outputs/fig8/triads
mkdir -p "$OUT" "$REPO/logs/fig8_triads"
for sess in $REPO/sdannce_triads/2023_*; do
(
  name=$(basename "$sess")
  log=$REPO/logs/fig8_triads/$name.log
  {
    [ -f "$OUT/${name}_com.bin" ] || $PY $REPO/scripts/fig8/triads_prepare.py \
        --session "$sess" --out-dir "$OUT"
    [ -f "$OUT/track_${name}.json" ] || node $REPO/scripts/fig8/sdannce_track.mjs \
        --input "$OUT/${name}_com.json" --out "$OUT/track_${name}.json"
    $PY $REPO/scripts/fig8/triads_adjudicate.py --track "$OUT/track_${name}.json" \
        --prepared "$OUT/${name}_com.json" --session-dir "$sess" \
        --out "$OUT/adjudicate_${name}.csv"
  } > "$log" 2>&1
) &
done
wait
$PY $REPO/scripts/fig8/sdannce_score.py $OUT/track_*.json --out $REPO/outputs/fig8/triads_identity.csv
$PY - <<'PY'
import pandas as pd, glob
fs=sorted(glob.glob('/root/vast/eric/luc3d-bench/outputs/fig8/triads/adjudicate_*.csv'))
if fs:
    d=pd.concat([pd.read_csv(f) for f in fs])
    d.to_csv('/root/vast/eric/luc3d-bench/outputs/fig8/triads_adjudication.csv',index=False)
    print(d.to_string(index=False))
PY
echo TRIADSRUNDONE
