#!/bin/bash
# Fig 8d: prepare -> track -> score, every BEDDING dyad session.
set -u
REPO=/root/vast/eric/luc3d-bench
PY=$REPO/eks_env/bin/python
OUT=$REPO/outputs/fig8/sdannce
STRIDE=${STRIDE:-30}
mkdir -p "$OUT"

for sess in $REPO/sdannce_data/2024_*; do
  name=$(basename "$sess")
  $PY $REPO/scripts/fig8/sdannce_prepare.py --session "$sess" --out-dir "$OUT" --stride "$STRIDE" || continue
  for src in human projected; do
    in="$OUT/${name}_${src}.json"
    [ -f "$in" ] || continue
    node $REPO/scripts/fig8/sdannce_track.mjs --input "$in" --out "$OUT/track_${name}_${src}.json" || echo "  TRACK FAIL $name/$src"
  done
done
$PY $REPO/scripts/fig8/sdannce_score.py $OUT/track_*.json --out $REPO/outputs/fig8/sdannce_identity.csv
