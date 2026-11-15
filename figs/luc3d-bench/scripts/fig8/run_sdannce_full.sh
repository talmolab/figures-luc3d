#!/bin/bash
# Fig 8d at full session length: every BEDDING dyad session, ALL 90,000 frames,
# contiguous, identity-stripped. One process per session (each is single-threaded).
set -u
REPO=/root/vast/eric/luc3d-bench
PY=$REPO/eks_env/bin/python
OUT=$REPO/outputs/fig8/sdannce_full
mkdir -p "$OUT" "$REPO/logs/fig8_full"

for sess in $REPO/sdannce_data/2024_*; do
(
  name=$(basename "$sess")
  log=$REPO/logs/fig8_full/$name.log
  {
    echo "[$(date +%T)] prepare $name"
    if [ ! -f "$OUT/${name}_projected.bin" ]; then
      $PY $REPO/scripts/fig8/sdannce_prepare.py --session "$sess" --out-dir "$OUT" \
          --stride 1 --binary
    fi
    for src in human projected; do
      [ -f "$OUT/${name}_${src}.json" ] || continue
      [ -f "$OUT/track_${name}_${src}.json" ] && continue
      echo "[$(date +%T)] track $name/$src"
      node --max-old-space-size=16384 $REPO/scripts/fig8/sdannce_track.mjs \
           --input "$OUT/${name}_${src}.json" --out "$OUT/track_${name}_${src}.json"
    done
    echo "[$(date +%T)] done $name"
  } > "$log" 2>&1
) &
done
wait
$PY $REPO/scripts/fig8/sdannce_score.py $OUT/track_*.json --out $REPO/outputs/fig8/sdannce_identity_full.csv
echo FULLDONE
