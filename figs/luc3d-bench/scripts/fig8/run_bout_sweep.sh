#!/bin/bash
# Fig 8e: degrade the 2D and re-run, ON THE CLOSE-INTERACTION BOUTS ONLY.
# The whole-session sweep was flat at 1.000 out to 40 px because the rats are
# >200 mm apart in most frames; the bouts are the regime that can actually fail.
set -u
REPO=/root/vast/eric/luc3d-bench
PY=$REPO/eks_env/bin/python
OUT=$REPO/outputs/fig8/sdannce_bout_sweep
mkdir -p "$OUT"
for sess in $REPO/sdannce_data/2024_*; do
(
  name=$(basename "$sess")
  for cfg in "noise:2" "noise:5" "noise:10" "noise:20" "noise:40" "noise:80" "noise:160" \
             "drop:0.25" "drop:0.5" "drop:0.75" "drop:0.9" \
             "occ:0.25" "occ:0.5" "occ:0.75"; do
    kind=${cfg%%:*}; val=${cfg##*:}; tag="${kind}${val}"
    [ -f "$OUT/track_${name}_projected_${tag}.json" ] && continue
    case $kind in
      noise) flags="--noise-px $val" ;;
      drop)  flags="--dropout $val" ;;
      occ)   flags="--occlude-views $val" ;;
    esac
    $PY $REPO/scripts/fig8/sdannce_prepare.py --session "$sess" --out-dir "$OUT" \
        --stride 1 --bouts 50 --binary --tag "$tag" $flags > /dev/null 2>&1
    node $REPO/scripts/fig8/sdannce_track.mjs \
        --input "$OUT/${name}_projected_${tag}.json" \
        --out "$OUT/track_${name}_projected_${tag}.json" --per-bout > /dev/null 2>&1
  done
) &
done
wait
$PY $REPO/scripts/fig8/sdannce_score.py $OUT/track_*.json --out $REPO/outputs/fig8/sdannce_bout_degradation.csv
echo BOUTSWEEPDONE
