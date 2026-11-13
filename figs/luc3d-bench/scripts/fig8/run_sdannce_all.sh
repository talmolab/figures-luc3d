#!/bin/bash
# Fig 8d + 8e: LUC3D's cross-view tracker on social-DANNCE BEDDING.
#   base   contiguous frames, clean projected 2D + the human-label frames
#   sweep  the same, with the projected 2D degraded (panel e)
set -u
REPO=/root/vast/eric/luc3d-bench
PY=$REPO/eks_env/bin/python
NFRAMES=${NFRAMES:-9000}
BASE=$REPO/outputs/fig8/sdannce_base
SWEEP=$REPO/outputs/fig8/sdannce_sweep
mkdir -p "$BASE" "$SWEEP"

echo "########## BASE ##########"
for sess in $REPO/sdannce_data/2024_*; do
  name=$(basename "$sess")
  $PY $REPO/scripts/fig8/sdannce_prepare.py --session "$sess" --out-dir "$BASE" \
      --stride 1 --max-frames "$NFRAMES" | grep -E "wrote|calibration check"
  for src in human projected; do
    in="$BASE/${name}_${src}.json"
    [ -f "$in" ] || continue
    [ -f "$BASE/track_${name}_${src}.json" ] && continue
    node $REPO/scripts/fig8/sdannce_track.mjs --input "$in" --out "$BASE/track_${name}_${src}.json"
  done
done
$PY $REPO/scripts/fig8/sdannce_score.py $BASE/track_*.json --out $REPO/outputs/fig8/sdannce_identity.csv

echo "########## SWEEP (panel e) ##########"
SWEEP_FRAMES=${SWEEP_FRAMES:-3000}
for sess in $REPO/sdannce_data/2024_05_07_F1_F3 $REPO/sdannce_data/2024_05_07_F6_F2; do
  name=$(basename "$sess")
  for cfg in "noise:0.5" "noise:1" "noise:2" "noise:5" "noise:10" "noise:20" "noise:40" \
             "drop:0.1" "drop:0.25" "drop:0.5" "drop:0.75" \
             "occ:0.1" "occ:0.25" "occ:0.5"; do
    kind=${cfg%%:*}; val=${cfg##*:}
    tag="${kind}${val}"
    [ -f "$SWEEP/track_${name}_projected_${tag}.json" ] && continue
    case $kind in
      noise) flags="--noise-px $val" ;;
      drop)  flags="--dropout $val" ;;
      occ)   flags="--occlude-views $val" ;;
    esac
    $PY $REPO/scripts/fig8/sdannce_prepare.py --session "$sess" --out-dir "$SWEEP" \
        --stride 1 --max-frames "$SWEEP_FRAMES" --tag "$tag" $flags > /dev/null
    node $REPO/scripts/fig8/sdannce_track.mjs \
        --input "$SWEEP/${name}_projected_${tag}.json" \
        --out "$SWEEP/track_${name}_projected_${tag}.json" 2>&1 | tail -1
  done
done
$PY $REPO/scripts/fig8/sdannce_score.py $SWEEP/track_*.json --out $REPO/outputs/fig8/sdannce_degradation.csv
echo ALLDONE
