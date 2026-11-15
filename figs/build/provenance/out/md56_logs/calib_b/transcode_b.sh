#!/usr/bin/env bash
# Supp Fig 8 (repo fig7) "b" rerun -- STAGE 1: build the ONE set of input bytes both tools read.
#
# Why: the published figure's 18-camera and three held-out SLAP-2M inputs were H.264
# transcodes written to /tmp scratchpads that no longer exist. Headless Chromium has no
# HEVC decoder, so calibrat3 cannot read the originals; Anipose CAN, but then the two
# tools would read different bytes. So every HEVC original is transcoded ONCE, here, on
# /root/vast, with the parameters recorded and every output hashed. Native-H.264 originals
# (SLAP-2M 2022-10-07, all Mouse-Dyad-10M) are COPIED, not transcoded, and hashed against
# their source, so those two inputs are byte-identical to what the published figure read.
#
# Not timed; run before the quiet sitting. ~10-15 min (parallel x264, ~2-3 GB written).
#
#   OUT=/root/vast/eric/calib-bench-b bash transcode_b.sh
set -euo pipefail
OUT="${OUT:-/root/vast/eric/calib-bench-b}"
IN="$OUT/inputs"
FF=/root/vast/eric/luc3d-bench/liezl_env/lib/python3.12/site-packages/imageio_ffmpeg/binaries/ffmpeg-linux-x86_64-v7.0.2
SLAP=/root/talmolab-smb/eric/slap_2m
MD10=/root/vast/eric/calibration_test/Eric_calibration_videos
C18SRC="/root/vast/eric/calibration_test/2024-12-06_19-12-39calibration"
# IDENTICAL to figs/fig7_calib18_transcode.sh / calib-bench-413/transcode.sh, plus pinned
# x264 threads so the bytes do not depend on the core count of the box that made them.
X264=(-c:v libx264 -preset slow -crf 12 -pix_fmt yuv420p -g 20 -bf 0 -threads 8 -movflags +faststart)
mkdir -p "$IN" "$OUT/logs"
PROV="$OUT/logs/transcode_provenance.tsv"
echo -e "rig\trecording\tcamera\tsource\tsource_sha256\toutput\toutput_sha256\tmode" > "$PROV"
"$FF" -version | head -1 > "$OUT/logs/ffmpeg_version.txt"
printf '%q ' "${X264[@]}" > "$OUT/logs/x264_args.txt"

jobs_running() { jobs -rp | wc -l; }
throttle() { while [ "$(jobs_running)" -ge "${PAR:-12}" ]; do sleep 1; done; }

# --- SLAP-2M: four distinct recordings (README "Fig 7": every other folder on a date is a byte copy)
declare -A SLAPDATE=([10072022120554]=2022-10-07 [10192022174304]=2022-10-19 [10212022155401]=2022-10-21 [10302022010116]=2022-10-30)
for sid in 10072022120554 10192022174304 10212022155401 10302022010116; do
  dst="$IN/slap8/$sid"; mkdir -p "$dst"
  for cam in back backL mid midL side sideL top topL; do
    src="$SLAP/${SLAPDATE[$sid]}/$sid/$cam/calibration_images/$sid-$cam-calibration.mp4"
    out="$dst/$sid-$cam-calibration.mp4"
    # `ffmpeg -i` with no output always exits 1; under set -e + pipefail that killed the script
    # silently (2026-10-19), so the probe's own status is ignored -- an empty codec still fails below.
    codec=$( { "$FF" -hide_banner -i "$src" 2>&1 || true; } | grep -m1 'Video:' | sed -E 's/.*Video: ([a-z0-9]+).*/\1/')
    [ -n "$codec" ] || { echo "no video stream in $src" >&2; exit 1; }
    if [ "$codec" = "h264" ]; then
      ( cp "$src" "$out"; echo -e "slap8\t$sid\t$cam\t$src\t$(sha256sum "$src"|cut -c1-64)\t$out\t$(sha256sum "$out"|cut -c1-64)\tcopy" >> "$PROV" ) &
    else
      ( "$FF" -y -loglevel error -i "$src" "${X264[@]}" "$out"
        echo -e "slap8\t$sid\t$cam\t$src\t$(sha256sum "$src"|cut -c1-64)\t$out\t$(sha256sum "$out"|cut -c1-64)\ttranscode:$codec" >> "$PROV" ) &
    fi
    throttle
  done
done

# --- Mouse-Dyad-10M: native H.264, five recordings, copied with their board/metadata files
for sdir in "$MD10"/*/; do
  sid=$(basename "$sdir"); dst="$IN/md10/$sid"; mkdir -p "$dst"
  for f in "$sdir"*; do
    ( cp "$f" "$dst/"; case "$f" in *.mp4)
        echo -e "md10\t$sid\t$(basename "$f" | cut -d- -f1)\t$f\t$(sha256sum "$f"|cut -c1-64)\t$dst/$(basename "$f")\t$(sha256sum "$dst/$(basename "$f")"|cut -c1-64)\tcopy" >> "$PROV";; esac ) &
    throttle
  done
done

# --- 18-camera rig: HEVC -> H.264 (the third transcode of this recording; the first is lost,
#     the second is /root/vast/eric/calib18_h264 from 2026-09-29 -- left untouched)
for d in "$C18SRC"/Camera*; do
  c=$(basename "$d"); mkdir -p "$IN/calib18/$c/calibration_images"
  src=$(ls "$d"/calibration_images/*.mp4 | head -1); out="$IN/calib18/$c/calibration_images/calib.mp4"
  ( "$FF" -y -loglevel error -i "$src" "${X264[@]}" "$out"
    echo -e "calib18\tcalib18\t$c\t$src\t$(sha256sum "$src"|cut -c1-64)\t$out\t$(sha256sum "$out"|cut -c1-64)\ttranscode:hevc" >> "$PROV" ) &
  throttle
done
printf 'board_x = 8\nboard_y = 11\nsquare_length = 24.0\nmarker_length = 18.75\nmarker_bits = 4\ndict_size = 1000\n' > "$IN/calib18/board.toml"
wait
# Every copy must equal its source.
awk -F'\t' 'NR>1 && $8=="copy" && $5!=$7 {print "COPY MISMATCH", $0; bad=1} END{exit bad}' "$PROV"
echo "inputs ready under $IN; provenance $PROV ($(($(wc -l < "$PROV")-1)) videos)"
