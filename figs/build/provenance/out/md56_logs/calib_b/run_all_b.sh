#!/usr/bin/env bash
# Supp Fig 8 (repo fig7, tex label fig8) -- apples-to-apples rerun, tagged "b".
#
# Writes ONLY: /root/vast/eric/calib-bench-b/**, figs/out/fig7_calibration_b.json,
# figs/figures/fig7b/**, figs/data/fig7b/**. Nothing the published figure reads is touched.
#
# What is equalised, and how:
#   inputs    one transcode/copy per recording (transcode_b.sh), sha256 recorded; BOTH tools
#             read the same files (spec.json `cameras` for Anipose, `session_dir` for calibrat3)
#   OpenCV    4.13.0 on both sides (aniposelib413_env asserted; calibrat3 vendors OpenCV.js 4.13.0)
#   scoring   aniposelib triangulation + reprojection (figs/fig7_calib_score.py, unchanged) on
#             Anipose's own detections from the SAME run whose time is reported; every arm on
#             one shared observation set; calibrat3's corners kept as the secondary set
#   timing    ONE quiet sitting, back to back, same cpuset (taskset -c $CPUS) for both tools,
#             each tool sizing its own parallelism to that budget (Anipose: min(cams,N)
#             processes x max(1,N/cams) OpenCV threads; calibrat3: navigator.hardwareConcurrency=N
#             -> min(12,N-2) detect workers); REPS repeats, order alternated per rep; a quiet
#             gate before every run; load inside/outside the cpuset sampled every second
#   Anipose   stochastic solver: R=5 solver repeats per recording; PRE-REGISTERED arm rule
#             below; divergence count deposited
#
#   bash run_all_b.sh prep        # transcode + specs + env + page-cache warm      (~15 min, not timed)
#   bash run_all_b.sh sitting     # the timed sitting -- NEEDS A QUIET MACHINE    (~1.5-2 h at REPS=3)
#   bash run_all_b.sh accuracy    # the 7 non-timing recordings, same cpuset      (~1.2 h, quiet preferred)
#   bash run_all_b.sh controls    # repeats, equal frames, same ref, anipose-on-ours, ablation, held out (~1.5 h, parallel ok)
#   bash run_all_b.sh synth       # synthetic GT, both GT models                  (~30 min)
#   bash run_all_b.sh score       # scoring + deposit + panels                     (~45 min, parallel ok)
set -euo pipefail
STAGE="${1:?stage: prep|sitting|accuracy|controls|synth|score}"
HERE="$(cd "$(dirname "$0")" && pwd)"
FIGS=/root/vast/eric/sleap-3d-gui/scratch/repos/lucid/figs
CAL3=/root/vast/eric/sleap-3d-gui/scratch/repos/calibrat3
BENCH="${BENCH:-/root/vast/eric/calib-bench-b}"
W="$BENCH/work"; IN="$BENCH/inputs"; LOG="$BENCH/logs"
APY=/root/vast/eric/aniposelib413_env/bin/python
NODE_BIN=/root/vast/eric/node26_env/bin           # speed numbers on Node 26 (bench-env-workarounds)
export PATH="$NODE_BIN:$PATH" PLAYWRIGHT_BROWSERS_PATH=/root/vast/eric/pw-browsers
NCORES="${NCORES:-16}"                            # THE thread budget, identical for both tools
CPUS="${CPUS:-0-$((NCORES-1))}"
SERVER_CPU="${SERVER_CPU:-63}"                    # static file server sits OUTSIDE the cpuset
PORT="${PORT:-8093}"; BASE="http://localhost:$PORT"
REPS="${REPS:-3}"
TIMING_KEYS=(md10_20250514_134355 slap_10072022120554 calib18)   # the same three as the published H
mkdir -p "$W" "$LOG"

refcam() { [ "$1" = calib18 ] && echo Camera6 || true; }   # as published: Camera6 on calib18, app default elsewhere
server_up() {
  if ! curl -sf "$BASE/index.html" >/dev/null 2>&1; then
    (cd "$CAL3" && taskset -c "$SERVER_CPU" python3 server.py "$PORT" >> "$LOG/calibrat3_server.log" 2>&1 &)
    for _ in $(seq 30); do curl -sf "$BASE/index.html" >/dev/null 2>&1 && return 0; sleep 1; done
    echo "calibrat3 server did not come up on $PORT" >&2; exit 1
  fi
}

run_anipose() {   # <key> <tag> <outdir>
  local key=$1 tag=$2 od=$3
  CORES=$NCORES OMP_NUM_THREADS=$NCORES OPENBLAS_NUM_THREADS=$NCORES MKL_NUM_THREADS=$NCORES \
    $APY "$HERE/timed_b.py" "$od/timed_anipose_$tag.json" "$CPUS" -- \
    $APY "$HERE/anipose_run_b.py" "$W/$key/spec.json" "$od" "$tag" 2>&1 | tee "$od/anipose_$tag.log"
}
run_c3() {        # <key> <tag> <outdir> [ALL]
  local key=$1 tag=$2 od=$3 all=${4:-} sess ref name
  sess=$($APY -c "import json;print(json.load(open('$W/$key/spec.json'))['session_dir'])")
  ref=$(refcam "$key"); name=calibrat3_800; [ -n "$all" ] && name=calibrat3_all
  local envs=(SESSION="$sess" OUT="$od/${name}_$tag" HW="$NCORES" BASE="$BASE")
  if [ -n "$all" ]; then envs+=(ALL_FRAMES=1); else envs+=(TARGET=800); fi
  [ -n "$ref" ] && envs+=(REF_CAM="$ref")
  server_up
  env "${envs[@]}" $APY "$HERE/timed_b.py" "$od/timed_${name}_$tag.json" "$CPUS" -- \
    node "$HERE/calibrat3_run_b.mjs" 2>&1 | tee "$od/${name}_$tag.log"
}
promote() {       # rep-1 outputs become the recording's accuracy/scoring files (copied, never moved)
  local key=$1 od=$2
  cp "$od/anipose_rows_r1.pkl" "$W/$key/anipose_rows.pkl"
  cp "$od/calibration_anipose_r1.toml" "$W/$key/calibration_anipose.toml"
  cp "$od/anipose_summary_r1.json" "$W/$key/anipose_summary.json"
  for x in toml session.json timing.json; do cp "$od/calibrat3_800_r1.$x" "$W/$key/calibrat3_800.$x"; done
}

case "$STAGE" in
prep)
  $APY -c "import cv2,sys; sys.exit(0 if cv2.__version__=='4.13.0' else 'cv2 '+cv2.__version__+' -- REFUSING')"
  OUT="$BENCH" bash "$HERE/transcode_b.sh" 2>&1 | tee "$LOG/transcode.log"
  $APY "$HERE/make_specs_b.py" "$BENCH"
  $APY - "$LOG/env.json" <<'PY'
import sys, json, subprocess, platform, os, cv2, numpy, scipy
run = lambda *c: subprocess.run(c, capture_output=True, text=True).stdout.strip()
json.dump(dict(host=platform.node(), kernel=platform.release(), nproc=os.cpu_count(),
               cpu=run('sh', '-c', "grep -m1 'model name' /proc/cpuinfo | cut -d: -f2").strip(),
               python=platform.python_version(), cv2=cv2.__version__, numpy=numpy.__version__, scipy=scipy.__version__,
               aniposelib=__import__('importlib.metadata').metadata.version('aniposelib'),  # env has no pip binary (2026-10-19)
               node=run('/root/vast/eric/node26_env/bin/node', '--version'),
               playwright=json.load(open('/root/vast/eric/sleap-3d-gui/scratch/repos/lucid/tests/e2e/node_modules/playwright/package.json'))['version'],
               calibrat3=run('git', '-C', '/root/vast/eric/sleap-3d-gui/scratch/repos/calibrat3', 'rev-parse', 'HEAD'),
               calibrat3_dirty=run('git', '-C', '/root/vast/eric/sleap-3d-gui/scratch/repos/calibrat3', 'status', '--porcelain', '--untracked-files=no'),
               lucid=run('git', '-C', '/root/vast/eric/sleap-3d-gui/scratch/repos/lucid', 'rev-parse', 'HEAD')),
          open(sys.argv[1], 'w'), indent=1)
print(open(sys.argv[1]).read())
PY
  ;;

sitting)
  # Page cache: read every input once so no run pays a cold read the other did not.
  cat "$IN"/slap8/10072022120554/*.mp4 "$IN"/md10/20250514_134355/*.mp4 "$IN"/calib18/*/calibration_images/*.mp4 > /dev/null
  echo "sitting start $(date -Is) cpus=$CPUS reps=$REPS loadavg=$(cut -d' ' -f1-3 /proc/loadavg)" | tee -a "$LOG/sitting.log"
  for rep in $(seq 1 "$REPS"); do
    for key in "${TIMING_KEYS[@]}"; do
      od="$W/$key/timing"; mkdir -p "$od"
      if [ $((rep % 2)) -eq 1 ]; then run_anipose "$key" "r$rep" "$od"; run_c3 "$key" "r$rep" "$od"
      else                            run_c3 "$key" "r$rep" "$od"; run_anipose "$key" "r$rep" "$od"; fi
      echo "rep $rep $key done $(date -Is) loadavg=$(cut -d' ' -f1-3 /proc/loadavg)" | tee -a "$LOG/sitting.log"
    done
  done
  # Secondary condition: calibrat3 on EVERY frame (the frame budget Anipose's detector sees), once.
  [ "${ALLFRAMES:-1}" = 1 ] && for key in "${TIMING_KEYS[@]}"; do run_c3 "$key" r1 "$W/$key/timing" ALL; done
  for key in "${TIMING_KEYS[@]}"; do promote "$key" "$W/$key/timing"; done
  echo "sitting end $(date -Is)" | tee -a "$LOG/sitting.log"
  ;;

accuracy)
  for d in "$W"/slap_* "$W"/md10_*; do
    key=$(basename "$d"); [ -s "$d/scores_anipose.npz" ] && continue
    [[ " ${TIMING_KEYS[*]} " == *" $key "* ]] && continue
    od="$d/timing"; mkdir -p "$od"
    run_anipose "$key" r1 "$od"; run_c3 "$key" r1 "$od"; promote "$key" "$od"
  done
  ;;

controls)
  PAR="${PAR:-8}"
  ctl() {   # one recording's controls; untimed, so they may run in parallel
    local d=$1 key; key=$(basename "$d")
    export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
    $APY "$FIGS/fig7_calib_anipose_repeats.py" "$d" 5 > "$d/anipose_repeats.log" 2>&1
    # PRE-REGISTERED Anipose arm: the end-to-end run, unless its reported error exceeds 10x the
    # median reported error of the 5 solver repeats (= diverged); then the converged repeat with
    # the median reported error. Recorded either way in anipose_arm.json.
    $APY - "$d" <<'PY'
import sys, json, os, statistics
d = sys.argv[1]; e2e = json.load(open(f'{d}/anipose_summary.json'))['error']
reps = json.load(open(f'{d}/anipose_repeats.json')); med = statistics.median(r['reported_error'] for r in reps)
div = [r['repeat'] for r in reps if r['reported_error'] > 10 * med]
conv = sorted((r for r in reps if r['repeat'] not in div), key=lambda r: r['reported_error'])
if e2e <= 10 * med: arm = dict(toml=f'{d}/calibration_anipose.toml', source='end-to-end run')
else: c = conv[len(conv) // 2]; arm = dict(toml=c['toml'], source=f"solver repeat {c['repeat']} (end-to-end run diverged)")
arm.update(end_to_end_error=e2e, repeat_errors=[r['reported_error'] for r in reps], diverged_repeats=div,
           end_to_end_diverged=e2e > 10 * med, rule='end-to-end unless error > 10x median of 5 repeats')
json.dump(arm, open(f'{d}/anipose_arm.json', 'w'), indent=1); print(key := os.path.basename(d), arm['source'])
PY
    # calibrat3's reference camera (given on calib18, app-chosen elsewhere), mapped onto spec names
    ref=$($APY -c "
import json,re
s=json.load(open('$d/calibrat3_800.session.json')); names=[c[0] for c in json.load(open('$d/spec.json'))['cameras']]
v=s['session']['views'][s['referenceView']]['name']
m=[n for n in names if n==v or re.search(r'(^|[-_ ])'+re.escape(n)+r'([-_ .]|$)',v)]
assert len(m)==1,(v,m); print(m[0])")
    $APY "$FIGS/fig7_calib_equal_frames.py" "$d" "$d/calibrat3_800.session.json" "$d/anipose_equal_frames.toml" > "$d/equal_frames.log" 2>&1
    $APY "$FIGS/fig7_calib_reference_cam.py" "$d" "$ref" "$d/anipose_same_ref.toml" > "$d/same_ref.log" 2>&1
    $APY "$FIGS/fig7_calib_reference_cam.py" "$d" "$ref" "$d/anipose_equal_same_ref.toml" --sameframes "$d/calibrat3_800.session.json" > "$d/equal_same_ref.log" 2>&1
    $APY "$FIGS/fig7_calib_anipose_on_ours.py" "$d" "$d/calibrat3_800.session.json" "$d/calibration_anipose_on_ours.toml" > "$d/anipose_on_ours.log" 2>&1 || true
    echo "controls done: $key (calibrat3 reference $ref)"
  }
  export -f ctl; export APY FIGS
  for d in "$W"/slap_* "$W"/md10_* "$W"/calib18; do while [ "$(jobs -rp | wc -l)" -ge "$PAR" ]; do sleep 2; done; ctl "$d" & done; wait
  # calib18 solver ablation (panel G) + held-out frames, on the NEW calibrat3 session
  d="$W/calib18"; mkdir -p "$d/ablation/tomls"; server_up
  (cd "$CAL3" && SESSION="$IN/calib18" SAVED="$d/calibrat3_800.session.json" REF_CAM=Camera6 \
     OUT="$d/ablation/ablation.json" TOML_DIR="$d/ablation/tomls" BASE="$BASE" node "$FIGS/fig7_calib_ablation.mjs") 2>&1 | tee "$d/ablation/ablation.log"
  ;;

synth)
  mkdir -p "$BENCH/pyextra"; ln -sfn /root/vast/eric/luc3d-bench/liezl_env/lib/python3.12/site-packages/imageio_ffmpeg "$BENCH/pyextra/imageio_ffmpeg"
  for gt in full anipose; do
    sd="$IN/synth_$gt"; wd="$W/synth_$gt"; mkdir -p "$wd"
    [ -s "$sd/calibration_gt.toml" ] || PYTHONPATH="$BENCH/pyextra" $APY "$HERE/make_synthetic_session_b.py" \
        --out "$sd" --cams 8 --frames 1800 --seed 0 --gt-model "$gt" 2>&1 | tail -2 | tee "$wd/make.log"
    $APY - "$sd" "$wd/spec.json" <<'PY'
import sys, json, glob, os
sd, out = sys.argv[1:3]
order = ["back", "mid", "side", "top", "cam5", "cam6", "cam7", "cam8"]
cams = [[c, f'{sd}/{c}/calibration_images/calib.mp4'] for c in order if os.path.isdir(f'{sd}/{c}')]
json.dump(dict(cameras=cams, board=dict(board_x=8, board_y=11, square_length=24.0, marker_length=18.75,
               marker_bits=4, dict_size=1000), size=[1280, 1024], session_dir=sd, date=None,
               note=f'synthetic', tuning=False), open(out, 'w'), indent=1)
PY
    CORES=$NCORES taskset -c "$CPUS" $APY "$HERE/anipose_run_b.py" "$wd/spec.json" "$wd" > "$wd/anipose.log" 2>&1
    server_up
    SESSION="$sd" OUT="$wd/calibrat3_800" HW=$NCORES TARGET=800 BASE="$BASE" taskset -c "$CPUS" node "$HERE/calibrat3_run_b.mjs" > "$wd/calibrat3.log" 2>&1
  done
  ;;

score)
  PAR="${PAR:-6}"
  sc() {
    local d=$1 arm; arm=$($APY -c "import json;print(json.load(open('$d/anipose_arm.json'))['toml'])")
    # main set (panels A-F): ONLY the two arms, so no third calibration can shrink the shared set
    $APY "$FIGS/fig7_calib_score.py" "$d" "anipose=$d/anipose_rows.pkl" "calibrat3=$d/calibrat3_800.session.json" -- \
        "calibrat3_800=$d/calibrat3_800.toml" "anipose=$arm" > "$d/score.log" 2>&1
    # controls: own directory (score.py writes scores.json/npz next to spec.json)
    mkdir -p "$d/controls"; cp "$d/spec.json" "$d/controls/"; ln -sf "$d/anipose_rows.pkl" "$d/controls/anipose_rows.pkl"
    args=("calibrat3_800=$d/calibrat3_800.toml" "anipose=$arm")
    for t in anipose_equal_frames anipose_same_ref anipose_equal_same_ref; do [ -s "$d/$t.toml" ] && args+=("$t=$d/$t.toml"); done
    [ -s "$d/calibration_anipose_on_ours.toml" ] && args+=("anipose_on_ours=$d/calibration_anipose_on_ours.toml")
    $APY "$FIGS/fig7_calib_score.py" "$d/controls" "anipose=$d/anipose_rows.pkl" "calibrat3=$d/calibrat3_800.session.json" -- "${args[@]}" > "$d/controls/score.log" 2>&1
    echo "scored $(basename "$d")"
  }
  export -f sc; export APY FIGS
  export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
  for d in "$W"/slap_* "$W"/md10_* "$W"/calib18; do while [ "$(jobs -rp | wc -l)" -ge "$PAR" ]; do sleep 2; done; sc "$d" & done; wait
  d="$W/calib18"; mkdir -p "$d/ablation"; cp "$d/spec.json" "$d/ablation/"; ln -sf "$d/anipose_rows.pkl" "$d/ablation/anipose_rows.pkl"
  arm=$($APY -c "import json;print(json.load(open('$d/anipose_arm.json'))['toml'])")
  args=("anipose=$arm" "calibrat3_800=$d/calibrat3_800.toml"); for f in "$d"/ablation/tomls/*.toml; do args+=("$(basename "$f" .toml)=$f"); done
  $APY "$FIGS/fig7_calib_score.py" "$d/ablation" "anipose=$d/anipose_rows.pkl" -- "${args[@]}" > "$d/ablation/score.log" 2>&1
  $APY "$FIGS/fig7_calib_heldout.py" "$d" "$d/calibrat3_800.session.json" -- "${args[@]}" > "$d/heldout.log" 2>&1
  $APY "$HERE/deposit_b.py" "$BENCH" "$FIGS/out/fig7_calibration_b.json"
  cd "$FIGS/.." && for p in figs/panels/fig7_0*.py; do FIG_VARIANT=b FIG_TAG=b python3 "$p"; done && FIG_VARIANT=b FIG_TAG=b python3 figs/assemble.py 7
  ;;
*) echo "unknown stage $STAGE" >&2; exit 2;;
esac
