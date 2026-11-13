#!/usr/bin/env bash
# md56 phase Q (2026-10-14): every TIMED number, on a quiet machine, against ONE pinned code
# version -- origin/main 2dc001fe at /root/vast/eric/lucid-main-md56 -- under Node 26.
#   Q1 Fig 2F/G Anipose arm on all 56 sessions, aniposelib 0.7.2 + OpenCV 4.13.0
#      (anipose072_cv413_env; the base run had OpenCV 5.0.0)
#   Q2 Fig 2G same-sitting retime, 5 rounds, best round (Methods)
#   Q3 Fig 3 head-to-head: BOTH arms re-run on main for all 98 sessions in one pass
#      (seconds/frame for Fig 3F under identical conditions; outputs shown identical to
#      the cached ones), all scores recomputed; then quality, fresh anchor,
#      frame-matched, exhaustive_bmimica
#   Q4 Fig 3F LUC3D seconds/frame (fig3_scale_runtime) on main
# Waits for phases U, F2a and T so nothing else is running.
set -uo pipefail
cd /root/vast/eric/sleap-3d-gui/scratch/repos/lucid
L=figs/out/md56_logs
MAIN=/root/vast/eric/lucid-main-md56
export PATH=/root/vast/eric/node26_env/bin:$PATH NODE=/root/vast/eric/node26_env/bin/node
export FIG2_SUFFIX=md56 POSE_DIR=$MAIN/pose NUMBA_NUM_THREADS=1 SCORE_NOTAIL=1 FIG3_VARIANT=md56
export PYTHONPATH=/tmp/claude-0/-root-vast-eric-sleap-3d-gui-scratch-repos-lucid/44adbb8d-7377-41ea-ab2c-5dae7a320d17/scratchpad/cv2h
PY=/root/vast/eric/luc3d-bench/liezl_env/bin/python
AP=/root/vast/eric/luc3d-bench/anipose072_cv413_env/bin/python
t() { echo "[$(date +%T)] $* (load $(cut -d' ' -f1 /proc/loadavg))"; }
for f in run_phaseU.out:"PHASE U DONE" run_F2a.out:"F2a DONE" run_T.out:"PHASE T DONE" run_T2.out:"PHASE T2 DONE"; do  # (run_S on ericmimicc does not load this box)
  until grep -q "${f#*:}" $L/${f%%:*} 2>/dev/null; do sleep 60; done
done
t "all prior phases done; pose $(git -C $MAIN rev-parse --short HEAD), node $(node --version)"

# Q1 (the Anipose accuracy arm) runs early, in run_Q1.sh: Fig 2G reads its timings only from
# the Q2 same-sitting retime, so Q1 needs no quiet machine. Q waits for it to finish.
until grep -q "Q1 anipose exit" $L/run_Q1.out 2>/dev/null; do sleep 60; done
env -u PYTHONPATH $AP figs/fig2g_retime_all.py --rounds 5 > $L/Q2_retime.log 2>&1; t "Q2 retime exit=$?"
env -u PYTHONPATH $AP figs/fig2_anipose_optim_retime.py > $L/Q2_optim_retime.log 2>&1; t "Q2 optim retime exit=$?"

# Q3 -- exhaustive on main for every session (timing), scores recomputed from it
C=figs/out/tmp/headtohead_md56
# Both arms re-run on main in this one pass, so Fig 3F compares them under identical
# conditions (same sessions, same detections, same 16-job load). Greedy outputs are
# bit-identical to the cached ones (gated on 20250827_141755); only the timing is new.
find $C \( -name exhaustive.json -o -name greedy.json \) -delete
find $C \( -name score.json -o -name quality.json -o -name rescore.json -o -name 'score_framematched_*.json' \) -delete
find figs/out/tmp/headtohead_var_md56 -name quality.json -delete
LUCID_ROOT=$MAIN HH_JOBS=16 $PY figs/fig3_headtohead.py > $L/Q3_headtohead.log 2>&1
t "Q3 headtohead exit=$? failed=$(grep -c FAILED $L/Q3_headtohead.log)"
echo "exhaustive cells tail-free: $(grep -l '"excludeTail": true' -r $C --include=exhaustive.json | wc -l) of $(find $C -name exhaustive.json | wc -l)"
$PY figs/fig3_quality.py > $L/Q3_quality.log 2>&1; t "Q3 quality exit=$?"
$PY figs/fig3_hh_freshanchor.py --run --jobs 16 > $L/Q3_freshanchor.log 2>&1; t "Q3 freshanchor exit=$?"
FM_JOBS=12 $PY figs/fig3_frame_matched.py > $L/Q3_frame_matched.log 2>&1; t "Q3 frame_matched exit=$?"
$PY figs/fig3_exhaustive_bmimica.py > $L/Q3_exh_bm.log 2>&1; t "Q3 exhaustive_bmimica exit=$?"

# Q4 -- LUC3D seconds/frame on main
LUCID_ROOT=$MAIN $PY figs/fig3_scale_runtime.py > $L/Q4_scale_runtime.log 2>&1; t "Q4 scale_runtime exit=$?"
ls -la figs/out/*md56*.json
t "PHASE Q DONE"
