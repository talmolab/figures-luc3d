#!/usr/bin/env bash
# Overnight queue for eric-mimic2 (2026-10-15). Accuracy first, timing last and alone.
# Every step checks its own result; a failed step stops its chain (and the timing phase).
# Status lines go to figs/figures/drafts/RUN-STATUS-2026-10-15.md (shared with the other job).
set -uo pipefail
R=/root/vast/eric/sleap-3d-gui/scratch/repos/lucid
cd $R
L=figs/out/md56_logs; O=figs/out
ST=figs/figures/drafts/RUN-STATUS-2026-10-15.md
SN02=/root/vast/eric/luc3d-bench/sleap_nn_env/bin/python                  # sleap-nn 0.2.0 (uv, py 3.11.14)
LZ=/root/vast/eric/luc3d-bench/liezl_env/bin/python                       # scoring (motmetrics 1.4.0)
CV2H=/tmp/claude-0/-root-vast-eric-sleap-3d-gui-scratch-repos-lucid/44adbb8d-7377-41ea-ab2c-5dae7a320d17/scratchpad/cv2h
AP=/root/vast/eric/luc3d-bench/anipose072_cv413_env/bin/python             # Fig 2G timing
MAIN=/root/vast/eric/lucid-main-md56
export PATH=/root/vast/eric/node26_env/bin:$PATH
st() { echo "- $(date -u +'%F %T') UTC $(hostname -s) [main box] $*" | tee -a $ST; }
fail() { st "FAILED: $*"; echo "OVERNIGHT FAILED: $*"; exit 1; }
[ -e $ST ] || printf "# Run status 2026-10-15 (both machines append here)\n\n" > $ST

# ---------------------------------------------------------------------------------------
# A2 (resumed). B7 steps 4-6: S6SMALL's SLAP-2M ByteTrack re-stitch failed at 12:27 because it
# ran under liezl_env (no `supervision`); the original stage ran under eks_env (supervision 0.30.0).
# Steps 1-3 (Mouse-Dyad) succeeded and are not rerun.
# OVERNIGHT4 (2026-10-15 ~15:50 UTC): trimmed per Eric. B1 scoring + C1 (Fig 2G) only. B7 already done in OVERNIGHT3; C2 dropped.
st "waiting for B1(mb) (run_B1_mainbox_mb.out)"
until grep -qE "B1MB (DONE|FAILED)" $L/run_B1_mainbox_mb.out 2>/dev/null; do sleep 60; done
grep -q "B1MB DONE" $L/run_B1_mainbox_mb.out || fail "B1(mb) failed: $(tail -2 $L/run_B1_mainbox_mb.out | tr '\n' ' ')"
env SCORE_NOTAIL=1 PYTHONPATH=$CV2H $LZ figs/fig6_sleap_scoped.py \
    --retracked /root/vast/eric/luc3d-bench/outputs/bmimica/retracked_max2_notail_mb \
    --out fig7_sleap_max2_sleapnt.json --workers 12 > $L/SLEAPNT_md_score.log 2>&1 || fail "B1 scoring"
st "B1 scored: wrote $O/fig7_sleap_max2_sleapnt.json (from retracked_max2_notail_mb). Not installed into any figure yet."

# ---------------------------------------------------------------------------------------
# C. Timing, alone. Nothing of ours may run; wait for a real idle machine.
# ---------------------------------------------------------------------------------------
idle_wait() {   # 1-min load < 2.0 for 5 consecutive minutes; give up after 6 h
  local ok=0
  for i in $(seq 1 1440); do
    if pgrep -af "fig3_|fig6_|fig10_|sleap_nn|bfs|find /|chrome|ffmpeg" | grep -v pgrep | grep -qv "run_OVERNIGHT"; then ok=0;
    elif awk '{exit !($1 < 2.0)}' /proc/loadavg; then ok=$((ok+1)); else ok=0; fi
    [ $ok -ge 20 ] && return 0
    sleep 15
  done
  return 1
}
loadlog() { while :; do p=$(pgrep -f "$1" | head -1); th=$([ -n "$p" ] && ls /proc/$p/task 2>/dev/null | wc -l || echo 0)
            echo "$(date -u +%T) $(cut -d' ' -f1-3 /proc/loadavg) threads=$th"; sleep 5; done; }
export NUMBA_NUM_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 UV_THREADPOOL_SIZE=1

# C1. Fig 2G: every solver on the same 200,010 keypoints. New immutable deposit fig2g_retime_all_md56C1.json.
idle_wait || fail "C1: machine never idle (load < 2 for 5 min) within 6 h -- timing NOT run"
st "C1 start (Fig 2G), load $(cut -d' ' -f1-3 /proc/loadavg)"
ln -sfn fig4_input_md56.json $O/fig4_input_md56C1.json
[ -e $O/fig2g_retime_all_md56C1.json ] && fail "C1 deposit exists"
loadlog "fig2g_retime" > $L/C1_load.log & LP=$!
FIG2_SUFFIX=md56C1 FIG2G_FRAMES=13334 POSE_DIR=$MAIN/pose NODE=/root/vast/eric/node26_env/bin/node \
  $AP figs/fig2g_retime_all.py --rounds 5 > $L/C1_retime.log 2>&1; rc1=$?
FIG2_SUFFIX=md56C1 FIG2G_FRAMES=13334 $AP figs/fig2g_retime_extra.py > $L/C1_extra.log 2>&1; rc2=$?
kill $LP
MAXL=$(awk '{print $2}' $L/C1_load.log | sort -n | tail -1); MAXT=$(sed 's/.*threads=//' $L/C1_load.log | sort -n | tail -1)
[ $rc1 -eq 0 ] && [ $rc2 -eq 0 ] || fail "C1 exit $rc1/$rc2"
$AP - <<PY
import json; p = "$O/fig2g_retime_all_md56C1.json"; d = json.load(open(p))
d["sitting"] = {"id": "C1", "max_1min_load": $MAXL, "max_threads_timing_process": $MAXT, "cores": 64,
  "idle_rule": "1-min load < 2.0 for 5 consecutive minutes, nothing of ours running", "load_log": "md56_logs/C1_load.log",
  "thread_env": "NUMBA/OMP/OPENBLAS/MKL/VECLIB/NUMEXPR_NUM_THREADS=1, UV_THREADPOOL_SIZE=1",
  "code": "LUC3D v1.0.5 (2dc001fe)", "keypoints": "every solver on the same first 13,334 frames x 15 joints = 200,010",
  "valid": $MAXL <= 4.0}
json.dump(d, open(p, "w"), indent=1)
PY
st "C1 DONE: max 1-min load $MAXL, max threads $MAXT -> $O/fig2g_retime_all_md56C1.json $( awk -v m=$MAXL 'BEGIN{if (m>4.0) print "(INVALID: load ceiling 4 exceeded)"}')"
echo "OVERNIGHT4 DONE"; st "OVERNIGHT4 DONE (main box): B1 scored, C1 done. C2 dropped."
