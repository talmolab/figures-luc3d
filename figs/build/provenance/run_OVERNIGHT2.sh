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
# A1. B2: SLEAP tail-free re-track, SLAP-2M (Supp 6B-D). Gate (uv env reproduces sleap_cap/
#     exactly) PASSED at ~06:33 UTC: 6/6 cameras identical (run_SLAPNT_ctrl.out).
# ---------------------------------------------------------------------------------------
# B2 sleapsrc + src check (PASS 444 files, 07:07) and the sleap stage were launched by
# run_OVERNIGHT.sh; its parent was stopped at 07:19 (to take B1 back from the other machine),
# the sleap stage itself kept running. Wait for it, then continue exactly as before.
FV="FAIR_VARIANT=bvis5 FAIR_SLEAP_NOTAIL=1 FAIR_SLEAP_PY=$SN02"
st "OVERNIGHT2 start: waiting for the running B2 sleap stage"
until grep -q "^\[sleap\] done:" $L/SLAPNT_track.log 2>/dev/null && ! pgrep -f "fig6_slap2m_fair.py --stage sleap" >/dev/null; do sleep 60; done
grep -q "0 failed" $L/SLAPNT_track.log || fail "B2 sleap had failures: $(tail -1 $L/SLAPNT_track.log)"
env $FV $SN02 figs/fig6_slap2m_fair.py --stage verify > $L/SLAPNT_verify.log 2>&1 || fail "B2 verify: $(tail -1 $L/SLAPNT_verify.log)"
$SN02 - >> $L/SLAPNT_verify.log 2>&1 <<'PY' || fail "B2 tail present in outputs"
import sleap_io as sio, numpy as np, os, sys
B='figs/out/tmp/slap2m_fair_bvis5/sleap_cap_notail'; bad=0; n=0
for s in os.listdir(B):
    for f in os.listdir(f'{B}/{s}'):
        if f.endswith('.slp') and 'partial' not in f:
            L=sio.load_slp(f'{B}/{s}/{f}'); n+=1
            bad += not all(np.isnan(i.numpy()[[4,7,8,9]]).all() for lf in L for i in lf.instances)
print('OUTPUT TAIL CHECK', n, 'files', bad, 'with tail'); sys.exit(1 if bad else 0)
PY
st "B2 tracked + verified: $(grep -E 'verify\]' $L/SLAPNT_verify.log | tail -1); $(tail -1 $L/SLAPNT_verify.log)"
env FAIR_VARIANT=bvis5 FAIR_SLEAP_NOTAIL=1 SCORE_NOTAIL=1 PYTHONPATH=$CV2H \
    FAIR_LUC3D_DIR=$O/tmp/fig9slap_lite_bvis5_main2dc001fe/sync_stale20_dist25 \
    FAIR_REF_JSON=fig9_slap2m_lite_bvis5_notailscore_md56.json FAIR_OUT_SUFFIX=sleapnt \
    $LZ figs/fig6_slap2m_fair.py --stage score --workers 32 > $L/SLAPNT_score.log 2>&1 || fail "B2 score"
grep -q "FAILED" $L/SLAPNT_score.log && fail "B2 score had failed sessions (SLAPNT_score.log)"
st "B2 DONE: wrote $O/fig6_slap2m_fair_sleapnt.json (+_percam.csv). Not installed into any figure yet."

# ---------------------------------------------------------------------------------------
# A2. B7: Supp 6 small fixes (ByteTrack dedupe, true median, ties-excluded sign test).
# ---------------------------------------------------------------------------------------
st "B7 start: run_S6SMALL.sh"
bash $L/run_S6SMALL.sh > $L/run_S6SMALL.out 2>&1
grep -q "S6SMALL DONE$" $L/run_S6SMALL.out || fail "B7 S6SMALL: $(tail -2 $L/run_S6SMALL.out | tr '\n' ' ')"
grep -qE "exit=[1-9]" $L/run_S6SMALL.out && fail "B7 S6SMALL step failed: $(grep -E 'exit=[1-9]' $L/run_S6SMALL.out | head -2 | tr '\n' ' ')"
st "B7 DONE (tag s6small; not installed)"

# ---------------------------------------------------------------------------------------
# B. B1 now runs on THIS box in private dirs (run_B1_mainbox_mb.sh). Wait for it, then score.
# ---------------------------------------------------------------------------------------
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

# C2. Fig 3F: fresh head-to-head (both arms, 16 jobs, the original protocol), then greedy on
#     exactly the exhaustive frames (same 16 jobs). New tag c2; nothing existing is touched.
idle_wait || fail "C2: machine never idle within 6 h -- timing NOT run"
st "C2 start (Fig 3F head-to-head, tag c2), load $(cut -d' ' -f1-3 /proc/loadavg)"
loadlog "fig3_bench|fig3_exhaustive" > $L/C2_load.log & LP=$!
env LUCID_ROOT=$MAIN FIG3_VARIANT=md56 SCORE_NOTAIL=1 HH_RUN_TAG=c2 HH_JOBS=16 PYTHONPATH=$CV2H \
  $LZ figs/fig3_headtohead.py > $L/C2_headtohead.log 2>&1 || { kill $LP; fail "C2 head-to-head"; }
grep -q "FAILED" $L/C2_headtohead.log && { kill $LP; fail "C2 head-to-head had failed cells"; }
env LUCID_ROOT=$MAIN FIG3_VARIANT=md56 SCORE_NOTAIL=1 HH_RUN_TAG=c2 HH_JOBS=16 PYTHONPATH=$CV2H \
  $LZ figs/fig3_samefr_runtime.py > $L/C2_samefr.log 2>&1 || { kill $LP; fail "C2 same-frames greedy"; }
kill $LP
MAXL=$(awk '{print $2}' $L/C2_load.log | sort -n | tail -1)
st "C2 DONE: max 1-min load $MAXL (16 parallel jobs by design) -> fig3_headtohead_md56_c2.json, fig3_runtime_samefr_md56_c2.json"
echo "OVERNIGHT DONE"; st "OVERNIGHT DONE (main box)"
