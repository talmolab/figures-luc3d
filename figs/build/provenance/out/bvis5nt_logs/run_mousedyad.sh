#!/usr/bin/env bash
# bvis5nt phase 2 (2026-10-12): tail-free re-scoring of every Mouse-Dyad-10M tracking deposit,
# plus ByteTrack's never-retire arm re-tracked with tail-free boxes (as SLAP-2M was).
# LUC3D and SLEAP tracker outputs are re-scored as they are (no tracker re-runs). Tagged
# outputs only (*_bvis5nt.json); no base or bvis5 deposit is touched. Waits for phase 1
# (run_fig3.sh) so Fig 3F's exhaustive timings ran on a quiet machine.
set -uo pipefail
cd /root/vast/eric/sleap-3d-gui/scratch/repos/lucid
export PATH=/root/vast/eric/node26_env/bin:$PATH SCORE_NOTAIL=1
export SWEEP50_SUFFIX=bvis5nt FIG8M_SUFFIX=bvis5nt CAMS_SUFFIX=bvis5nt FAIR_SUFFIX=bvis5nt FIG3_VARIANT=bvis5nt
export BYTE_B1_DIR=/root/vast/eric/luc3d-bench/outputs/bmimica/results/bytetrack_noretire_notail
export PYTHONPATH=/tmp/claude-0/-root-vast-eric-sleap-3d-gui-scratch-repos-lucid/44adbb8d-7377-41ea-ab2c-5dae7a320d17/scratchpad/cv2h
PY=/root/vast/eric/luc3d-bench/liezl_env/bin/python; EKS=/root/vast/eric/luc3d-bench/eks_env/bin/python
L=figs/out/bvis5nt_logs; G="--grid-corr3d 0,0.25,0.5,1,2,3,4,6,8,12,16,24 --grid-corr2d 1"
until grep -q "FIG3 DONE" $L/run_fig3.out 2>/dev/null; do sleep 120; done
echo "[$(date +%T)] phase 1 finished; phase 2 start (load $(cut -d' ' -f1 /proc/loadavg))"

laneA() {   # LUC3D re-scoring: staleness sweep -> 3D-term sweep (both arms) -> camera subsets
  FIG8M_WORKERS=28 $PY figs/fig6_methods.py --all-sessions --configs shipped,sync_stale1_dist25,sync_stale10_dist25,sync_stale20_dist25,sync_stale30_dist25,dist25_corr36 > $L/fig8m50.log 2>&1
  echo "[$(date +%T)] A fig8m50 exit=$?"
  $PY figs/fig3_sweep50.py $G --method '{"sync":true,"stale":20}' --thresholds '{"distanceThreshold":25}' --workers 28 > $L/sweep50_fresh.log 2>&1
  echo "[$(date +%T)] A sweep fresh exit=$?"
  $PY figs/fig3_sweep50.py $G --workers 28 > $L/sweep50_shipped.log 2>&1
  echo "[$(date +%T)] A sweep shipped exit=$?"
  $PY figs/fig2_cams_identity.py --stage score --workers 28 > $L/cams.log 2>&1
  echo "[$(date +%T)] A cams exit=$?"
}
laneB() {   # exhaustive reference lines, SLEAP re-score, ByteTrack re-track + score, fair baselines
  $PY figs/fig3_exhaustive_bmimica.py > $L/exh_bm.log 2>&1; echo "[$(date +%T)] B exhaustive_bmimica exit=$?"
  FM_JOBS=12 $PY figs/fig3_frame_matched.py > $L/frame_matched.log 2>&1; echo "[$(date +%T)] B frame_matched exit=$?"
  $PY figs/fig6_sleap_scoped.py --retracked /root/vast/eric/luc3d-bench/outputs/bmimica/retracked_max2 --out fig7_sleap_max2_bvis5nt.json --workers 12 > $L/sleap.log 2>&1
  echo "[$(date +%T)] B sleap exit=$?"
  $EKS figs/fig6_bytetrack_max2.py --stage track --workers 10 > $L/byte_track.log 2>&1; echo "[$(date +%T)] B byte track exit=$?"
  $PY figs/fig6_bytetrack_max2.py --stage score --arms B1,B1s --out fig7_bytetrack_max2_bvis5nt.json --workers 12 > $L/byte_score.log 2>&1
  echo "[$(date +%T)] B byte score exit=$?"
  $PY figs/fig6_fair_baselines.py > $L/fair.log 2>&1; echo "[$(date +%T)] B fair exit=$?"
}
laneA & laneB & wait
ls -la figs/out/*_bvis5nt.json
echo "[$(date +%T)] PHASE2 DONE"
