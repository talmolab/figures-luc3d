#!/usr/bin/env bash
# ericmimicc queue, Job 1 of HANDOFF-OTHER-JOB-2026-10-15.md: Fig 3G/H frame-matched
# rescoring, then Gate 2. Follows the handoff exactly. Writes OTHER_JOB1.finished
# (PASS/FAIL) when it ends, which Job 2's queue watches to raise its worker count.
set -uo pipefail
REPO=/root/vast/eric/sleap-3d-gui/scratch/repos/lucid
L=$REPO/figs/out/md56_logs
E=/root/vast/eric/envs/md56_score_env
ST=$REPO/figs/figures/drafts/RUN-STATUS-2026-10-15.md
log() { echo "[$(date -u +%FT%TZ) $(hostname -s)] $* (load $(cut -d' ' -f1 /proc/loadavg))"; }
status() { echo "- $(date -u +%H:%M) | ericmimicc | $*" >> $ST; }
finish() { echo "$1" > $L/OTHER_JOB1.finished; log "JOB1 $1"; status "B8 Job 1 G3sfm | $1 | $2"; echo "OTHER JOB1 $1"; exit $([ "$1" = PASS ] && echo 0 || echo 1); }
rm -f $L/OTHER_JOB1.finished
cd $REPO

log "JOB1 start"
status "B8 Job 1 G3sfm | started | figs/out/md56_logs/OTHER_G3sfm.log"
# 6 cells were rescored on the main box before it was stopped; delete them so every cell
# comes from ONE environment:
find figs/out/tmp/sweep50_2dc001fe -name "*.framematched.json" -print -delete > $L/OTHER_G3sfm_deleted.log
log "deleted $(wc -l < $L/OTHER_G3sfm_deleted.log) stale framematched caches"
W=$(( ($(nproc) - 4) / 2 ))
log "SFM_JOBS=$W"
FIG3_VARIANT=md56 SCORE_NOTAIL=1 SFM_JOBS=$W \
  $E/bin/python figs/fig3_sweep_framematched.py > $L/OTHER_G3sfm.log 2>&1
rc=$?
log "fig3_sweep_framematched exit=$rc"
echo "exit=$rc" >> $L/OTHER_G3sfm.log
[ $rc -eq 0 ] || finish FAIL "exit $rc, see OTHER_G3sfm.log"
grep -q "SFM DONE" $L/OTHER_G3sfm.log || finish FAIL "no SFM DONE in log"
[ "$(grep -c "wrote .*_framematched.json" $L/OTHER_G3sfm.log)" -eq 2 ] || finish FAIL "expected 2 'wrote' lines"

log "Gate 2"
cd $REPO/figs/out
$E/bin/python - > $L/OTHER_G3sfm_gate2.log 2>&1 <<'EOF'
import json
fm = json.load(open('fig3_frame_matched_bmimica_md56.json'))      # exhaustive's own frames
exh_camf = fm['arms']['exhaustive']['camera_frames_scored']
for n in ['fig3_sweep50__distanceThreshold25-stale20_f7733dbf_md56main_framematched.json',
          'fig3_sweep50__distanceThreshold50-stale0_ce1d7f8d_md56main_framematched.json']:
    d = json.load(open(n))
    ok = d['total_camera_frames'] == exh_camf and len(d['sessions']) == 56 and len(d['cells']) == 12
    print(n[:70], 'camera frames', d['total_camera_frames'], 'vs exhaustive', exh_camf, 'cells', len(d['cells']), 'PASS' if ok else 'FAIL')
cur = json.load(open('fig3_sweep50__distanceThreshold25-stale20_f7733dbf_md56main_framematched.json'))
c6 = next(c for c in cur['cells'] if c['corr3d'] == 6)
g = fm['arms']['greedy']
print('r=6 vs existing frame-matched greedy arm: switches', c6['switches'], g['switches_total'],
      ' idf1_cross', round(c6['idf1_cross'], 6), round(g['idf1_cross_mean'], 6),
      'PASS' if (c6['switches'] == g['switches_total'] and abs(c6['idf1_cross'] - g['idf1_cross_mean']) < 1e-12) else 'FAIL')
EOF
rc=$?
cat $L/OTHER_G3sfm_gate2.log
if [ $rc -eq 0 ] && [ "$(grep -c ' PASS$' $L/OTHER_G3sfm_gate2.log)" -eq 3 ] && ! grep -q "FAIL" $L/OTHER_G3sfm_gate2.log; then
  finish PASS "Gate 2 PASS x3, OTHER_G3sfm_gate2.log"
else
  finish FAIL "Gate 2 failed (rc=$rc), OTHER_G3sfm_gate2.log"
fi
