#!/usr/bin/env bash
# S6SMALL (prepared 2026-10-15, NOT YET RUN): Supplementary Fig 6 small audit fixes.
# Additive: every output uses the new tag `s6small`; no md56/v105 deposit is touched.
#   1. True medians (F3_TRUE_MEDIAN=1): fig3_trackers.summarise() used s[n//2], the
#      upper-middle value for even n (74 SLAP-2M sessions). ByteTrack within-view median
#      should become 0.987 (was printed 0.988).
#   2. ByteTrack per-frame slot dedupe (STITCH_DEDUPE=1): stitch_to_2's no-free-slot
#      fallback could put two tracklets on one identity in one frame (IDF1 > 1 in 10 of
#      444 SLAP-2M camera-sessions). Re-stitch only; ByteTrack itself is not re-run.
#        SLAP-2M:    fig6_slap2m_fair.py --stage bytetrack with FAIR_BYTE_RESTITCH_FROM
#        Mouse-Dyad: fig6_bytetrack_max2.py --stage score (the stitch is applied at score time)
#   3. Sign tests with ties excluded in the Supp 6C panel (SIGN_TIES_EXCLUDED=1), matching
#      the text (>= 2 animals: 29 vs 12, 1 tie, P = 0.012).
# Then: fair baselines -> variant_best rebuild -> fig6 rebuilt as FIG_TAG=s6small.
# Accuracy only, nothing timed. Must still not overlap a timing run.
set -uo pipefail
cd /root/vast/eric/sleap-3d-gui/scratch/repos/lucid
L=figs/out/md56_logs; O=figs/out
PY=/root/vast/eric/luc3d-bench/liezl_env/bin/python; FPY=figs/.venv/bin/python
t() { echo "[$(date +%T)] $* (load $(cut -d' ' -f1 /proc/loadavg))"; }
export PATH=/root/vast/eric/node26_env/bin:$PATH SCORE_NOTAIL=1
export PYTHONPATH=/tmp/claude-0/-root-vast-eric-sleap-3d-gui-scratch-repos-lucid/44adbb8d-7377-41ea-ab2c-5dae7a320d17/scratchpad/cv2h
TAG=s6small
for f in fig7_sleap_max2 fig7_bytetrack_max2 fig7_fair_baselines fig7_variant_best fig6_slap2m_fair; do
  [ -e $O/${f}_$TAG.json ] && { t "REFUSING: $O/${f}_$TAG.json exists"; echo "S6SMALL DONE (FAILED)"; exit 1; }
done

# --- Mouse-Dyad (Supp 6A): ByteTrack re-scored with the dedupe; SLEAP and LUC3D unchanged.
cp $O/fig7_sleap_max2_md56.json $O/fig7_sleap_max2_$TAG.json
ln -sfn fig8_methods_50_md56.json $O/fig8_methods_50_$TAG.json
STITCH_DEDUPE=1 BYTE_B1_DIR=/root/vast/eric/luc3d-bench/outputs/bmimica/results/bytetrack_noretire_notail \
  $PY figs/fig6_bytetrack_max2.py --stage score --arms B1,B1s --out fig7_bytetrack_max2_$TAG.json --workers 12 \
  > $L/S6SMALL_byte_md.log 2>&1; t "1/6 Mouse-Dyad ByteTrack rescore exit=$?"
FAIR_SUFFIX=$TAG F3_TRUE_MEDIAN=1 $PY figs/fig6_fair_baselines.py > $L/S6SMALL_fair.log 2>&1; t "2/6 fair baselines exit=$?"
F3_TRUE_MEDIAN=1 $PY figs/fig6_variant_best_rebuild.py --tag $TAG --base bvis5nt > $L/S6SMALL_variant.log 2>&1
t "3/6 variant rebuild exit=$? $(tail -n 1 $L/S6SMALL_variant.log)"

# --- SLAP-2M (Supp 6B-D): re-stitch ByteTrack with the dedupe, re-score, install with true medians.
FAIR="FAIR_VARIANT=bvis5 FAIR_LUC3D_DIR=figs/out/tmp/fig9slap_lite_bvis5_main2dc001fe/sync_stale20_dist25 FAIR_REF_JSON=fig9_slap2m_lite_bvis5_notailscore_md56.json FAIR_OUT_SUFFIX=$TAG"
env $FAIR STITCH_DEDUPE=1 FAIR_BYTE_RESTITCH_FROM=$O/tmp/slap2m_fair_bvis5/bytetrack_noretire_stitch \
  FAIR_BYTE_OUT=$O/tmp/slap2m_fair_bvis5/bytetrack_noretire_stitch_dedupe \
  $PY figs/fig6_slap2m_fair.py --stage bytetrack --workers 32 > $L/S6SMALL_byte_slap.log 2>&1; t "4/6 SLAP-2M ByteTrack re-stitch exit=$?"
env $FAIR FAIR_BYTE_OUT=$O/tmp/slap2m_fair_bvis5/bytetrack_noretire_stitch_dedupe \
  $PY figs/fig6_slap2m_fair.py --stage score --workers 32 > $L/S6SMALL_score_slap.log 2>&1; t "5/6 SLAP-2M score exit=$?"
# install seeds fig7_variant_best_<suffix> from the BASE file when missing; it already exists
# here (step 3), so only its slap2m_fair block is replaced.
env $FAIR F3_TRUE_MEDIAN=1 $PY figs/fig6_slap2m_fair.py --stage install > $L/S6SMALL_install.log 2>&1
t "6/6 SLAP-2M install exit=$?"; cat $L/S6SMALL_install.log

# --- Figure: fig6 under FIG_TAG=s6small (published fig6 untouched).
for f in $O/*_v105.json; do b=$(basename $f _v105.json); [ -e $O/${b}_$TAG.json ] || ln -sfn $(readlink $f) $O/${b}_$TAG.json; done
FIG_VARIANT=$TAG FIG_TAG=$TAG FIG6F_NOEVICT=stale0_dist25 SIGN_TIES_EXCLUDED=1 $FPY figs/make_figures.py 6 > $L/S6SMALL_fig6.log 2>&1
t "fig6 s6small exit=$? $(tail -n 1 $L/S6SMALL_fig6.log)"
echo "S6SMALL DONE"
