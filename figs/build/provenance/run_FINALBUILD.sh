#!/usr/bin/env bash
# Phase D1 (2026-10-15): assemble every verified rerun into ONE deposit set, tag `final`, and
# rebuild the main Figs 2-6. Run only after run_OVERNIGHT3.sh printed "OVERNIGHT DONE".
# Nothing existing is overwritten: new *_final files and symlinks only; figures/figN rebuilt.
#   Supp 6A (Mouse-Dyad): SLEAP tail-free (B1, fig7_sleap_max2_sleapnt) + ByteTrack dedupe
#     (B7, fig7_bytetrack_max2_s6small) + LUC3D (fig8_methods_50_md56) -> fair baselines + variant best, true medians
#   Supp 6B-D (SLAP-2M): fig6_slap2m_fair_final (SLEAP tail-free + ByteTrack dedupe) installed with true medians
#   Fig 3G/H: frame-matched sweeps (B8). Fig 3F: same-frame timing (C2). Fig 2G: clean sitting (C1).
#   Supp 6F stale-0 point = v1.0.5 stale0_dist25; Supp 6C ties-excluded sign test; Supp 6G-I s-DANNCE v1.0.5;
#   Fig 5G per-direction far/null.
set -uo pipefail
R=/root/vast/eric/sleap-3d-gui/scratch/repos/lucid; cd $R
L=figs/out/md56_logs; O=figs/out; ST=figs/figures/drafts/RUN-STATUS-2026-10-15.md
LZ=/root/vast/eric/luc3d-bench/liezl_env/bin/python; FPY=figs/.venv/bin/python
CV2H=/tmp/claude-0/-root-vast-eric-sleap-3d-gui-scratch-repos-lucid/44adbb8d-7377-41ea-ab2c-5dae7a320d17/scratchpad/cv2h
st() { echo "- $(date -u +'%F %T') UTC $(hostname -s) [main box] $*" | tee -a $ST; }
fail() { st "FINALBUILD FAILED: $*"; echo "FINALBUILD FAILED"; exit 1; }
grep -q "OVERNIGHT DONE" $L/run_OVERNIGHT3.out || fail "run_OVERNIGHT3 not done"
grep -q "F2PAD DONE" $ST || fail "Fig 2F padded Anipose rerun (run_F2PAD.sh) not done/verified"
for f in fig7_sleap_max2_sleapnt fig7_bytetrack_max2_s6small fig6_slap2m_fair_final fig2g_retime_all_md56C1 \
         fig3_headtohead_samefr_md56_c2 fig3_runtime_samefr_md56_c2 \
         fig3_sweep50__distanceThreshold25-stale20_f7733dbf_md56main_framematched \
         fig3_sweep50__distanceThreshold50-stale0_ce1d7f8d_md56main_framematched; do
  [ -s $O/$f.json ] || fail "missing input $f.json"
done
python3 -c "import json,sys; d=json.load(open('$O/fig2g_retime_all_md56C1.json')); sys.exit(0 if d['sitting']['valid'] else 1)" \
  || fail "C1 sitting marked invalid (load ceiling) -- do not build Fig 2G from it"
for f in fig7_fair_baselines_final fig7_variant_best_final; do [ -e $O/$f.json ] && fail "$f.json exists"; done
st "FINALBUILD start"

# 1. Supp 6A inputs under tag `final` (links to the verified deposits), then fair baselines + variant best.
ln -sfn fig7_sleap_max2_sleapnt.json      $O/fig7_sleap_max2_final.json
ln -sfn fig7_bytetrack_max2_s6small.json  $O/fig7_bytetrack_max2_final.json
ln -sfn fig8_methods_50_md56.json         $O/fig8_methods_50_final.json
env SCORE_NOTAIL=1 PYTHONPATH=$CV2H FAIR_SUFFIX=final F3_TRUE_MEDIAN=1 $LZ figs/fig6_fair_baselines.py > $L/FINAL_fair.log 2>&1 || fail "fair baselines"
env SCORE_NOTAIL=1 PYTHONPATH=$CV2H F3_TRUE_MEDIAN=1 $LZ figs/fig6_variant_best_rebuild.py --tag final --base bvis5nt > $L/FINAL_variant.log 2>&1 || fail "variant rebuild"
# 2. Supp 6B-D: install the SLAP-2M fair block (true medians) into fig7_variant_best_final.json.
env BYTE_B1_DIR=/root/vast/eric/luc3d-bench/outputs/bmimica/results/bytetrack_noretire_notail FAIR_VARIANT=bvis5 \
    FAIR_OUT_SUFFIX=final F3_TRUE_MEDIAN=1 SCORE_NOTAIL=1 PYTHONPATH=$CV2H \
    $LZ figs/fig6_slap2m_fair.py --stage install > $L/FINAL_install.log 2>&1 || fail "SLAP-2M install: $(tail -2 $L/FINAL_install.log | tr '\n' ' ')"
st "FINALBUILD Supp 6 deposits: $(tail -1 $L/FINAL_variant.log); install: $(head -1 $L/FINAL_install.log)"

# 3. The `final` link set: every v105 link, then the overrides.
for f in $O/*_v105.json; do b=$(basename $f _v105.json); [ -e $O/${b}_final.json ] || ln -sfn $(readlink $f) $O/${b}_final.json; done
ln -sfn fig3_sweep50__distanceThreshold50-stale0_ce1d7f8d_md56main_framematched.json $O/fig3_sweep50_final.json
ln -sfn fig3_sweep50__distanceThreshold25-stale20_f7733dbf_md56main_framematched.json $O/fig3_sweep50__distanceThreshold25-stale20-sync_e508a7ab_final.json
ln -sfn fig2g_retime_all_md56C1.json         $O/fig2g_retime_all_final.json
ln -sfn fig3_headtohead_samefr_md56_c2.json  $O/fig3_headtohead_final.json
ln -sfn fig3_runtime_samefr_md56_c2.json     $O/fig3_runtime_final.json
ln -sfn fig4_anipose_md56pad.json             $O/fig4_anipose_final.json   # Fig 2F: Anipose on LUC3D's exact keypoints
ls -la $O/*_final.json > $L/FINAL_links.txt

# 4. Build the corrected series as FIG_TAG=c (main figures/figN untouched).
export FIG6F_NOEVICT=stale0_dist25 SIGN_TIES_EXCLUDED=1 FIG5G_PERDIR=1 F3_TRUE_MEDIAN=1 \
       FIG6_SDANNCE_AGG=$R/figs/fig10-bench/results_v105/agg
export FIG_TAG=c     # new series: figures/fig{2..6}c, data/fig{2..6}c -- nothing existing is overwritten
for n in 2 3 4 5 6; do [ -e figs/figures/fig${n}c ] || [ -e figs/data/fig${n}c ] && fail "figs/figures/fig${n}c or data/fig${n}c exists"; done
rm -f $L/FINAL_loads.tsv
for f in 2 3 5 6; do
  FIG_VARIANT=final FIG_VARIANT_LOG=$PWD/$L/FINAL_loads.tsv $FPY figs/make_figures.py $f > $L/FINAL_fig$f.log 2>&1
  st "fig$f exit=$? $(tail -n1 $L/FINAL_fig$f.log)"
done
FIG_VARIANT=md56 FIG_VARIANT_LOG=$PWD/$L/FINAL_loads.tsv $FPY figs/make_figures.py 4 > $L/FINAL_fig4.log 2>&1   # fig4 panels key on md56
st "fig4 exit=$? $(tail -n1 $L/FINAL_fig4.log)"
rm -f figs/figures/fig6c/fig6ca_within_vs_cross.pdf figs/figures/fig6c/fig6ca_within_vs_cross.png figs/data/fig6c/fig6a_within_vs_cross.csv

# 5. No variant footer anywhere.
PYMU=/tmp/claude-0/-root-vast-eric-sleap-3d-gui-scratch-repos-lucid/4407f0b0-e869-43bd-bce9-054c05a06416/scratchpad/adv_figs/pylib
PYTHONPATH=$PYMU python3 - > $L/FINAL_footer_check.txt 2>&1 <<'PY'
import fitz, glob, sys
bad = []
for p in sorted(glob.glob("figs/figures/fig[2-6]c/fig[2-6]c.pdf")):
    txt = "".join(pg.get_text() for pg in fitz.open(p))
    if "Variant fig" in txt or "proofread per-camera 2D labels" in txt: bad.append(p)
print("FOOTER CHECK", "FAIL" if bad else "PASS", bad); sys.exit(1 if bad else 0)
PY
st "footer: $(cat $L/FINAL_footer_check.txt | tail -1)"
echo "--- loads not from final/md56:"; awk -F'\t' '$3 !~ /_final|_md56/' $L/FINAL_loads.tsv | cut -f2,3 | sort | uniq -c
st "FINALBUILD DONE: figures/fig2c..fig6c built (main figures untouched); tex numbers still to regenerate"
echo "FINALBUILD DONE"
