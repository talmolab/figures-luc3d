#!/usr/bin/env bash
# 2026-10-15 (Eric): build every corrected figure as the B SERIES (figures/fig2b..fig6b,
# data/fig2b..fig6b). Replaces run_FINALBUILD.sh (c series), with the scope cut:
#   - waits for run_OVERNIGHT4.sh (B1 scored + C1 Fig 2G timing) and run_F2PAD.sh, and never
#     builds while a timing sitting runs
#   - Fig 3F: no re-timing (C2 dropped) -> the v105 head-to-head/runtime links stand
#   - Fig 3G/H: full-session sweeps (Eric, 14:45 via ericmimicc) -> the v105 sweep links stand
# Promoting b to the main series happens only after Eric approves.
set -uo pipefail
R=/root/vast/eric/sleap-3d-gui/scratch/repos/lucid; cd $R
L=figs/out/md56_logs; O=figs/out; ST=figs/figures/drafts/RUN-STATUS-2026-10-15.md
LZ=/root/vast/eric/luc3d-bench/liezl_env/bin/python; FPY=figs/.venv/bin/python
CV2H=/tmp/claude-0/-root-vast-eric-sleap-3d-gui-scratch-repos-lucid/44adbb8d-7377-41ea-ab2c-5dae7a320d17/scratchpad/cv2h
st() { echo "- $(date -u +'%F %T') UTC $(hostname -s) [main box] $*" | tee -a $ST; }
fail() { st "BUILD_B FAILED: $*"; echo "BUILD_B FAILED"; exit 1; }

until grep -qE "OVERNIGHT4 DONE|OVERNIGHT FAILED" $L/run_OVERNIGHT4.out 2>/dev/null && grep -qE "F2PAD (DONE|FAILED)" $L/run_F2PAD.out; do sleep 60; done   # run_F2PAD.out = the 15:02 run only (ST holds the 15:01 false start)
grep -q "OVERNIGHT4 DONE" $L/run_OVERNIGHT4.out || fail "run_OVERNIGHT4 failed: $(tail -1 $L/run_OVERNIGHT4.out)"
grep -q "F2PAD DONE" $L/run_F2PAD.out || fail "run_F2PAD failed"
for f in fig7_sleap_max2_sleapnt fig7_bytetrack_max2_s6small fig6_slap2m_fair_final fig2g_retime_all_md56C1 fig4_anipose_md56pad; do
  [ -s $O/$f.json ] || fail "missing input $f.json"
done
python3 -c "import json,sys; d=json.load(open('$O/fig2g_retime_all_md56C1.json')); sys.exit(0 if d['sitting']['valid'] else 1)" \
  || fail "C1 sitting marked invalid (load ceiling) -- do not build Fig 2G from it"
for f in fig7_fair_baselines_final fig7_variant_best_final; do [ -e $O/$f.json ] && fail "$f.json exists"; done
for n in 2 3 4 5 6; do { [ -e figs/figures/fig${n}b ] || [ -e figs/data/fig${n}b ]; } && fail "figs/figures/fig${n}b or data/fig${n}b exists"; done
st "BUILD_B start"

# 1. Supp 6A inputs under tag `final`, then fair baselines + variant best.
ln -sfn fig7_sleap_max2_sleapnt.json      $O/fig7_sleap_max2_final.json
ln -sfn fig7_bytetrack_max2_s6small.json  $O/fig7_bytetrack_max2_final.json
ln -sfn fig8_methods_50_md56.json         $O/fig8_methods_50_final.json
env SCORE_NOTAIL=1 PYTHONPATH=$CV2H FAIR_SUFFIX=final F3_TRUE_MEDIAN=1 $LZ figs/fig6_fair_baselines.py > $L/BUILDB_fair.log 2>&1 || fail "fair baselines"
env SCORE_NOTAIL=1 PYTHONPATH=$CV2H F3_TRUE_MEDIAN=1 $LZ figs/fig6_variant_best_rebuild.py --tag final --base bvis5nt > $L/BUILDB_variant.log 2>&1 || fail "variant rebuild"
# 2. Supp 6B-D: install the SLAP-2M fair block (true medians) into fig7_variant_best_final.json.
env BYTE_B1_DIR=/root/vast/eric/luc3d-bench/outputs/bmimica/results/bytetrack_noretire_notail FAIR_VARIANT=bvis5 \
    FAIR_OUT_SUFFIX=final F3_TRUE_MEDIAN=1 SCORE_NOTAIL=1 PYTHONPATH=$CV2H \
    $LZ figs/fig6_slap2m_fair.py --stage install > $L/BUILDB_install.log 2>&1 || fail "SLAP-2M install: $(tail -2 $L/BUILDB_install.log | tr '\n' ' ')"
st "BUILD_B Supp 6 deposits: $(tail -1 $L/BUILDB_variant.log); install: $(head -1 $L/BUILDB_install.log)"

# 3. The `final` link set: every v105 link, then the two overrides (Fig 2G C1, Fig 2F padded Anipose).
for f in $O/*_v105.json; do b=$(basename $f _v105.json); [ -e $O/${b}_final.json ] || ln -sfn $(readlink $f) $O/${b}_final.json; done
ln -sfn fig2g_retime_all_md56C1.json $O/fig2g_retime_all_final.json
ln -sfn fig4_anipose_md56pad.json    $O/fig4_anipose_final.json
ls -la $O/*_final.json > $L/BUILDB_links.txt

# 4. Build Figs 2-6 as FIG_TAG=b (main figures/figN untouched).
export FIG6F_NOEVICT=stale0_dist25 SIGN_TIES_EXCLUDED=1 FIG5G_PERDIR=1 F3_TRUE_MEDIAN=1 \
       FIG6_SDANNCE_AGG=$R/figs/fig10-bench/results_v105/agg
export FIG_TAG=b
rm -f $L/BUILDB_loads.tsv
for f in 2 3 5 6; do
  FIG_VARIANT=final FIG_VARIANT_LOG=$PWD/$L/BUILDB_loads.tsv $FPY figs/make_figures.py $f > $L/BUILDB_fig$f.log 2>&1
  st "fig${f}b exit=$? $(tail -n1 $L/BUILDB_fig$f.log)"
done
FIG_VARIANT=md56 FIG_VARIANT_LOG=$PWD/$L/BUILDB_loads.tsv $FPY figs/make_figures.py 4 > $L/BUILDB_fig4.log 2>&1   # fig4 panels key on md56
st "fig4b exit=$? $(tail -n1 $L/BUILDB_fig4.log)"
rm -f figs/figures/fig6b/fig6ba_within_vs_cross.pdf figs/figures/fig6b/fig6ba_within_vs_cross.png figs/data/fig6b/fig6a_within_vs_cross.csv

# 5. No variant footer anywhere.
PYMU=/tmp/claude-0/-root-vast-eric-sleap-3d-gui-scratch-repos-lucid/4407f0b0-e869-43bd-bce9-054c05a06416/scratchpad/adv_figs/pylib
PYTHONPATH=$PYMU python3 - > $L/BUILDB_footer_check.txt 2>&1 <<'PY'
import fitz, glob, sys
ps = sorted(glob.glob("figs/figures/fig[2-6]b/fig[2-6]b.pdf")); bad = []
for p in ps:
    txt = "".join(pg.get_text() for pg in fitz.open(p))
    if "Variant fig" in txt or "proofread per-camera 2D labels" in txt: bad.append(p)
print("FOOTER CHECK", "FAIL" if bad or len(ps) != 5 else "PASS", len(ps), "pdfs", bad); sys.exit(1 if bad or len(ps) != 5 else 0)
PY
st "footer: $(tail -1 $L/BUILDB_footer_check.txt)"
echo "--- loads not from final/md56:"; awk -F'\t' '$3 !~ /_final|_md56/' $L/BUILDB_loads.tsv | cut -f2,3 | sort | uniq -c
st "BUILD_B DONE: figures/fig2b..fig6b built (main figures untouched); awaiting review, then commit/push; promotion to main only on Eric's approval"
echo "BUILD_B DONE"
