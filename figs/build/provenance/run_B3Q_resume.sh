#!/usr/bin/env bash
# B3Q (2026-10-15, Eric): every LUC3D Mouse-Dyad tracking arm of Figs 3 and 6 on config b3
# (det_h5_bvis5 body>=5 pool, no sleap-nn filters, tail IN tracking, tail-free scoring),
# then fig3b / fig6b. SLEAP and ByteTrack are NOT run here (Eric's other machine, fig6c);
# fig6b keeps their existing deposits. Writes only *_b3 deposits, *_b3 caches, and
# figures/fig3b, fig6b, data/fig3b, data/fig6b. Stops at the first failed gate.
# The 3G/H sweeps run separately in run_B3S.sh; stage C waits for them.
set -uo pipefail
R=/root/vast/eric/sleap-3d-gui/scratch/repos/lucid; cd $R
L=figs/out/md56_logs; O=figs/out; T=$O/tmp; ST=figs/figures/drafts/RUN-STATUS-2026-10-15.md
MAIN=/root/vast/eric/lucid-main-md56
PY=/root/vast/eric/luc3d-bench/liezl_env/bin/python        # as run_M.sh / run_Q.sh
FPY=figs/.venv/bin/python
CV2H=/tmp/claude-0/-root-vast-eric-sleap-3d-gui-scratch-repos-lucid/44adbb8d-7377-41ea-ab2c-5dae7a320d17/scratchpad/cv2h
export PATH=/root/vast/eric/node26_env/bin:$PATH NODE=/root/vast/eric/node26_env/bin/node SCORE_NOTAIL=1 \
       PYTHONPATH=$CV2H NUMBA_NUM_THREADS=1
st() { echo "- $(date -u '+%F %T') UTC $(hostname -s) [main box] $* (load $(cut -d' ' -f1 /proc/loadavg))" | tee -a $ST; }
fail() { st "B3Q FAILED: $*"; echo "B3Q FAILED"; exit 1; }
need_new() { for p in "$@"; do [ -e "$p" ] && fail "$p already exists (b3 must not reuse or overwrite)"; done; }

[ "$(git -C $MAIN rev-parse --short=8 HEAD)" = 2dc001fe ] || fail "main not at 2dc001fe"
[ -z "$(git -C $MAIN status --porcelain --untracked-files=no)" ] || fail "main checkout dirty"
[ "$(ls /root/vast/eric/luc3d-bench/outputs/bmimica/det_h5_bvis5/*/*_predictions.h5 | wc -l)" = 280 ] || fail "det_h5_bvis5 not 280 files"
# RESUME (2026-10-16 ~10:40 UTC): C1-C4 done; C5 failed on a 50-session guard that did not know b3 (fixed). Resume from C5.
need_new $O/fig3_exhaustive_bmimica_b3.json $O/fig3_runtime_b3.json $T/scale_runtime_b3 $O/fig3_scale_b3.json $O/fig7_variant_best_b3.json $O/fig7_fair_baselines_b3.json figs/figures/fig3b figs/figures/fig6b figs/data/fig3b figs/data/fig6b
st "B3Q resume from C5"
export FIG3_VARIANT=b3
$PY figs/fig3_exhaustive_bmimica.py > $L/B3Q_C5_exh_bm.log 2>&1 || fail "C5 exhaustive_bmimica exit $?"
env LUCID_ROOT=$MAIN $PY figs/fig3_scale_runtime.py > $L/B3Q_C6_scale_runtime.log 2>&1 || fail "C6 scale_runtime exit $?"
unset FIG3_VARIANT
# Gates: tail flags and pool per arm, SLAP-2M cells untouched, b3 greedy == mdb b3.
$PY - > $L/B3Q_C_gate.log 2>&1 <<'EOF' || fail "C gate: $(tail -1 $L/B3Q_C_gate.log)"
import json, sys, glob; sys.path.insert(0, "figs")
from pathlib import Path
import fig3_hh_freshanchor as fa
T = Path("figs/out/tmp"); bad = []
bm = sorted(Path(p).name for p in glob.glob(str(T/"headtohead_b3/A2_C5_bmimica/2025*")))
assert len(bm) == 56, len(bm)
for s in bm:
    e = json.load(open(T/"headtohead_b3/A2_C5_bmimica"/s/"exhaustive.json"))
    g = json.load(open(T/"headtohead_b3/A2_C5_bmimica"/s/"greedy.json"))
    if e.get("excludeTail") is not False or g.get("excludeTail") is not False: bad.append(("tail", s))
vr = T/"headtohead_var_b3_2dc001fe/distanceThreshold25-stale20_f7733dbf/A2_C5_bmimica"
same = sum(fa.payload_digest(vr/s/"greedy.json") == fa.payload_digest(T/"mdb_2dc001fe/b3"/f"{s}.json") for s in bm)
for c in ("A2_C6_slap2m", "A3_C5_slap2m", "A4_C3_slap2m"):
    for f in glob.glob(str(T/f"headtohead_md56/{c}/*/exhaustive.json")):
        g = f.replace("headtohead_md56", "headtohead_b3")
        if open(f, "rb").read() != open(g, "rb").read(): bad.append(("slap2m changed", g))
print(f"GATE C: Mouse-Dyad tail-in both arms 56/56: {not any(b[0]=='tail' for b in bad)}; "
      f"3E greedy identical to mdb b3 {same}/56; SLAP-2M exhaustive unchanged: {not any(b[0]!='tail' for b in bad)}; bad={bad[:4]}")
sys.exit(0 if not bad and same == 56 else 1)
EOF
st "B3Q C done: $(tail -1 $L/B3Q_C_gate.log)"

# ---------------------------------------------------------------- D: deposits for the build
# SLEAP / ByteTrack baselines are NOT rerun here: fig6b keeps v105's fair-baseline deposit.
[ -e $O/fig7_fair_baselines_b3.json ] || ln -s $(readlink $O/fig7_fair_baselines_v105.json) $O/fig7_fair_baselines_b3.json
# fig7_variant_best_b3.json must be a NEW REAL file: a symlink there would make the rebuild
# write through it into the md56 deposit.
[ -e $O/fig7_variant_best_b3.json ] || [ -L $O/fig7_variant_best_b3.json ] && fail "fig7_variant_best_b3.json exists before the rebuild"
env VB_SHIPPED_CELL=stale0_dist25 $PY figs/fig6_variant_best_rebuild.py --tag b3 --base md56 > $L/B3Q_D_variant.log 2>&1 || fail "D variant_best exit $? ($(tail -2 $L/B3Q_D_variant.log | tr '\n' ' '))"
[ -f $O/fig7_variant_best_b3.json ] && [ ! -L $O/fig7_variant_best_b3.json ] || fail "fig7_variant_best_b3.json is not a real file"
# Link set: start from v105 (what the main figures use), then point every Mouse-Dyad LUC3D arm at b3.
# Only links that do not exist yet are created, so no b3 file written above is replaced.
for f in $O/*_v105.json; do b=$(basename $f _v105.json); t=$(readlink $f); [ -n "$t" ] && [ ! -e $O/${b}_b3.json ] && [ ! -L $O/${b}_b3.json ] && ln -s $t $O/${b}_b3.json; done
ln -sfn fig3_sweep50__distanceThreshold25-stale20_f7733dbf_b3.json $O/fig3_sweep50__distanceThreshold25-stale20-sync_e508a7ab_b3.json
ln -sfn fig3_sweep50__distanceThreshold50-stale0_ce1d7f8d_b3.json  $O/fig3_sweep50_b3.json
ln -sfn md56main/fig3_quality__distanceThreshold25-stale20_f7733dbf_b3.json    $O/fig3_quality__distanceThreshold25-stale20-sync_e508a7ab_b3.json
ln -sfn md56main/fig3_headtohead__distanceThreshold25-stale20_f7733dbf_b3.json $O/fig3_headtohead__distanceThreshold25-stale20-sync_e508a7ab_b3.json
# Real b3 files (fig3_quality_b3, fig3_headtohead_b3, fig3_runtime_b3, fig3_scale_b3, fig3_exhaustive_bmimica_b3,
# fig3_frame_matched_bmimica_b3, fig8_methods_50_b3, fig7_variant_best_b3, fig2_cams_identity_b3) were written above;
# the loop skipped them. SLEAP/ByteTrack/fair-baseline and SLAP-2M links stay at v105's targets.
ls -la $O/*_b3.json > $L/B3Q_D_links.txt
$PY - > $L/B3Q_D_gate.log 2>&1 <<'EOF' || fail "D link gate: $(tail -1 $L/B3Q_D_gate.log)"
import os, sys
O = "figs/out"
mine = ["fig3_quality", "fig3_headtohead", "fig3_runtime", "fig3_scale", "fig3_exhaustive_bmimica",
        "fig3_frame_matched_bmimica", "fig8_methods_50", "fig7_variant_best", "fig2_cams_identity",
        "fig3_sweep50", "fig3_sweep50__distanceThreshold25-stale20-sync_e508a7ab",
        "fig3_quality__distanceThreshold25-stale20-sync_e508a7ab", "fig3_headtohead__distanceThreshold25-stale20-sync_e508a7ab"]
bad = []
for n in mine:
    p = f"{O}/{n}_b3.json"
    real = os.path.realpath(p)
    if not os.path.exists(p) or not real.endswith("_b3.json"): bad.append((n, real))
print(f"GATE D: {len(mine)-len(bad)}/{len(mine)} Mouse-Dyad LUC3D deposits resolve to b3 files; bad={bad}")
sys.exit(1 if bad else 0)
EOF
st "B3Q D done: $(tail -1 $L/B3Q_D_gate.log)"

# ---------------------------------------------------------------- E: build fig3b, fig6b
REQ=$(ls $O/*_b3.json | xargs -n1 basename | sed 's/_b3\.json$/.json/' | paste -sd, -)
export FIG_VARIANT=b3 FIG_TAG=b FIG_VARIANT_REQUIRE="$REQ" FIG_VARIANT_LOG=$PWD/$L/B3Q_E_loads.tsv \
       FIG6F_NOEVICT=stale0_dist25
rm -f $L/B3Q_E_loads.tsv
$FPY figs/make_figures.py 3 6 > $L/B3Q_E_make_figures.log 2>&1; E=$?
FAILED_PANELS=$(grep -c "    ! " $L/B3Q_E_make_figures.log || true)
ls figs/figures/fig3b figs/figures/fig6b > $L/B3Q_E_outputs.txt 2>&1
if [ $E -eq 0 ] && [ "$FAILED_PANELS" = 0 ]; then
  st "B3Q DONE: fig3b + fig6b built (figs/figures/fig3b, fig6b). Loads: $L/B3Q_E_loads.tsv. Awaiting Eric's review."
else
  st "B3Q BUILD INCOMPLETE: make_figures exit $E, $FAILED_PANELS failed panels -- see $L/B3Q_E_make_figures.log"
fi
echo "B3Q END"
