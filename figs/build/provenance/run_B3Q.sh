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
need_new $O/fig8_methods_50_b3.json $T/fig8m50_2dc001fe_b3 $O/fig2_cams_identity_b3.json $T/cams_identity_2dc001fe_b3 \
         $T/headtohead_b3 $T/headtohead_var_b3_2dc001fe $O/fig3_headtohead_b3.json $O/fig3_quality_b3.json \
         $O/fig3_exhaustive_bmimica_b3.json $O/fig3_frame_matched_bmimica_b3.json $O/fig3_runtime_b3.json \
         $T/scale_runtime_b3 $O/fig3_scale_b3.json $O/md56main/fig3_quality__distanceThreshold25-stale20_f7733dbf_b3.json \
         $O/md56main/fig3_headtohead__distanceThreshold25-stale20_f7733dbf_b3.json $O/fig7_variant_best_b3.json $O/fig7_fair_baselines_b3.json figs/figures/fig3b figs/figures/fig6b figs/data/fig3b figs/data/fig6b
st "B3Q start: LUC3D Mouse-Dyad arms on b3 (det_h5_bvis5, tail in tracking), main 2dc001fe, node $(node --version)"

# ---------------------------------------------------------------- A1: Supp 6F + 6A LUC3D
CFGS=sync_stale1_dist25,sync_stale10_dist25,sync_stale20_dist25,sync_stale30_dist25,stale0_dist25_corr36,stale0_dist25
env MD_CFG=b3 FIG8M_LUCID_ROOT=$MAIN FIG8M_SUFFIX=b3 FIG8M_WORKERS=20 \
  $PY figs/fig6_methods.py --all-sessions --configs $CFGS > $L/B3Q_A1_fig8m.log 2>&1 || fail "A1 fig6_methods exit $? ($(tail -1 $L/B3Q_A1_fig8m.log))"
# Gate: the v1.0.5 default cell must be byte-identical to the validated mdb b3 run, 56/56.
$PY - > $L/B3Q_A1_gate.log 2>&1 <<'EOF' || fail "A1 gate: $(tail -1 $L/B3Q_A1_gate.log)"
import json, sys; sys.path.insert(0, "figs")
from pathlib import Path
import fig3_hh_freshanchor as fa
O = Path("figs/out"); d = json.load(open(O / "fig8_methods_50_b3.json"))
cells = {c["config"]: c for c in d["cells"]}
want = {"sync_stale1_dist25","sync_stale10_dist25","sync_stale20_dist25","sync_stale30_dist25","stale0_dist25_corr36","stale0_dist25"}
assert want <= set(cells), f"missing cells {want - set(cells)}"
assert "shipped" not in cells, "a shipped (paper-pool fork) cell leaked into the b3 deposit"
for c in want: assert len(cells[c]["per_session"]) == 56, (c, len(cells[c]["per_session"]))
S = sorted(p["session"] for p in cells["sync_stale20_dist25"]["per_session"])
same = sum(fa.payload_digest(O/"tmp/fig8m50_2dc001fe_b3/sync_stale20_dist25"/f"{s}.json") == fa.payload_digest(O/"tmp/mdb_2dc001fe/b3"/f"{s}.json") for s in S)
ref = json.load(open(O / "mdb_experiment_v105.json"))["configs"]["b3"]["per_session"]
idf = sum(abs(p["cross_idf1"] - ref[p["session"]]["cross_idf1"]) < 1e-12 for p in cells["sync_stale20_dist25"]["per_session"])
print(f"GATE A1: outputs identical to mdb b3 {same}/56, cross IDF1 identical {idf}/56, mean {cells['sync_stale20_dist25']['idf1_cross']:.4f}")
sys.exit(0 if same == 56 and idf == 56 else 1)
EOF
st "B3Q A1 done: $(tail -1 $L/B3Q_A1_gate.log)"

# ---------------------------------------------------------------- A2: camera subsets (text)
for stage in track score; do
  env MD_CFG=b3 LUCID_ROOT=$MAIN CAMS_SUFFIX=b3 CAMS_HH_CACHE=$T/headtohead_md56/A2_C5_bmimica \
    $PY figs/fig2_cams_identity.py --stage $stage --workers 20 > $L/B3Q_A2_cams_$stage.log 2>&1 || fail "A2 cams $stage exit $?"
done
grep -q "GATE FAILED" $L/B3Q_A2_cams_score.log && fail "A2 cams k5 gate failed"
grep "GATE k5_full" $L/B3Q_A2_cams_score.log | grep -q "56 sessions" || fail "A2 cams gate not on 56 sessions: $(grep GATE $L/B3Q_A2_cams_score.log)"
st "B3Q A2 done: $(grep 'GATE k5_full' $L/B3Q_A2_cams_score.log)"

# ---------------------------------------------------------------- B: seed SLAP-2M cells (unchanged, md56)
# Fig 3E/F pools Mouse-Dyad with 42 SLAP-2M sessions. Those cells are not b3's business: COPY
# (not link, so nothing can write back into md56's cache) md56's cached SLAP-2M cells.
mkdir -p $T/headtohead_b3 $T/headtohead_var_b3_2dc001fe/distanceThreshold25-stale20_f7733dbf
for c in A2_C6_slap2m A3_C5_slap2m A4_C3_slap2m; do
  cp -a $T/headtohead_md56/$c $T/headtohead_b3/ || fail "B copy $c"
  cp -a $T/headtohead_var_md56_2dc001fe/distanceThreshold25-stale20_f7733dbf/$c $T/headtohead_var_b3_2dc001fe/distanceThreshold25-stale20_f7733dbf/ || fail "B copy var $c"
done
st "B3Q B done: SLAP-2M head-to-head cells copied from md56"

# ---------------------------------------------------------------- C: Fig 3E/F/G/H rules (needs a quiet box)
# 3F times the exhaustive arm, so wait until the sweeps (run_B3S.sh) are finished.
until grep -qE "B3S (DONE|FAILED)" $ST; do sleep 120; done
grep -q "B3S DONE" $ST || fail "sweeps (run_B3S.sh) failed"
st "B3Q C start (sweeps done; nothing else of ours running: $(pgrep -fc 'fig3_sweep50|fig6_methods|fig2_cams' || true) jobs)"
export FIG3_VARIANT=b3
env LUCID_ROOT=$MAIN HH_JOBS=16 $PY figs/fig3_headtohead.py > $L/B3Q_C1_headtohead.log 2>&1 || fail "C1 headtohead exit $?"
$PY figs/fig3_quality.py > $L/B3Q_C2_quality.log 2>&1 || fail "C2 quality exit $?"
env FRESHANCHOR_LUCID_ROOT=$MAIN $PY figs/fig3_hh_freshanchor.py --run --jobs 16 > $L/B3Q_C3_freshanchor.log 2>&1 || fail "C3 freshanchor exit $?"
env FM_JOBS=12 $PY figs/fig3_frame_matched.py > $L/B3Q_C4_frame_matched.log 2>&1 || fail "C4 frame_matched exit $?"
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
