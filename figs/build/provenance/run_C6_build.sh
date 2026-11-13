#!/usr/bin/env bash
# C6 final: Fig 6c = the b-series fig6 (LUC3D on b3, from the main box's B3Q) with the SLEAP and
# ByteTrack arms ALSO on b3 (this box). Waits for (1) C6 SLEAP score, (2) C6 ByteTrack score,
# (3) main box B3Q finished. Writes only *_c6b3 deposits/links and figures/fig6c, data/fig6c.
set -uo pipefail
R=/root/vast/eric/sleap-3d-gui/scratch/repos/lucid; cd $R
L=figs/out/md56_logs; O=figs/out; ST=figs/figures/drafts/RUN-STATUS-2026-10-15.md
PY=/root/vast/eric/luc3d-bench/liezl_env/bin/python; FPY=figs/.venv/bin/python
CV2H=/tmp/claude-0/-root-vast-eric-sleap-3d-gui-scratch-repos-lucid/44adbb8d-7377-41ea-ab2c-5dae7a320d17/scratchpad/cv2h
export PATH=/root/vast/eric/node26_env/bin:$PATH NODE=/root/vast/eric/node26_env/bin/node SCORE_NOTAIL=1 \
       PYTHONPATH=$CV2H NUMBA_NUM_THREADS=1
st() { echo "- $(date -u +%H:%M) | ericmimicc | C6 build | $*" >> $ST; echo "$*"; }
fail() { st "FAILED: $*"; echo "C6 BUILD FAILED"; exit 1; }
N=${C6_FROM_LINE:-$(wc -l < $ST)}   # only status lines written after this point count for SLEAP (attempt 1 failed at 23:41)
new() { tail -n +$((N + 1)) $ST; }
st "waiting for SLEAP b3 score, ByteTrack b3 score, and main-box B3Q"
until new | grep -q "C6 SLEAP b3 | score exit="; do new | grep -q "C6 SLEAP b3 | FAILED" && fail "SLEAP chain failed"; sleep 300; done
new | grep -q "C6 SLEAP b3 | score exit=0" || fail "SLEAP score nonzero exit"
grep -q "C6 ByteTrack b3 | score exit=0" $ST || fail "ByteTrack score not ok"
until grep -qE "B3Q DONE|B3Q BUILD INCOMPLETE|B3Q FAILED" $ST; do sleep 300; done
grep -q "B3Q D done" $ST || fail "main-box B3Q did not reach its deposit stage"
for f in fig7_fair_baselines fig7_variant_best fig8_methods_50; do
  [ -e $O/${f}_c6b3.json ] || [ -L $O/${f}_c6b3.json ] && fail "$O/${f}_c6b3.json exists already"
done
FAIR_SUFFIX=c6b3 $PY figs/fig6_fair_baselines.py > $L/C6_build_fair.log 2>&1 || fail "fair baselines: $(tail -1 $L/C6_build_fair.log)"
ln -s fig8_methods_50_b3.json $O/fig8_methods_50_c6b3.json
env VB_SHIPPED_CELL=stale0_dist25 $PY figs/fig6_variant_best_rebuild.py --tag c6b3 --base md56 > $L/C6_build_variant.log 2>&1 || fail "variant_best: $(tail -2 $L/C6_build_variant.log | tr '\n' ' ')"
# every other deposit: exactly what fig6b used
for f in $O/*_b3.json; do b=$(basename $f _b3.json); [ -e $O/${b}_c6b3.json ] || [ -L $O/${b}_c6b3.json ] && continue
  if [ -L $f ]; then ln -s $(readlink $f) $O/${b}_c6b3.json; else ln -s $(basename $f) $O/${b}_c6b3.json; fi; done
$PY - > $L/C6_build_gate.log 2>&1 <<'PYEOF' || fail "gate: $(tail -1 $L/C6_build_gate.log)"
import json, os, sys
O = "figs/out"
fair = json.load(open(f"{O}/fig7_fair_baselines_c6b3.json"))
assert fair["sources"]["sleap"] == "fig7_sleap_max2_c6b3.json" and fair["sources"]["bytetrack"] == "fig7_bytetrack_max2_c6b3.json", fair["sources"]
assert len(fair["sessions"]) == 56, len(fair["sessions"])
vb = json.load(open(f"{O}/fig7_variant_best_c6b3.json"))
assert len(vb["bmimica_sessions"]) == 56
assert os.path.realpath(f"{O}/fig8_methods_50_c6b3.json").endswith("fig8_methods_50_b3.json")
for k, a in fair["arms"].items():
    print(f"  {k}: within {a['within']['mean']:.4f} cross {a['cross']['mean']:.4f}")
l = vb["bmimica_50_sessions"]["LUC3D"]
print(f"  LUC3D (b3): within {l['within']['mean']:.4f} cross {l['cross']['mean']:.4f}")
print("GATE C6: 56 sessions, SLEAP+ByteTrack from c6b3, LUC3D from b3")
PYEOF
REQ=$(ls $O/*_c6b3.json | xargs -n1 basename | sed 's/_c6b3\.json$/.json/' | paste -sd, -)
export FIG_VARIANT=c6b3 FIG_TAG=c FIG_VARIANT_REQUIRE="$REQ" FIG_VARIANT_LOG=$PWD/$L/C6_build_loads.tsv FIG6F_NOEVICT=stale0_dist25
rm -f $L/C6_build_loads.tsv
$FPY figs/make_figures.py 6 > $L/C6_build_make_figures.log 2>&1; E=$?
NF=$(grep -c "    ! " $L/C6_build_make_figures.log || true)
[ $E -eq 0 ] && [ "$NF" = 0 ] || fail "make_figures exit $E, $NF failed panels (C6_build_make_figures.log)"
st "DONE: figures/fig6c built. $(grep -A3 'GATE C6' $L/C6_build_gate.log | head -1) | $(head -4 $L/C6_build_gate.log | tr '\n' ' ')"
echo "C6 BUILD DONE"
