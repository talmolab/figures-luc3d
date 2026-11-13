#!/usr/bin/env bash
# B series build (2026-10-20, eric-mimicc-0-1; HANDOFF-B-SERIES-LUC3D-2026-10-20.md step 4).
# Figs 1, 2, 3, 5 as FIG_TAG=b. Inputs: x7's link set (what the main figures were built
# from) with these overrides, every one REQUIRED:
#   fig1_hardfight_scene / fig2a_scene  -> the _b scene JSONs (LUC3D v1.0.5 poses + alignment)
#   fig1 / fig2-protocol                -> out/b_app/ (app captures shot on v1.0.5)
#   fig3_*_clean_j17*                   -> the J17 watched timing sitting (the tex quotes it)
# plus FIG_RENDER_REQUIRE=1 (every render a _b file) and FIG_APP_OUT=out/b_app (app tiles).
# Writes only out/*_b.json links and figures/fig{1,2,3,5}b + data/fig{1,2,3,5}b.
set -uo pipefail
R=/root/vast/eric/sleap-3d-gui/scratch/repos/lucid; F=$R/figs; cd $F
L=out/md56_logs; O=out; FPY=.venv/bin/python
fail() { echo "B BUILD FAILED: $*"; exit 1; }
for n in 1 2 3 5; do { [ -e figures/fig${n}b ] || [ -e data/fig${n}b ]; } && fail "figures/fig${n}b or data/fig${n}b exists"; done

# --- the b link set
$FPY - <<'PY' || fail "link set"
import os, sys
O, L = "out", "out/md56_logs"
over = {"fig1_hardfight_scene.json": "fig1_hardfight_scene_b.json",   # real files, already _b
        "fig2a_scene.json": "fig2a_scene_b.json",
        "fig1.json": "b_app/fig1.json",
        "fig2-protocol.json": "b_app/fig2-protocol.json",
        "fig3_headtohead_clean_j17.json": "fig3_headtohead_clean_b3_j17.json",
        "fig3_runtime_clean_j17.json": "fig3_runtime_clean_b3_j17.json",
        "fig3_runtime_clean_j17_a4c6.json": "fig3_runtime_clean_b3_j17_a4c6.json"}
m = {}
for line in open(f"{L}/X7_links.tsv"):
    name, real, _ = line.rstrip("\n").split("\t")
    m[name] = real
m.update(over)
rows = []
for name, tgt in sorted(m.items()):
    stem, ext = os.path.splitext(name)
    link = f"{O}/{stem}_b{ext}"
    if not os.path.exists(f"{O}/{tgt}"):
        sys.exit(f"target {tgt} missing")
    if os.path.realpath(link) == os.path.realpath(f"{O}/{tgt}"):
        pass                                   # the _b scene JSONs are the files themselves
    elif os.path.lexists(link):
        sys.exit(f"{link} exists and is not {tgt}")
    else:
        os.symlink(os.path.relpath(f"{O}/{tgt}", O), link)
    rows.append(f"{name}\t{os.path.relpath(os.path.realpath(f'{O}/{tgt}'), os.path.realpath(O))}\t"
                f"{'override' if name in over else 'X7_links.tsv'}\n")
open(f"{L}/B_links.tsv", "w").writelines(rows)
print(f"{len(rows)} b links ({len(over)} overrides)")
PY

export FIG_TAG=b FIG_RENDER_REQUIRE=1 FIG_APP_OUT=out/b_app
export FIG6F_NOEVICT=stale0_dist25 SIGN_TIES_EXCLUDED=1 FIG5G_PERDIR=1 F3_TRUE_MEDIAN=1 FIG3F_CLEAN=1 \
       FIG3F_CLEAN_TAG=_j17 FIG6_SDANNCE_AGG=$F/fig10-bench/results_v105/agg
export FIG_VARIANT=b FIG_VARIANT_LOG=$PWD/$L/B_loads.tsv
export FIG_VARIANT_REQUIRE=$(cut -f1 $L/B_links.tsv | paste -sd, -)
rm -f $L/B_loads.tsv
for n in 1 2 3 5; do
  $FPY make_figures.py $n > $L/B_fig$n.log 2>&1; E=$?
  NF=$(grep -c "    ! " $L/B_fig$n.log || true)
  echo "fig${n}b exit=$E failed_panels=$NF $(tail -n1 $L/B_fig$n.log)"
  [ $E -eq 0 ] && [ "$NF" = 0 ] || fail "fig$n (see $L/B_fig$n.log)"
done
echo "--- renders used:"; grep -h "\[render b\]" $L/B_fig[1235].log | sort | uniq -c
echo "--- loads that did NOT come from a b link:"; awk -F'\t' '$3 !~ /_b\./' $L/B_loads.tsv | cut -f2,3 | sort | uniq -c

# --- what moved, and proof the main inputs did not
for n in 1 2 3 5; do echo "=== data/fig${n}b vs data/fig$n"; for f in data/fig${n}b/*.csv; do
  b=$(basename $f); [ -e data/fig$n/$b ] || { echo "NEW $b"; continue; }; cmp -s $f data/fig$n/$b || echo "DIFF $b"; done; done
sha256sum -c --quiet $L/B_main_inputs_before.sha256 && echo "MAIN INPUTS UNCHANGED" || fail "a main input changed"
git -C $R status --short -- figs/figures/fig1 figs/figures/fig2 figs/figures/fig3 figs/figures/fig5 \
    figs/data/fig1 figs/data/fig2 figs/data/fig3 figs/data/fig5 | grep . && fail "main figures changed"
echo "B BUILD DONE"
