#!/usr/bin/env bash
# Rebuild the seven placed figures exactly as published, from this repo alone.
#
#   bash figs/build/run_FINALBUILD.sh            # rebuild in place (figs/figures, figs/data)
#   bash figs/build/run_FINALBUILD.sh --verify   # rebuild a clean copy of this checkout in a
#                                                # temp dir and pixel-compare it to this checkout
#
# FIG_SERIES=q builds the same figures from the q deposits (out/<stem>_q.*), the b series
# with the sessions in src/qc_exclude.py left out (figs/qc/). Default b. --verify is for b.
#
# PYTHON=/path/to/python overrides the interpreter (3.12 with numpy, pandas, scipy,
# matplotlib, seaborn, PyMuPDF). Verified 2026-11-12: every placed composite and panel is
# pixel-identical to the committed PDFs (compare.py), and every plotted data/ CSV is
# byte-identical.
#
# What the published figures are. They are the "b" series (2026-10-20/21, promoted to the
# main figures in luc3d eric/figs 9f25344b8), not plain x7:
#   * deposits: x7's link set plus 8 overrides (build/B_links.tsv, every name REQUIRED):
#     LUC3D v1.0.5 app captures (out/b_app/) for Fig 1 and the labeling protocol, the
#     LUC3D-triangulated HardFight / two-anchor scenes, the J18 timing sitting for the
#     Fig 2 time-per-frame panel, and the NumPy-free Fig 4 detection deposit;
#   * renders: the *_b.png Blender renders made from LUC3D v1.0.5 output
#     (FIG_TAG=b + FIG_RENDER_REQUIRE=1 makes tagged_render() refuse the older base renders);
#   * two exceptions, kept exactly as placed:
#       - Fig 3a (labeling protocol) was re-drawn at the Fig 2<->3 swap (2026-10-29) with
#         FIG_VARIANT=x7, i.e. its reprojection overlays come from the August app export
#         out/fig2-protocol.json, not out/b_app/. Both are LUC3D app exports (no NumPy);
#         the overlay lines differ by <= 23/255 on 713 px.
#       - Fig 7 (Supp Fig 2, calibration) keeps its x7 build: its data equal b, but a b
#         build stamps a variant footer on it.
# Producer history and the original launch scripts are in build/provenance/
# (run_X7_build.sh -> run_PROMOTE_x7.sh -> run_B_luc3d_renders.sh -> run_B_luc3d_build.sh
# -> run_B_luc3d_build_467.sh -> b promote -> run_J18_fig3f_timing.sh). run_FINALBUILD.sh
# there is the 2026-10-15 "c" series and run_B_numpyfree.sh an aborted first attempt at b;
# neither produced a placed figure.
set -euo pipefail
FIGS="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PYTHON:-python3}"
SERIES="${FIG_SERIES:-b}"
case "$SERIES" in b|q) ;; *) echo "FIG_SERIES must be b or q" >&2; exit 1 ;; esac
BUILD="$FIGS/build"

if [ "${1:-}" = "--verify" ]; then
    T="$(mktemp -d)"
    trap 'rm -rf "$T"' EXIT
    # A clean copy of this checkout's figs/ (tracked + untracked, nothing git ignores).
    (cd "$FIGS" && git ls-files -co --exclude-standard -z . | tar --null -T - -c) | tar -x -C "$T"
    # Empty the figure folders, so a composite can only be built from freshly drawn panels.
    rm -f "$T"/figures/fig[1-7]/*
    touch "$T/.build_start"
    PYTHON="$PY" bash "$T/build/run_FINALBUILD.sh"
    "$PY" "$BUILD/compare.py" "$T" "$FIGS" --since "$T/.build_start"
    exit $?
fi

cd "$FIGS"
fail() { echo "FINALBUILD FAILED: $*" >&2; exit 1; }

# --- 0. inputs this build needs that a plain x7 build does not
for f in blender-images/renders/fig1d_pose_b.png blender-images/renders/fig1d_rig_b.png \
         blender-images/renders/fig2a_pose_b.png blender-images/renders/fig5a_upright_b.png \
         blender-images/renders/fig1_renders/mouse-dyad_f55701_b.png \
         figures/drafts/figs/lucid_idswitch_style_hardfight_b.png \
         out/b_app/fig1.json out/b_app/fig2-protocol.json out/stale_sweep_slap2m.json \
         fig10-bench/results_v105/agg/summary.csv; do
    [ -e "$f" ] || fail "missing input $f"
done
while IFS=$'\t' read -r name _ _; do
    stem="${name%.*}"; ext="${name##*.}"
    [ -e "out/${stem}_${SERIES}.${ext}" ] || fail "missing $SERIES link out/${stem}_${SERIES}.${ext}"
done < "$BUILD/B_links.tsv"
for n in 1 2 3 4 5 6; do rm -rf "figures/fig${n}b" "data/fig${n}b"; done

# --- the b environment (run_B_luc3d_build.sh, with the J18 timing that the paper quotes)
export FIG6F_NOEVICT=stale0_dist25 SIGN_TIES_EXCLUDED=1 FIG5G_PERDIR=1 F3_TRUE_MEDIAN=1 \
       FIG3F_CLEAN=1 FIG3F_CLEAN_TAG=_j18 FIG6_SDANNCE_AGG="$FIGS/fig10-bench/results_v105/agg"
B_REQUIRE="$(cut -f1 "$BUILD/B_links.tsv" | paste -sd, -)"
X7_REQUIRE="$(cut -f1 "$BUILD/X7_links.tsv" | paste -sd, -)"

# --- 1. Supp Fig 4 f first: it writes the axes json that the s-DANNCE panel aligns to
mkdir -p figures/fig6b
FIG_TAG=b FIG_VARIANT="$SERIES" "$PY" panels/fig6slapstale_reduction.py

# --- 2. Figs 1-6 panels as the b series (written to figures/fig{N}b, data/fig{N}b)
FIG_TAG=b FIG_RENDER_REQUIRE=1 FIG_APP_OUT=out/b_app FIG_VARIANT="$SERIES" \
FIG_VARIANT_REQUIRE="$B_REQUIRE" "$PY" make_figures.py --panels 1 2 3 4 5 6 | tee "$BUILD/last_b.log"
grep -q " 0 failed" "$BUILD/last_b.log" || fail "a b panel failed (see build/last_b.log)"

# --- 3. Fig 3a as placed at the swap: x7 deposits, b renders
FIG_TAG=b FIG_RENDER_REQUIRE=1 FIG_VARIANT=x7 "$PY" panels/fig3_01_protocol.py

# --- 4. promote fig{N}b -> fig{N}, as the 2026-10-21 promote did
"$PY" "$BUILD/promote.py" . b 1,2,3,4,5,6
for n in 1 2 3 4 5 6; do rm -rf "figures/fig${n}b" "data/fig${n}b"; done

# --- 5. Fig 7 (Supp Fig 2): x7, untagged
FIG_VARIANT=x7 FIG_VARIANT_REQUIRE="$X7_REQUIRE" "$PY" make_figures.py --panels 7 \
    | tee "$BUILD/last_x7.log"
grep -q " 0 failed" "$BUILD/last_x7.log" || fail "a Fig 7 panel failed (see build/last_x7.log)"

# --- 6. composites, untagged and variant-free (no footer)
env -u FIG_TAG -u FIG_VARIANT "$PY" assemble.py 1 2 3 4 5 6 7
rm -f "$BUILD/last_b.log" "$BUILD/last_x7.log"
echo "FINALBUILD DONE"
