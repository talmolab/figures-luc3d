#!/bin/bash
# B series, LUC3D only (2026-10-20; Eric: "make the b series the completely luc3d
# generated ones and not use any numpy triangulation"). Renders every Blender tile
# whose 3D, identities or camera placement used Python geometry, from the LUC3D
# v1.0.5 deposits, into *_b files. The main series' renders are not touched (their
# hashes are in B_main_inputs_before.sha256 and checked at the end). Arguments are each
# render's shipped recipe (run_numpyfree_renders.sh), with --scene/--out pointed at _b.
# Report: figures/drafts/B-SERIES-LUC3D-ONLY-2026-10-20.md
set -euo pipefail
FIGS=/root/vast/eric/sleap-3d-gui/scratch/repos/lucid/figs
export FIG_TAG=b
export POSE_DIR=/root/vast/eric/lucid-main-md56/pose
export PATH=/root/vast/eric/luc3d-bench/video_env/bin:$PATH
# bpy 5.2 needs X11 client libs this container lacks (symlinks into miniforge's pkgs cache)
export LD_LIBRARY_PATH=/tmp/claude-0/-root-vast-eric-sleap-3d-gui-scratch-repos-lucid/4d4f3317-8497-49a9-8c88-e58530b1c39e/scratchpad/bpylibs
cd $FIGS/blender-images
PY=bpyenv/bin/python
R=renders
read AZ EL < <(python3 -c "import json;d=json.load(open('$FIGS/out/fig2a_scene_b.json'));print(d['suggested_azim_deg'],d['suggested_elev_deg'])")

echo "== Fig 3D idswitch prep $(date -u +%T)"; $PY idswitch_fig_prep.py
echo "== Fig 3D idswitch $(date -u +%T)";  $PY idswitch_fig_scene.py
echo "== Fig 3D style $(date -u +%T)";     (cd $FIGS && .venv/bin/python idswitch_fig_style.py)
echo "== Fig 1d pose $(date -u +%T)"
$PY fig1d_scene.py --mode pose --samples 200 --scene $FIGS/out/fig1_hardfight_scene_b.json --out $R/fig1d_pose_b.png
echo "== Fig 1d rig $(date -u +%T)"
$PY fig1d_scene.py --mode rig --samples 200 --scene $FIGS/out/fig1_hardfight_scene_b.json --out $R/fig1d_rig_b.png
echo "== Fig 2A pose az=$AZ el=$EL $(date -u +%T)"
$PY fig1d_scene.py --mode pose --scene $FIGS/out/fig2a_scene_b.json --azim $AZ --elev $EL \
    --aspect 1.19 --out $R/fig2a_pose_b.png
echo "== Fig 1a Mouse-Dyad $(date -u +%T)"
$PY bmimica_scene.py --scene $FIGS/out/fig1_bmimica_scene_b.json --res 1280 960 --ortho 2.15 \
    --out $R/fig1_renders/mouse-dyad_f55701_b.png
echo "== Fig 5A $(date -u +%T)"
$PY fig5a_scene.py --views $FIGS/out/fig5_views_js.json --azim 255 --elev 22 --ortho 0.335 \
    --box 230 230 140 --samples 200 --res 1900 1900 --out $R/fig5a_upright_b.png
echo "== main series untouched? $(date -u +%T)"
(cd $FIGS && sha256sum -c out/md56_logs/B_main_inputs_before.sha256)
echo "== done $(date -u +%T)"
