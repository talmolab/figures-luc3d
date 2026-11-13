#!/bin/bash
# LUC3D-JS plan, 2026-10-20 (Eric: "get rid of all the numpy triangulation from all the
# figures ... use luc3d triangulations and rerender everything"). Re-renders every
# Blender tile whose 3D, identities or camera placement used Python geometry, from the
# LUC3D v1.0.5 deposits (hardfight_luc3d.npz, fig5_views_js.json, the rebuilt scene
# JSONs). Arguments are each render's documented shipped recipe. Previous renders are
# in out/md56_logs/numpy_renders_backup_2026-10-20/.
set -euo pipefail
FIGS=/root/vast/eric/sleap-3d-gui/scratch/repos/lucid/figs
export POSE_DIR=/root/vast/eric/lucid-main-md56/pose
export PATH=/root/vast/eric/luc3d-bench/video_env/bin:$PATH
# bpy 5.2 needs X11 client libs this container lacks; symlinks into miniforge's pkgs cache
export LD_LIBRARY_PATH=/tmp/claude-0/-root-vast-eric-sleap-3d-gui-scratch-repos-lucid/4d4f3317-8497-49a9-8c88-e58530b1c39e/scratchpad/bpylibs
cd $FIGS/blender-images
PY=bpyenv/bin/python
read AZ EL < <(python3 -c "import json;d=json.load(open('$FIGS/out/fig2a_scene.json'));print(d['suggested_azim_deg'],d['suggested_elev_deg'])")

echo "== Fig 3D idswitch $(date -u +%T)";  $PY idswitch_fig_scene.py
echo "== Fig 3D style $(date -u +%T)";     (cd $FIGS && .venv/bin/python idswitch_fig_style.py)
echo "== Fig 1d pose $(date -u +%T)";      $PY fig1d_scene.py --mode pose --samples 200
echo "== Fig 1d rig $(date -u +%T)";       $PY fig1d_scene.py --mode rig --samples 200
echo "== Fig 2A pose az=$AZ el=$EL $(date -u +%T)"
$PY fig1d_scene.py --mode pose --scene $FIGS/out/fig2a_scene.json --azim $AZ --elev $EL \
    --aspect 1.19 --out renders/fig2a_pose.png
echo "== Fig 1a Mouse-Dyad $(date -u +%T)"
$PY bmimica_scene.py --res 1280 960 --ortho 2.15 --out renders/fig1_renders/mouse-dyad_f55701.png
echo "== Fig 5A $(date -u +%T)"
$PY fig5a_scene.py --azim 255 --elev 22 --ortho 0.335 --box 230 230 140 --samples 200 \
    --res 1900 1900 --out renders/fig5a_upright.png
echo "== done $(date -u +%T)"
