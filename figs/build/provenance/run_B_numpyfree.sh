#!/usr/bin/env bash
# New b series (2026-10-20, eric-mimicc-0-1). Eric: "get rid of all the numpy triangulation
# from all the figures, we need to make sure to use luc3d triangulations and rerender
# everything" / "youll have to make a new b series for the figs so you dont overwrite what we
# have". Figs 1, 2, 3, 5 rebuilt as FIG_TAG=b from x7's inputs (the deposit set the main figures
# were built from) plus Blender renders made from LUC3D v1.0.5 output:
#   HardFight (Fig 1d pose + rig, Fig 2A pose, Fig 3D): out/hardfight_luc3d.npz (LUC3D tracker +
#     DLT over all 36,000 frames, figs/hardfight_luc3d.py), floor alignment fitted to it;
#   Mouse-Dyad (Fig 1a right tile): camera placement aligned on LUC3D's DLT (lucid_geom);
#   Fig 5A render: fig5_views_js.json (LUC3D-aligned) instead of the August fig5_views.json.
# Writes only *_b deposits/renders and figures/fig{1,2,3,5}b + data/fig{1,2,3,5}b. Refuses to
# start if any of them exists. Main figures, renders and deposits are not touched.
set -uo pipefail
R=/root/vast/eric/sleap-3d-gui/scratch/repos/lucid; F=$R/figs; cd $F
L=out/md56_logs; FPY=.venv/bin/python; BPY=blender-images/bpyenv/bin/python
LP3D=/root/vast/eric/luc3d-bench/lp3d_env/bin/python
fail() { echo "B FAILED: $*"; exit 1; }
t() { echo "== $(date -u +%T) $*"; }
export FIG_TAG=b POSE_DIR=/root/vast/eric/lucid-main-md56/pose
export PATH=/root/vast/eric/luc3d-bench/video_env/bin:$PATH      # ffmpeg for the Fig 3D photos
# bpy 5.2 needs X11 client libs this container lacks (symlinks into miniforge's pkgs cache)
export LD_LIBRARY_PATH=/tmp/claude-0/-root-vast-eric-sleap-3d-gui-scratch-repos-lucid/4d4f3317-8497-49a9-8c88-e58530b1c39e/scratchpad/bpylibs

BOUT="out/fig1_hardfight_scene_b.json out/fig2a_scene_b.json out/fig1_bmimica_scene_b.json
      blender-images/renders/hardfight_alignment_b.json blender-images/renders/idswitch_fig_data_hardfight_b.json
      blender-images/renders/fig1d_pose_b.png blender-images/renders/fig1d_rig_b.png
      blender-images/renders/fig2a_pose_b.png blender-images/renders/fig5a_upright_b.png
      blender-images/renders/fig1_renders/mouse-dyad_f55701_b.png
      figures/drafts/figs/lucid_idswitch_style_hardfight_b.png"
for f in $BOUT; do [ -e $f ] && fail "$f exists"; done
for n in 1 2 3 5; do { [ -e figures/fig${n}b ] || [ -e data/fig${n}b ]; } && fail "figures/fig${n}b or data/fig${n}b exists"; done
python3 -c "import json,sys;d=json.load(open('out/hardfight_luc3d.json'));sys.exit(d['lucid_commit']!='2dc001fe527c4dbaad17fa171da132cfdfe010e6')" \
  || fail "out/hardfight_luc3d.json is not LUC3D v1.0.5"
# fingerprint of the main figures' inputs, to prove afterwards nothing was overwritten
MAIN="out/fig1_hardfight_scene.json out/fig2a_scene.json out/fig1_bmimica_scene.json
      blender-images/renders/hardfight_alignment.json blender-images/renders/fig1d_pose.png
      blender-images/renders/fig1d_rig.png blender-images/renders/fig2a_pose.png
      blender-images/renders/fig5a_upright.png blender-images/renders/fig1_renders/mouse-dyad_f55701.png
      figures/drafts/figs/lucid_idswitch_style_hardfight.png"
sha256sum $MAIN > $L/B_main_inputs_before.sha256

# --- 1. scene deposits (LUC3D 3D / identities / alignment)
t "Fig 1d scene";  $BPY fig1_hardfight_scene.py --out out/fig1_hardfight_scene_b.json > $L/B_fig1_hardfight_scene.log 2>&1 || fail "fig1_hardfight_scene"
t "Fig 2A scene";  $BPY fig2a_scene.py --out out/fig2a_scene_b.json > $L/B_fig2a_scene.log 2>&1 || fail "fig2a_scene"
t "Fig 1a scene";  $LP3D fig1_bmimica_scene.py --frame 55701 --out out/fig1_bmimica_scene_b.json > $L/B_fig1_bmimica_scene.log 2>&1 || fail "fig1_bmimica_scene"
t "Fig 3D prep";   (cd blender-images && bpyenv/bin/python idswitch_fig_prep.py) > $L/B_idswitch_prep.log 2>&1 || fail "idswitch prep"

# --- 2. renders (each one's documented shipped recipe, _b in and out)
read AZ EL < <(python3 -c "import json;d=json.load(open('out/fig2a_scene_b.json'));print(d['suggested_azim_deg'],d['suggested_elev_deg'])")
cd blender-images
t "Fig 3D render";  bpyenv/bin/python idswitch_fig_scene.py > ../$L/B_idswitch_scene.log 2>&1 || fail "idswitch render"
t "Fig 1d pose";    bpyenv/bin/python fig1d_scene.py --mode pose --samples 200 --scene ../out/fig1_hardfight_scene_b.json \
                      --out renders/fig1d_pose_b.png > ../$L/B_fig1d_pose.log 2>&1 || fail "fig1d pose"
t "Fig 1d rig";     bpyenv/bin/python fig1d_scene.py --mode rig --samples 200 --scene ../out/fig1_hardfight_scene_b.json \
                      --out renders/fig1d_rig_b.png > ../$L/B_fig1d_rig.log 2>&1 || fail "fig1d rig"
t "Fig 2A pose az=$AZ el=$EL"
bpyenv/bin/python fig1d_scene.py --mode pose --scene ../out/fig2a_scene_b.json --azim $AZ --elev $EL --aspect 1.19 \
    --out renders/fig2a_pose_b.png > ../$L/B_fig2a_pose.log 2>&1 || fail "fig2a pose"
t "Fig 1a Mouse-Dyad"
bpyenv/bin/python bmimica_scene.py --scene ../out/fig1_bmimica_scene_b.json --res 1280 960 --ortho 2.15 \
    --out renders/fig1_renders/mouse-dyad_f55701_b.png > ../$L/B_bmimica.log 2>&1 || fail "bmimica render"
t "Fig 5A"
bpyenv/bin/python fig5a_scene.py --azim 255 --elev 22 --ortho 0.335 --box 230 230 140 --samples 200 \
    --res 1900 1900 --out renders/fig5a_upright_b.png > ../$L/B_fig5a.log 2>&1 || fail "fig5a render"
cd $F
t "Fig 3D style";   $FPY idswitch_fig_style.py > $L/B_idswitch_style.log 2>&1 || fail "idswitch style"
for f in $BOUT; do [ -s $f ] || fail "missing $f"; done

# --- 3. figures: x7's environment and inputs, FIG_TAG=b, b renders required
export FIG6F_NOEVICT=stale0_dist25 SIGN_TIES_EXCLUDED=1 FIG5G_PERDIR=1 F3_TRUE_MEDIAN=1 FIG3F_CLEAN=1 \
       FIG6_SDANNCE_AGG=$F/fig10-bench/results_v105/agg
export FIG_VARIANT=x7 FIG_VARIANT_LOG=$PWD/$L/B_loads.tsv FIG_RENDER_REQUIRE=1
export FIG_VARIANT_REQUIRE=$(cut -f1 $L/X7_links.tsv | paste -sd, -)
rm -f $L/B_loads.tsv
for n in 1 2 3 5; do
  $FPY make_figures.py $n > $L/B_fig$n.log 2>&1; E=$?
  NF=$(grep -c "    ! " $L/B_fig$n.log || true)
  t "fig${n}b exit=$E failed_panels=$NF $(tail -n1 $L/B_fig$n.log)"
  [ $E -eq 0 ] && [ "$NF" = 0 ] || fail "fig$n (see $L/B_fig$n.log)"
done
echo "--- renders used:"; grep -h "\[render b\]" $L/B_fig*.log | sort -u
echo "--- loads that did NOT come from an x7 link:"; awk -F'\t' '$3 !~ /_x7\./' $L/B_loads.tsv | cut -f2,3 | sort | uniq -c

# --- 4. what moved, and proof the main inputs did not
for n in 1 2 3 5; do echo "=== data/fig${n}b vs data/fig$n"; for f in data/fig${n}b/*.csv; do
  b=$(basename $f); [ -e data/fig$n/$b ] || { echo "NEW $b"; continue; }; cmp -s $f data/fig$n/$b || echo "DIFF $b"; done; done
sha256sum -c --quiet $L/B_main_inputs_before.sha256 && echo "MAIN INPUTS UNCHANGED" || fail "a main input changed"
git -C $R status --short -- figs/figures/fig1 figs/figures/fig2 figs/figures/fig3 figs/figures/fig5 figs/data/fig1 figs/data/fig2 figs/data/fig3 figs/data/fig5 | grep . && fail "main figures changed"
echo "B BUILD DONE"
