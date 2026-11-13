#!/usr/bin/env bash
# md56 phase F2a (2026-10-14): Fig 2C/E/F accuracy on all 56 Mouse-Dyad-10M sessions.
# FULL rerun (the keypoint pool spans sessions), against ONE pinned checkout of the shipped
# code: origin/main 2dc001fe at /root/vast/eric/lucid-main-md56. The base run used the
# 2026-08-04 eric/bundle-adj pose/. Outputs are out/fig4*_md56.* (FIG2_SUFFIX); no base deposit
# is touched. The timed parts (Fig 2G, the anipose arm) are phase F2b, on a quiet machine.
set -uo pipefail
cd /root/vast/eric/sleap-3d-gui/scratch/repos/lucid
L=figs/out/md56_logs
export FIG2_SUFFIX=md56 POSE_DIR=/root/vast/eric/lucid-main-md56/pose
export PATH=/root/vast/eric/node26_env/bin:$PATH
export PYTHONPATH=/tmp/claude-0/-root-vast-eric-sleap-3d-gui-scratch-repos-lucid/44adbb8d-7377-41ea-ab2c-5dae7a320d17/scratchpad/cv2h
LP=/root/vast/eric/luc3d-bench/lp3d_env/bin/python
t() { echo "[$(date +%T)] $*"; }

t "start node $(node --version) pose $(git -C /root/vast/eric/lucid-main-md56 rev-parse --short HEAD)"
[ -s figs/out/fig4_input_md56.bin ] || { $LP figs/fig2_solvers_export.py --stride 15 > $L/F2_export.log 2>&1; t "export exit=$?"; }
python3 - <<'EOF' || exit 1
import json; d = json.load(open("figs/out/fig4_input_md56.json"))
print("export sessions", len(d["sessions"]), "skipped", d["skipped"], "keypoints", d["keypoints"])
assert len(d["sessions"]) == 56, "export does not cover all 56 sessions"
EOF

( for r in 0:5 5:9 9:13 13:17 17:21 21:25 25:29 29:33 33:37 37:41 41:45 45:50 50:53 53:56; do
    BLOCKS=$r STRIDE=1 OUT_JSON=figs/out/tmp/bv_md56-${r/:/-}.json node figs/fig2_solvers_by_views.mjs > $L/F2_bv-${r/:/-}.log 2>&1 &
  done; wait
  node figs/fig2_solvers_by_views.mjs --merge figs/out/tmp/bv_md56-*.json > $L/F2_bv_merge.log 2>&1
  t "by_views merge exit=$?" ) &

HOV_SUB=1 DSTEP=1 node figs/fig2_solvers_measure.mjs > $L/F2_measure.log 2>&1; t "measure exit=$?"
wait
node figs/fig2_solvers_robust_sessions.mjs > $L/F2_robust.log 2>&1; t "robust exit=$? $(tail -n 1 $L/F2_robust.log)"
t "F2a DONE"
