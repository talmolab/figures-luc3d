#!/usr/bin/env bash
# LUC3D-JS plan J2 + J3 (2026-10-20, eric-mimicc-0-1): every Fig 2 deposit rebuilt on an
# export whose geometry is LUC3D v1.0.5's JavaScript (fig2_solvers_export.py ->
# lucid_geom.mjs), tagged FIG2_SUFFIX=js so nothing current is overwritten.
#   1. export (stride 15, every session)            -> out/fig4_input_js.{json,bin}
#   2. by_views 14 blocks + merge (2C, 2E top, 2B, 2D) || measure (2F LUC3D, 2.41 px)
#   3. robust sessions (2E bottom)                  -> out/fig4_robust_sessions_js.json
#   4. Anipose on LUC3D's exact keypoints (2F)      -> out/fig4_anipose_jspad.json + check
# Every driver asserts POSE_DIR is the clean v1.0.5 checkout (lucid_v105) and stamps it.
set -uo pipefail
cd /root/vast/eric/sleap-3d-gui/scratch/repos/lucid
L=figs/out/md56_logs; O=figs/out
export PATH=/root/vast/eric/node26_env/bin:$PATH POSE_DIR=/root/vast/eric/lucid-main-md56/pose FIG2_SUFFIX=js
LP=/root/vast/eric/luc3d-bench/lp3d_env/bin/python
t() { echo "[$(date -u +%T)] $*"; }
fail() { t "J_FIG2 FAILED: $*"; exit 1; }
for f in fig4_input_js.json fig4_by_views_js.json fig4_js.json fig4_robust_sessions_js.json fig4_anipose_jspad.json; do
  [ -e $O/$f ] && fail "$O/$f exists"
done
t "start node $(node --version) pose $(git -C /root/vast/eric/lucid-main-md56 rev-parse --short HEAD)"

$LP figs/fig2_solvers_export.py --stride 15 > $L/J_export.log 2>&1 || fail "export: $(tail -2 $L/J_export.log | tr '\n' ' ')"
t "export: $(grep '^\[json\]' $L/J_export.log)"

mkdir -p $O/tmp/bv_js
( for r in 0:5 5:9 9:13 13:17 17:21 21:25 25:29 29:33 33:37 37:41 41:45 45:50 50:53 53:56; do
    BLOCKS=$r STRIDE=1 OUT_JSON=$O/tmp/bv_js/bv-${r/:/-}.json node figs/fig2_solvers_by_views.mjs > $L/J_bv-${r/:/-}.log 2>&1 &
  done; wait
  node figs/fig2_solvers_by_views.mjs --merge $O/tmp/bv_js/bv-*.json > $L/J_bv_merge.log 2>&1
  t "by_views merge exit=$?" ) &
HOV_SUB=1 DSTEP=1 node figs/fig2_solvers_measure.mjs > $L/J_measure.log 2>&1; t "measure exit=$?"
wait
[ -s $O/fig4_by_views_js.json ] || fail "no fig4_by_views_js.json"
[ -s $O/fig4_js.json ] || fail "no fig4_js.json"

node figs/fig2_solvers_robust_sessions.mjs > $L/J_robust.log 2>&1 || fail "robust: $(tail -2 $L/J_robust.log | tr '\n' ' ')"
t "robust: $(grep -v -i deprecat $L/J_robust.log | tail -1)"

NUMBA_NUM_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 FIG2_OPTIM_PAD=1 \
  /root/vast/eric/luc3d-bench/anipose072_cv413_env/bin/python figs/fig2_solvers_anipose.py --jobs 12 --optim-jobs 12 \
  --out $O/fig4_anipose_jspad.json > $L/J_anipose.log 2>&1 || fail "anipose: $(tail -2 $L/J_anipose.log | tr '\n' ' ')"
python3 - > $L/J_anipose_check.txt 2>&1 <<'PY'
import json, sys
a = json.load(open("figs/out/fig4_anipose_jspad.json")); l = json.load(open("figs/out/fig4_js.json"))
ln = {p["session"]: p["n_keypoints"] for p in l["per_session"]}
po = {p["session"]: p for p in a["per_session_optim"]}
bad = [s for s in ln if s not in po or po[s]["n"] != ln[s]]
lin = {p["session"]: p for p in a["per_session"]}
mism = [s for s in ln if abs(po[s]["linear_reproj_p50"] - lin[s]["reproj_p50"]) > 1e-12]
print("SESSIONS", len(ln), "optim n == LUC3D n:", len(ln) - len(bad), "bad", bad[:5])
print("optim-file linear median == per_session linear median:", len(ln) - len(mism), "bad", mism[:5])
sys.exit(1 if bad or mism else 0)
PY
t "anipose check exit=$? $(head -1 $L/J_anipose_check.txt)"
t "J_FIG2 DONE"
