#!/usr/bin/env bash
# Fig 2F apples-to-apples (2026-10-15): Anipose linear + optim accuracy on EXACTLY the keypoints
# LUC3D is scored on (FIG2_OPTIM_PAD=1). New deposit fig4_anipose_md56pad.json; nothing overwritten.
set -uo pipefail
cd /root/vast/eric/sleap-3d-gui/scratch/repos/lucid
L=figs/out/md56_logs; O=figs/out; ST=figs/figures/drafts/RUN-STATUS-2026-10-15.md
st() { echo "- $(date -u +'%F %T') UTC $(hostname -s) [main box] $*" | tee -a $ST; }
[ -e $O/fig4_anipose_md56pad.json ] && { st "F2PAD FAILED: deposit exists"; exit 1; }
export PATH=/root/vast/eric/node26_env/bin:$PATH POSE_DIR=/root/vast/eric/lucid-main-md56/pose FIG2_SUFFIX=md56 FIG2_OPTIM_PAD=1 NUMBA_NUM_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
st "F2PAD start: Anipose accuracy on LUC3D's exact keypoints (pad, not truncate)"
/root/vast/eric/luc3d-bench/anipose072_cv413_env/bin/python figs/fig2_solvers_anipose.py --jobs 12 --optim-jobs 12 \
   --out $O/fig4_anipose_md56pad.json > $L/F2PAD.log 2>&1 || { st "F2PAD FAILED: $(tail -2 $L/F2PAD.log | tr '\n' ' ')"; exit 1; }
python3 - > $L/F2PAD_check.txt 2>&1 <<'PY'
import json, sys
a = json.load(open("figs/out/fig4_anipose_md56pad.json")); l = json.load(open("figs/out/fig4_md56.json"))
ln = {p["session"]: p["n_keypoints"] for p in l["per_session"]}
po = {p["session"]: p for p in a["per_session_optim"]}
bad = [s for s in ln if s not in po or po[s]["n"] != ln[s]]
lin = {p["session"]: p for p in a["per_session"]}
mism = [s for s in ln if abs(po[s]["linear_reproj_p50"] - lin[s]["reproj_p50"]) > 1e-12]
print("SESSIONS", len(ln), "optim n == LUC3D n:", len(ln) - len(bad), "bad", bad[:5])
print("optim-file linear median == per_session linear median:", len(ln) - len(mism), "bad", mism[:5])
sys.exit(1 if bad or mism else 0)
PY
rc=$?; st "F2PAD $( [ $rc -eq 0 ] && echo DONE || echo 'FAILED check' ): $(head -2 $L/F2PAD_check.txt | tr '\n' ' ')"
