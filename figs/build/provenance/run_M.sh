#!/usr/bin/env bash
# md56 phase M (2026-10-14): every tracker-dependent number re-run on origin/main 2dc001fe's REAL
# tracker (fig3_bench.mjs + LUCID_ROOT), replacing the fork (fig6_bench.mjs + xv_experimental.js),
# which matched main on only 47/56 Mouse-Dyad and 36/74 SLAP-2M sessions. Fork deposits kept
# in md56_logs/*_md56.fork.json. Fig 3G/H sweeps run on a separate thread (ericmimicc).
set -uo pipefail
cd /root/vast/eric/sleap-3d-gui/scratch/repos/lucid
L=figs/out/md56_logs; MAIN=/root/vast/eric/lucid-main-md56
export PATH=/root/vast/eric/node26_env/bin:$PATH SCORE_NOTAIL=1
export PYTHONPATH=/tmp/claude-0/-root-vast-eric-sleap-3d-gui-scratch-repos-lucid/44adbb8d-7377-41ea-ab2c-5dae7a320d17/scratchpad/cv2h
PY=/root/vast/eric/luc3d-bench/liezl_env/bin/python
t() { echo "[$(date +%T)] $* (load $(cut -d' ' -f1 /proc/loadavg))"; }
t "start, main $(git -C $MAIN rev-parse --short HEAD)"

# M1 Supp Fig 6A/6F + eviction text: Mouse-Dyad configs on main (shipped = labelled pre-#210 arm)
FIG8M_LUCID_ROOT=$MAIN FIG8M_SUFFIX=md56 FIG8M_WORKERS=24 $PY figs/fig6_methods.py --all-sessions \
  --configs shipped,sync_stale1_dist25,sync_stale10_dist25,sync_stale20_dist25,sync_stale30_dist25,stale0_dist25_corr36,stale0_dist25 > $L/M1_fig8m.log 2>&1
t "M1 fig8m exit=$?"
$PY - <<'PY'
import sys, json; sys.path.insert(0, "figs")
import fig3_hh_freshanchor as fa
from pathlib import Path
a = Path("figs/out/tmp/fig8m50_2dc001fe/sync_stale20_dist25"); b = Path("figs/out/tmp/md_main2dc001fe")
r = [fa.payload_digest(a / p.name) == fa.payload_digest(p) for p in sorted(b.glob("2025*.json"))]
print(f"[M1 check] fig6_methods-on-main sync_stale20_dist25 vs plain main defaults: {sum(r)}/{len(r)} identical")
PY
FIG8M_SUFFIX=md56 $PY figs/fig6_report50.py > $L/M1_report50.log 2>&1

# M2 camera subsets on main
LUCID_ROOT=$MAIN CAMS_SUFFIX=md56 CAMS_HH_CACHE=figs/out/tmp/headtohead_md56/A2_C5_bmimica $PY figs/fig2_cams_identity.py --stage track --workers 24 > $L/M2_cams_track.log 2>&1
t "M2 cams track exit=$?"
LUCID_ROOT=$MAIN CAMS_SUFFIX=md56 CAMS_HH_CACHE=figs/out/tmp/headtohead_md56/A2_C5_bmimica $PY figs/fig2_cams_identity.py --stage score --workers 24 > $L/M2_cams_score.log 2>&1
t "M2 cams score exit=$?"

# M3 Fig 4D: score main's SLAP-2M tracking (already tracked by fig4_main_check.py)
$PY figs/fig4_slap2m.py --pool lite_bvis5 --configs sync_stale20_dist25 --tag main2dc001fe --workers 24 > $L/M3_fig4d.log 2>&1
t "M3 fig4d exit=$?"
mv figs/out/fig9_slap2m_lite_bvis5_main2dc001fe_notailscore.json figs/out/fig9_slap2m_lite_bvis5_notailscore_md56.json && t "M3 installed as fig9_slap2m_lite_bvis5_notailscore_md56.json"

# M4 Supp Fig 6A variant block (Mouse-Dyad) then M5 Supp Fig 6B-D SLAP-2M fair arms with main's LUC3D
$PY figs/fig6_variant_best_rebuild.py --tag md56 --base bvis5nt > $L/M4_variant.log 2>&1; t "M4 variant rebuild exit=$? $(tail -n 1 $L/M4_variant.log)"
FAIR_VARIANT=bvis5 FAIR_LUC3D_DIR=figs/out/tmp/fig9slap_lite_bvis5_main2dc001fe/sync_stale20_dist25 \
  FAIR_REF_JSON=fig9_slap2m_lite_bvis5_notailscore_md56.json FAIR_OUT_SUFFIX=md56 \
  $PY figs/fig6_slap2m_fair.py --stage score --workers 32 > $L/M5_slapfair_score.log 2>&1; t "M5 slap fair score exit=$?"
FAIR_VARIANT=bvis5 FAIR_LUC3D_DIR=figs/out/tmp/fig9slap_lite_bvis5_main2dc001fe/sync_stale20_dist25 \
  FAIR_REF_JSON=fig9_slap2m_lite_bvis5_notailscore_md56.json FAIR_OUT_SUFFIX=md56 \
  $PY figs/fig6_slap2m_fair.py --stage install > $L/M5_slapfair_install.log 2>&1; t "M5 slap fair install exit=$?"
t "PHASE M DONE"
