#!/usr/bin/env bash
# B3STALE (2026-10-16, Eric): Supp 6F staleness horizon past 30 on b3 -- stale 40 and 50.
# stale 30 (cached) is re-scored as an anchor: it must equal fig8_methods_50_b3.json exactly.
# Writes only fig8_methods_50_b3stale.json and new config dirs in tmp/fig8m50_2dc001fe_b3/.
set -uo pipefail
cd /root/vast/eric/sleap-3d-gui/scratch/repos/lucid
O=figs/out; L=$O/md56_logs; ST=figs/figures/drafts/RUN-STATUS-2026-10-15.md
PY=/root/vast/eric/luc3d-bench/liezl_env/bin/python
export PATH=/root/vast/eric/node26_env/bin:$PATH SCORE_NOTAIL=1 MD_CFG=b3 FIG8M_SUFFIX=b3stale \
       FIG8M_LUCID_ROOT=/root/vast/eric/lucid-main-md56 FIG8M_WORKERS=40
st() { echo "- $(date -u '+%F %T') UTC $(hostname -s) [main box] $* (load $(cut -d' ' -f1 /proc/loadavg))" >> $ST; echo "$*"; }
[ -e $O/fig8_methods_50_b3stale.json ] && { st "B3STALE FAILED: deposit exists"; exit 1; }
st "B3STALE start: b3 stale 40 + 50 (+ stale 30 anchor)"
$PY figs/fig6_methods.py --all-sessions --configs sync_stale30_dist25,sync_stale40_dist25,sync_stale50_dist25 > $L/B3STALE.log 2>&1 \
  || { st "B3STALE FAILED: exit $? ($(tail -1 $L/B3STALE.log))"; exit 1; }
$PY - >> $L/B3STALE.log 2>&1 <<'PY' || { st "B3STALE FAILED: anchor gate $(tail -1 $L/B3STALE.log)"; exit 1; }
import json, sys
n = {c["config"]: c for c in json.load(open("figs/out/fig8_methods_50_b3stale.json"))["cells"]}
o = {c["config"]: c for c in json.load(open("figs/out/fig8_methods_50_b3.json"))["cells"]}
ok = abs(n["sync_stale30_dist25"]["idf1_cross"] - o["sync_stale30_dist25"]["idf1_cross"]) < 1e-12 and n["sync_stale30_dist25"]["switches"] == o["sync_stale30_dist25"]["switches"]
rows = [(k, (n.get(k) or o[k])) for k in ("stale0_dist25","sync_stale1_dist25","sync_stale10_dist25","sync_stale20_dist25","sync_stale30_dist25","sync_stale40_dist25","sync_stale50_dist25")]
print("RESULT " + " | ".join(f"{k.replace('sync_','').replace('_dist25','')} IDF1 {c['idf1_cross']:.4f} sw {c['switches']}" for k, c in rows) + f" | anchor stale30 identical: {ok}")
sys.exit(0 if ok else 1)
PY
st "B3STALE DONE: $(grep '^RESULT' $L/B3STALE.log)"
