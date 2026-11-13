#!/usr/bin/env bash
# md56 phase B2 (2026-10-14): rebuild fig2b, fig4b, fig5b, fig6b after phase M (all on main
# 2dc001fe), gated (md56_check.py: 56 Mouse-Dyad / 74 SLAP-2M sessions) and in strict mode.
# fig3b waits for the main-run Fig 3G/H sweeps (separate thread) and the Fig 3E switch.
set -uo pipefail
cd /root/vast/eric/sleap-3d-gui/scratch/repos/lucid
L=figs/out/md56_logs; PY=figs/.venv/bin/python
t() { echo "[$(date +%T)] $*"; }
until grep -q "PHASE M DONE" $L/run_M.out 2>/dev/null; do sleep 60; done
$PY figs/md56_check.py > $L/B2_gate.log 2>&1 || { t "GATE FAILED"; cat $L/B2_gate.log; exit 1; }
t "gate passed"
REQ=$($PY -c "import sys; sys.path.insert(0,'figs'); import md56_check as m; print(','.join(n.replace('_md56','') for n in list(m.CHECKS)+list(m.SLAP)))")
rm -f $L/B2_loads.tsv
export FIG_VARIANT=md56 FIG_TAG=b FIG_VARIANT_REQUIRE="$REQ" FIG_VARIANT_LOG=$PWD/$L/B2_loads.tsv
for f in 2 4 5 6; do $PY figs/make_figures.py $f > $L/B2_fig$f.log 2>&1; t "fig${f}b exit=$? $(tail -n 1 $L/B2_fig$f.log)"; done
rm -f figs/figures/fig6b/fig6ba_within_vs_cross.pdf figs/figures/fig6b/fig6ba_within_vs_cross.png figs/data/fig6b/fig6a_within_vs_cross.csv
echo "--- read from BASE (no md56 file):"; awk -F'\t' '$3 !~ /_md56/' $L/B2_loads.tsv | cut -f2,3 | sort | uniq -c
t "PHASE B2 DONE"
