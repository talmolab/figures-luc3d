#!/usr/bin/env bash
# md56 phase B (2026-10-14): build fig2b / fig3b / fig6b from the md56 deposits -- ONLY if
# every Mouse-Dyad-10M deposit exists as _md56 and covers all 56 sessions (figs/md56_check.py),
# and in strict mode (FIG_VARIANT_REQUIRE): a required deposit without its _md56 file stops the
# panel instead of silently falling back to the 50-session base file. Base figures untouched.
set -uo pipefail
cd /root/vast/eric/sleap-3d-gui/scratch/repos/lucid
L=figs/out/md56_logs
PY=figs/.venv/bin/python
t() { echo "[$(date +%T)] $*"; }
until grep -q "PHASE Q DONE" $L/run_Q.out 2>/dev/null; do sleep 60; done
t "phase Q done; waiting for the ericmimicc sweeps (run_S.out)"
until grep -q "PHASE S DONE" $L/run_S.out 2>/dev/null; do sleep 60; done
t "phase S done"

$PY figs/md56_check.py > $L/B_gate.log 2>&1
if [ $? -ne 0 ]; then t "GATE FAILED -- not building; see $L/B_gate.log"; cat $L/B_gate.log; exit 1; fi
t "gate passed"

REQ=$($PY - <<'EOF'
import sys; sys.path.insert(0, "figs")
import md56_check as m
print(",".join(n.replace("_md56", "") for n in m.CHECKS))
EOF
)
rm -f $L/B_loads.tsv
export FIG_VARIANT=md56 FIG_TAG=b FIG_VARIANT_REQUIRE="$REQ" FIG_VARIANT_LOG=$PWD/$L/B_loads.tsv
$PY figs/make_figures.py 2 3 6 > $L/B_make_figures.log 2>&1; t "make_figures exit=$?"
echo "--- panels that failed:"; grep -B1 "    ! " $L/B_make_figures.log | head -40
echo "--- deposits read from the BASE file (no md56 variant; must all be non-Mouse-Dyad or already 56):"
awk -F'\t' '$3 !~ /_md56/' $L/B_loads.tsv | cut -f2,3 | sort | uniq -c
ls -la figs/figures/fig2b figs/figures/fig3b figs/figures/fig6b 2>&1 | head -40
t "PHASE B DONE"
