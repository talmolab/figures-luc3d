#!/usr/bin/env bash
# LOWR + GATE288 (2026-10-16, Eric).
#  LOWR: Fig 3b G/H sweep extended below r = 0.25 (0.001..0.15), both lines, b3. The cached
#        r >= 0.25 cells are reused, so the deposits are rewritten as a superset.
#        Pre-run copies: figs/out/md56_logs/pre_lowr_backup/.
#  GATE288: matchGate=1 on Mouse-Dyad b3 (r = 6) and s-DANNCE C1_sigma0, then a markdown report.
set -uo pipefail
cd /root/vast/eric/sleap-3d-gui/scratch/repos/lucid
PY=/root/vast/eric/envs/md56_score_env/bin/python
L=figs/out/md56_logs; ST=figs/figures/drafts/RUN-STATUS-2026-10-15.md
export PATH=/root/vast/eric/node26_env/bin:$PATH SCORE_NOTAIL=1 MD_CFG=b3 SWEEP50_SUFFIX=b3 FIG8M_SUFFIX=b3 \
       SWEEP50_LUCID_ROOT=/root/vast/eric/lucid-main-md56 FIG8M_LUCID_ROOT=/root/vast/eric/lucid-main-md56
st() { echo "- $(date -u '+%F %T') UTC $(hostname -s) [main box] $* (load $(cut -d' ' -f1 /proc/loadavg))" >> $ST; echo "$*"; }
G="--grid-corr3d 0,0.001,0.002,0.005,0.01,0.02,0.05,0.1,0.15,0.25,0.5,1,2,3,4,6,8,12,16,24 --grid-corr2d 1"
st "LOWR+GATE288 start: 3b G/H low-r sweep (both lines) + matchGate=1 checks, v1.0.5 $(git -C $SWEEP50_LUCID_ROOT rev-parse --short HEAD)"
$PY figs/fig3_sweep50.py $G --thresholds '{"distanceThreshold":25,"stale":20}' --workers 20 > $L/LOWR_sweep_current.log 2>&1 & P1=$!
$PY figs/fig3_sweep50.py $G --thresholds '{"distanceThreshold":50,"stale":0}'  --workers 20 > $L/LOWR_sweep_prevsettings.log 2>&1 & P2=$!
# gate, Mouse-Dyad (own cache dir + deposit: thresholds digest differs)
$PY figs/fig3_sweep50.py --grid-corr3d 6 --grid-corr2d 1 --thresholds '{"distanceThreshold":25,"stale":20,"matchGate":1}' \
    --workers 12 > $L/GATE288_mousedyad.log 2>&1; E3=$?; st "GATE288 Mouse-Dyad exit=$E3"
# gate, s-DANNCE (same as SDV105's C1_sigma0 except the --params file)
echo '{"thresholds":{"matchGate":1}}' > $L/GATE288_sdannce.params.json
(cd figs/fig10-bench && FIG10_PARAMS=$PWD/../out/md56_logs/GATE288_sdannce.params.json \
   LUCID_SC_ROOT=/root/vast/eric/lucid-main-md56 FIG10_RESULTS=$PWD/results_v105_gate1 FIG10_WORK=$PWD/work_v105_gate1 \
   ../.venv/bin/python fig10_run.py --datasets triads,bedding,soc1,soc3 --cells C1_sigma0 --jobs 12) > $L/GATE288_sdannce.log 2>&1
E4=$?; st "GATE288 s-DANNCE exit=$E4 ($(grep -c '^OK' $L/GATE288_sdannce.log) OK lines)"
wait $P1; E1=$?; wait $P2; E2=$?
st "LOWR sweeps exit current=$E1 prev=$E2"
[ $E1 -eq 0 ] && [ $E2 -eq 0 ] || { st "LOWR FAILED"; }
# report needs the current-line deposit (gate-0 r=6 cell) -> only after P1
$PY figs/gate288_report.py > $L/GATE288_report.log 2>&1 && st "GATE288 DONE: $(grep ^RESULT $L/GATE288_report.log)" \
  || st "GATE288 report FAILED: $(tail -1 $L/GATE288_report.log)"
# gate: old r >= 0.25 cells must be unchanged in the rewritten deposits
$PY - > $L/LOWR_gate.log 2>&1 <<'PY'
import json
bad = []
for n in ("fig3_sweep50__distanceThreshold25-stale20_f7733dbf_b3.json", "fig3_sweep50__distanceThreshold50-stale0_ce1d7f8d_b3.json"):
    o = {c["corr3d"]: c for c in json.load(open(f"figs/out/md56_logs/pre_lowr_backup/{n}"))["cells"]}
    d = {c["corr3d"]: c for c in json.load(open(f"figs/out/{n}"))["cells"]}
    for r, c in o.items():
        if json.dumps(c["per_session"], sort_keys=True) != json.dumps(d[r]["per_session"], sort_keys=True): bad.append((n[:40], r))
    print(n, "new r:", sorted(set(d) - set(o)))
print("GATE LOWR old cells identical:", not bad, bad)
PY
st "LOWR DONE: $(tail -1 $L/LOWR_gate.log). Deposits ready for the fig3b G/H rebuild (not rebuilt)."
