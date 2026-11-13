#!/usr/bin/env bash
# J16 (LUC3D-JS plan, 2026-10-20, eric-mimicc-0-1): Fig 2G timing sitting on the LUC3D-built
# export (fig4_input_js), same protocol as T2: every solver on the same first 13,334 frames x
# 15 joints = 200,010 keypoints, 5 interleaved rounds, single-threaded, best round.
# LUC3D = fig2_solvers_time_luc3d.mjs importing v1.0.5 pose/ (lucid_v105 asserts the clean
# 2dc001fe checkout and the deposit records it). Anipose = aniposelib 0.7.2 (baseline).
# NOTHING ELSE OF OURS RUNS: figs/timing_watch.py measures every process's CPU from /proc,
# refuses to start if anything outside this sitting uses > 0.5 core or looks like ours, and
# watches the whole sitting. Its verdict goes into the deposit.
set -uo pipefail
cd /root/vast/eric/sleap-3d-gui/scratch/repos/lucid
L=figs/out/md56_logs; O=figs/out
AP=/root/vast/eric/luc3d-bench/anipose072_cv413_env/bin/python
export PATH=/root/vast/eric/node26_env/bin:$PATH NODE=/root/vast/eric/node26_env/bin/node
export POSE_DIR=/root/vast/eric/lucid-main-md56/pose FIG2_SUFFIX=js FIG2G_FRAMES=13334
export NUMBA_NUM_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 UV_THREADPOOL_SIZE=1
t() { echo "[$(date -u +%T)] $*"; }
fail() { t "J16 FAILED: $*"; exit 1; }
[ -e $O/fig2g_retime_all_js.json ] && fail "deposit exists"

# 1. the Fig 2 rebuild must have finished
until grep -qE "J_FIG2 (DONE|FAILED)" $L/run_J_fig2.out 2>/dev/null; do sleep 60; done
grep -q "J_FIG2 DONE" $L/run_J_fig2.out || fail "run_J_fig2 did not finish cleanly"
[ -s $O/fig4_input_js.json ] || fail "no fig4_input_js.json"

# 2. quiet machine, measured
for i in $(seq 1 30); do
  python3 figs/timing_watch.py precheck --root $$ --seconds 30 > $L/J16_precheck.json && break
  t "precheck $i not quiet: $(head -c 300 $L/J16_precheck.json)"; sleep 30
done
python3 figs/timing_watch.py precheck --root $$ --seconds 30 > $L/J16_precheck.json || fail "machine never quiet"
t "precheck passed: $(cat $L/J16_precheck.json | head -c 200)"

# 3. the sitting, watched
( $AP figs/fig2g_retime_all.py --rounds 5 > $L/J16_retime.log 2>&1 || exit 11
  $AP figs/fig2g_retime_extra.py > $L/J16_extra.log 2>&1 || exit 12 ) &
SP=$!
python3 figs/timing_watch.py watch --root $SP --out $L/J16_watch.jsonl --interval 5 &
WP=$!
wait $SP; rc=$?; wait $WP
[ $rc -eq 0 ] || fail "sitting exit $rc (11 = retime, 12 = extra)"

# 4. verdict into the deposit
$AP - <<PY || fail "deposit update"
import json
L, O = "$L", "$O"
rows = [json.loads(l) for l in open(f"{L}/J16_watch.jsonl")]
summ = rows[-1]; loads = sorted(r["load1"] for r in rows[:-1])
p = f"{O}/fig2g_retime_all_js.json"; d = json.load(open(p))
assert d.get("lucid_commit") == "2dc001fe527c4dbaad17fa171da132cfdfe010e6", d.get("lucid_commit")
d["sitting"] = {"id": "J16 (LUC3D-JS-CONSISTENCY-PLAN-2026-10-20)", "cores": 64,
  "valid": summ["valid"], "max_outside_cores": summ["max_outside_cores"], "bench_seen": summ["bench_seen"],
  "median_1min_load": loads[len(loads) // 2], "max_1min_load": loads[-1],
  "precheck": json.load(open(f"{L}/J16_precheck.json")),
  "watch_log": "md56_logs/J16_watch.jsonl",
  "idle_rule": "nothing of ours running, measured from /proc by figs/timing_watch.py; host load from other containers logged, not removable",
  "thread_env": "NUMBA/OMP/OPENBLAS/MKL/VECLIB/NUMEXPR_NUM_THREADS=1, UV_THREADPOOL_SIZE=1",
  "input": "fig4_input_js (export geometry by LUC3D v1.0.5)",
  "keypoints": "every solver on the same first 13,334 frames x 15 joints = 200,010"}
json.dump(d, open(p, "w"), indent=1)
print("valid", summ["valid"], "max_outside_cores", summ["max_outside_cores"], "load median/max", loads[len(loads)//2], loads[-1])
PY
t "J16 DONE $(tail -1 $L/J16_watch.jsonl)"
