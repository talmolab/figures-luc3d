#!/usr/bin/env bash
# J17 (LUC3D-JS plan, 2026-10-20, eric-mimic2-0-6): Fig 3F timing sitting, rerun under the
# CPU watcher. Same protocol as T1 (fig3f_timing_clean.py: identical frames for both arms,
# one process at a time, interleaved, order flipped per round, 3 rounds, solve time only)
# plus the 4x6 greedy-only cell (T_A4C6=1), both under T_TAG=_j17 so nothing earlier is
# overwritten. LUC3D = fig3-bench/fig3_bench.mjs importing v1.0.5 pose/ (the script refuses
# anything but 2dc001fe). NOTHING ELSE OF OURS RUNS: figs/timing_watch.py (the same watcher
# as J16 on eric-mimicc) prechecks, then watches the whole sitting; fig3f_timing_clean.py
# also gates every call on other processes' CPU (<= 1 core). The verdict goes into the
# deposits.
set -uo pipefail
cd /root/vast/eric/sleap-3d-gui/scratch/repos/lucid
L=figs/out/md56_logs; O=figs/out
PY=/root/vast/eric/aniposelib413_env/bin/python
export PATH=/root/vast/eric/node26_env/bin:$PATH LUCID_ROOT=/root/vast/eric/lucid-main-md56 \
       FIG3_VARIANT=b3 SCORE_NOTAIL=1 T_TAG=_j17 T_RESUME=1
export NUMBA_NUM_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 UV_THREADPOOL_SIZE=1
t() { echo "[$(date -u +%T)] $*"; }
fail() { t "J17 FAILED: $*"; exit 1; }
for f in fig3_runtime_clean_b3_j17.json fig3_headtohead_clean_b3_j17.json fig3_runtime_clean_b3_j17_a4c6.json; do
  [ -e $O/$f ] && fail "$f exists"
done
[ "$(git -C $LUCID_ROOT rev-parse HEAD)" = 2dc001fe527c4dbaad17fa171da132cfdfe010e6 ] || fail "not 2dc001fe"
[ -z "$(git -C $LUCID_ROOT status --porcelain --untracked-files=no)" ] || fail "checkout dirty"

# 1. (J12 finished 04:30; part 1 of this sitting ran 14:29-15:34, kept via T_RESUME=1)

# 2. quiet machine, measured (up to 30 min, then give up rather than time on a busy host)
for i in $(seq 1 30); do
  python3 figs/timing_watch.py precheck --root $$ --seconds 30 > $L/J17_precheck.json && break
  t "precheck $i not quiet: $(head -c 300 $L/J17_precheck.json)"; sleep 30
done
python3 figs/timing_watch.py precheck --root $$ --seconds 30 > $L/J17_precheck.json || fail "machine never quiet"
t "precheck passed: $(head -c 200 $L/J17_precheck.json)"

# 3. the sitting, watched
( $PY figs/fig3f_timing_clean.py > $L/J17_timing.log 2>&1 || exit 11
  T_A4C6=1 $PY figs/fig3f_timing_clean.py > $L/J17_a4c6.log 2>&1 || exit 12 ) &
SP=$!
python3 figs/timing_watch.py watch --root $SP --out $L/J17_watch.jsonl --interval 5 &
WP=$!
wait $SP; rc=$?; wait $WP
[ $rc -eq 0 ] || fail "sitting exit $rc (11 = T1 cells, 12 = 4x6)"

# 4. verdict into the deposits, frames identical to T1, numbers next to T1
$PY - <<PY || fail "deposit update"
import json, pathlib
L, O = pathlib.Path("$L"), pathlib.Path("$O")
parts = [[json.loads(l) for l in open(L / f)] for f in ("J17_watch_part1.jsonl", "J17_watch.jsonl")]
rows = [r for p in parts for r in p if not r.get("summary")]
sums = [p[-1] for p in parts]
summ = {"valid": all(x["valid"] for x in sums), "max_outside_cores": max(x["max_outside_cores"] for x in sums),
        "bench_seen": [b for x in sums for b in x["bench_seen"]]}
loads = sorted(r["load1"] for r in rows)
over = [r for r in rows if r["outside_cores"] > 0.5]
sitting = {"id": "J17 (LUC3D-JS-CONSISTENCY-PLAN-2026-10-20)", "host": "eric-mimic2-0-6", "cores": 64,
  "valid": summ["valid"], "max_outside_cores": summ["max_outside_cores"], "bench_seen": summ["bench_seen"],
  "median_1min_load": loads[len(loads) // 2], "max_1min_load": loads[-1],
  "precheck": json.load(open(L / "J17_precheck.json")), "watch_log": "md56_logs/J17_watch_part1.jsonl + J17_watch.jsonl",
  "samples_over_0p5_core": [{"t": r["t"], "outside_cores": r["outside_cores"], "busy": [b["cmd"][:60] for b in r["busy"]]} for r in over],
  "parts": "part 1 14:29-15:34 UTC (stopped by the per-call gate at 1.2 cores, VS Code extension host); part 2 resumed with T_RESUME=1, gate now waits for quiet",
  "idle_rule": "nothing of ours running, measured from /proc by figs/timing_watch.py (outside <= 0.5 core) "
               "and per call by fig3f_timing_clean.py other_cores (<= 1 core); host load from other "
               "containers logged, not removable",
  "thread_env": "NUMBA/OMP/OPENBLAS/MKL_NUM_THREADS=1, UV_THREADPOOL_SIZE=1"}
print("samples over 0.5 core:", [(r["t"], r["outside_cores"]) for r in over])
for name in ("fig3_runtime_clean_b3_j17.json", "fig3_headtohead_clean_b3_j17.json", "fig3_runtime_clean_b3_j17_a4c6.json"):
    p = O / name; d = json.load(open(p))
    if "lucid_commit" in d:
        assert d["lucid_commit"] == "2dc001fe527c4dbaad17fa171da132cfdfe010e6", d["lucid_commit"]
    d["sitting"] = sitting
    json.dump(d, open(p, "w"), indent=1)
a, b = O / "tmp/timing_clean_b3", O / "tmp/timing_clean_b3_j17"
fl = sorted(b.glob("*/*/frames.json"))
same = sum((a / f.relative_to(b)).exists() and json.load(open(a / f.relative_to(b))) == json.load(open(f)) for f in fl)
print(f"frames identical to T1: {same}/{len(fl)}")
old = {m["key"]: m for m in json.load(open(O / "fig3_runtime_clean_b3.json"))["measured"]}
old.update({m["key"]: m for m in json.load(open(O / "fig3_runtime_clean_b3_a4c6.json"))["measured"]})
new = json.load(open(O / "fig3_runtime_clean_b3_j17.json"))["measured"] + json.load(open(O / "fig3_runtime_clean_b3_j17_a4c6.json"))["measured"]
for m in new:
    o = old[m["key"]]
    e = m.get("exhaustive_seconds_per_frame"); eo = o.get("exhaustive_seconds_per_frame")
    print(f'{m["key"]:>18} greedy {m["seconds_per_frame"]*1e3:.4f} ms (T1 {o["seconds_per_frame"]*1e3:.4f})'
          + (f'  exhaustive {e*1e3:.3f} ms (T1 {eo*1e3:.3f})' if e else ""))
print("valid", summ["valid"], "max_outside_cores", summ["max_outside_cores"], "load median/max", loads[len(loads)//2], loads[-1])
PY
t "J17 DONE $(tail -1 $L/J17_watch.jsonl)"
