#!/usr/bin/env bash
# J18 (2026-10-21, eric-mimic2-0-6): a fresh, single-part rerun of the J17 Fig 3F timing
# sitting, because J17 was valid=false (0.578 core outside in 2 of ~950 samples, VS Code
# server + Claude CLI). Identical protocol and frames to T1/J17, under T_TAG=_j18, no
# resume. The per-call gate is tightened to the watcher's own 0.5-core rule
# (T_MAX_OTHER_CORES=0.5) so a call waits rather than times through a busy moment.
# Launched detached from the Claude session (setsid), so that session's CPU counts as
# outside, not as part of the sitting.
set -uo pipefail
cd /root/vast/eric/sleap-3d-gui/scratch/repos/lucid
L=figs/out/md56_logs; O=figs/out
PY=/root/vast/eric/aniposelib413_env/bin/python
export PATH=/root/vast/eric/node26_env/bin:$PATH LUCID_ROOT=/root/vast/eric/lucid-main-md56 \
       FIG3_VARIANT=b3 SCORE_NOTAIL=1 T_TAG=_j18 T_MAX_OTHER_CORES=0.5
export NUMBA_NUM_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 UV_THREADPOOL_SIZE=1
t() { echo "[$(date -u +%T)] $*"; }
fail() { t "J18 FAILED: $*"; exit 1; }
for f in fig3_runtime_clean_b3_j18.json fig3_headtohead_clean_b3_j18.json fig3_runtime_clean_b3_j18_a4c6.json; do
  [ -e $O/$f ] && fail "$f exists"
done
[ "$(git -C $LUCID_ROOT rev-parse HEAD)" = 2dc001fe527c4dbaad17fa171da132cfdfe010e6 ] || fail "not 2dc001fe"
[ -z "$(git -C $LUCID_ROOT status --porcelain --untracked-files=no)" ] || fail "checkout dirty"


# 2. quiet machine, measured (up to 30 min, then give up rather than time on a busy host)
for i in $(seq 1 30); do
  python3 figs/timing_watch.py precheck --root $$ --seconds 30 > $L/J18_precheck.json && break
  t "precheck $i not quiet: $(head -c 300 $L/J18_precheck.json)"; sleep 30
done
python3 figs/timing_watch.py precheck --root $$ --seconds 30 > $L/J18_precheck.json || fail "machine never quiet"
t "precheck passed: $(head -c 200 $L/J18_precheck.json)"

# 3. the sitting, watched
( $PY figs/fig3f_timing_clean.py > $L/J18_timing.log 2>&1 || exit 11
  T_A4C6=1 $PY figs/fig3f_timing_clean.py > $L/J18_a4c6.log 2>&1 || exit 12 ) &
SP=$!
python3 figs/timing_watch.py watch --root $SP --out $L/J18_watch.jsonl --interval 5 &
WP=$!
wait $SP; rc=$?; wait $WP
[ $rc -eq 0 ] || fail "sitting exit $rc (11 = T1 cells, 12 = 4x6)"

# 4. verdict into the deposits, frames identical to J17, numbers next to J17
$PY - <<PY || fail "deposit update"
import json, pathlib
L, O = pathlib.Path("$L"), pathlib.Path("$O")
parts = [[json.loads(l) for l in open(L / f)] for f in ("J18_watch.jsonl",)]
rows = [r for p in parts for r in p if not r.get("summary")]
sums = [p[-1] for p in parts]
summ = {"valid": all(x["valid"] for x in sums), "max_outside_cores": max(x["max_outside_cores"] for x in sums),
        "bench_seen": [b for x in sums for b in x["bench_seen"]]}
loads = sorted(r["load1"] for r in rows)
over = [r for r in rows if r["outside_cores"] > 0.5]
sitting = {"id": "J18 (rerun of J17, 2026-10-21)", "host": "eric-mimic2-0-6", "cores": 64,
  "valid": summ["valid"], "max_outside_cores": summ["max_outside_cores"], "bench_seen": summ["bench_seen"],
  "median_1min_load": loads[len(loads) // 2], "max_1min_load": loads[-1],
  "precheck": json.load(open(L / "J18_precheck.json")), "watch_log": "md56_logs/J18_watch.jsonl",
  "samples_over_0p5_core": [{"t": r["t"], "outside_cores": r["outside_cores"], "busy": [b["cmd"][:60] for b in r["busy"]]} for r in over],
  "parts": "one part, no resume; per-call gate T_MAX_OTHER_CORES=0.5",
  "idle_rule": "nothing of ours running, measured from /proc by figs/timing_watch.py (outside <= 0.5 core) "
               "and per call by fig3f_timing_clean.py other_cores (<= 1 core); host load from other "
               "containers logged, not removable",
  "thread_env": "NUMBA/OMP/OPENBLAS/MKL_NUM_THREADS=1, UV_THREADPOOL_SIZE=1"}
print("samples over 0.5 core:", [(r["t"], r["outside_cores"]) for r in over])
for name in ("fig3_runtime_clean_b3_j18.json", "fig3_headtohead_clean_b3_j18.json", "fig3_runtime_clean_b3_j18_a4c6.json"):
    p = O / name; d = json.load(open(p))
    if "lucid_commit" in d:
        assert d["lucid_commit"] == "2dc001fe527c4dbaad17fa171da132cfdfe010e6", d["lucid_commit"]
    d["sitting"] = sitting
    json.dump(d, open(p, "w"), indent=1)
a, b = O / "tmp/timing_clean_b3_j17", O / "tmp/timing_clean_b3_j18"
fl = sorted(b.glob("*/*/frames.json"))
same = sum((a / f.relative_to(b)).exists() and json.load(open(a / f.relative_to(b))) == json.load(open(f)) for f in fl)
print(f"frames identical to J17: {same}/{len(fl)}")
old = {m["key"]: m for m in json.load(open(O / "fig3_runtime_clean_b3_j17.json"))["measured"]}
old.update({m["key"]: m for m in json.load(open(O / "fig3_runtime_clean_b3_j17_a4c6.json"))["measured"]})
new = json.load(open(O / "fig3_runtime_clean_b3_j18.json"))["measured"] + json.load(open(O / "fig3_runtime_clean_b3_j18_a4c6.json"))["measured"]
for m in new:
    o = old[m["key"]]
    e = m.get("exhaustive_seconds_per_frame"); eo = o.get("exhaustive_seconds_per_frame")
    print(f'{m["key"]:>18} greedy {m["seconds_per_frame"]*1e3:.4f} ms (J17 {o["seconds_per_frame"]*1e3:.4f})'
          + (f'  exhaustive {e*1e3:.3f} ms (J17 {eo*1e3:.3f})' if e else ""))
print("valid", summ["valid"], "max_outside_cores", summ["max_outside_cores"], "load median/max", loads[len(loads)//2], loads[-1])
PY
t "J18 DONE $(tail -1 $L/J18_watch.jsonl)"
