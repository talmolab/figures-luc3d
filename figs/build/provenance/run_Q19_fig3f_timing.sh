#!/usr/bin/env bash
# Q19 (2026-11-13, eric-mimicc-0-1): the Fig 2g timing sitting for the q series. Same
# protocol as J18 (run_J18_fig3f_timing.sh): both arms on identical frames, one process
# at a time, interleaved, 3 rounds, per-call gate T_MAX_OTHER_CORES=0.5, watched by
# figs/timing_watch.py -- with QC_EXCLUDE=1, so the sessions on src/qc_exclude.py's list
# are not timed. Frames for every other session are drawn by the same seeded rule as J18.
# Every bar is re-timed in this one sitting (not just the Mouse-Dyad one), so the bars
# stay comparable with each other; J18 ran on a different host.
#
# Run from the figures-luc3d checkout (figs/out/tmp/headtohead_b3, figs/out/tmp/lite_pools
# and figs/fig3-bench/node_modules point at the lucid checkout's caches), launched detached
# (setsid) so the launching session's CPU counts as outside the sitting.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
L=figs/build/provenance/q19_logs; O=figs/out
mkdir -p $L
PY=/root/vast/eric/aniposelib413_env/bin/python
export PATH=/root/vast/eric/node26_env/bin:$PATH LUCID_ROOT=/root/vast/eric/lucid-main-md56 \
       FIG3_VARIANT=b3 SCORE_NOTAIL=1 T_TAG=_q19 T_MAX_OTHER_CORES=0.5 QC_EXCLUDE=1
# The Claude Code daemon's command line carries the sleap-3d-gui path, which the watcher's
# "sleap" bench marker matched (0.001 core); allow it by name. Its CPU still counts.
export TW_ALLOW="claude daemon run --origin transient"
export NUMBA_NUM_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 UV_THREADPOOL_SIZE=1
t() { echo "[$(date -u +%T)] $*"; }
fail() { t "Q19 FAILED: $*"; exit 1; }
for f in fig3_runtime_clean_b3_q19.json fig3_headtohead_clean_b3_q19.json fig3_runtime_clean_b3_q19_a4c6.json; do
  [ -e $O/$f ] && fail "$f exists"
done
[ "$(git -C $LUCID_ROOT rev-parse HEAD)" = 2dc001fe527c4dbaad17fa171da132cfdfe010e6 ] || fail "not 2dc001fe"
[ -z "$(git -C $LUCID_ROOT status --porcelain --untracked-files=no)" ] || fail "checkout dirty"

# 1. the 4x6 eligibility scans are untimed and deterministic: reuse J18's
J18=/root/vast/eric/sleap-3d-gui/scratch/repos/lucid/figs/out/tmp/timing_clean_b3_j18
for e in $J18/A4_C6_slap2m_hard/*/eligible.json; do
  d=$O/tmp/timing_clean_b3_q19/A4_C6_slap2m_hard/$(basename "$(dirname "$e")")
  mkdir -p "$d"; [ -e "$d/eligible.json" ] || cp "$e" "$d/eligible.json"
done

# 2. quiet machine, measured (up to 30 min, then give up rather than time on a busy host)
for i in $(seq 1 30); do
  python3 figs/timing_watch.py precheck --root $$ --seconds 30 > $L/Q19_precheck.json && break
  t "precheck $i not quiet: $(head -c 300 $L/Q19_precheck.json)"; sleep 30
done
python3 figs/timing_watch.py precheck --root $$ --seconds 30 > $L/Q19_precheck.json || fail "machine never quiet"
t "precheck passed: $(head -c 200 $L/Q19_precheck.json)"

# 3. the sitting, watched
( $PY figs/fig3f_timing_clean.py > $L/Q19_timing.log 2>&1 || exit 11
  T_A4C6=1 $PY figs/fig3f_timing_clean.py > $L/Q19_a4c6.log 2>&1 || exit 12 ) &
SP=$!
python3 figs/timing_watch.py watch --root $SP --out $L/Q19_watch.jsonl --interval 5 &
WP=$!
wait $SP; rc=$?; wait $WP
[ $rc -eq 0 ] || fail "sitting exit $rc (11 = T1 cells, 12 = 4x6)"
cp $O/tmp/timing_clean_b3_q19/calls.jsonl $O/tmp/timing_clean_b3_q19/calls_a4c6.jsonl $L/

# 4. verdict into the deposits, frames identical to J18, numbers next to J18
$PY - <<PY || fail "deposit update"
import json, pathlib, socket
L, O = pathlib.Path("$L"), pathlib.Path("$O")
rows = [json.loads(l) for l in open(L / "Q19_watch.jsonl")]
summ = rows[-1]; rows = [r for r in rows if not r.get("summary")]
loads = sorted(r["load1"] for r in rows)
over = [r for r in rows if r["outside_cores"] > 0.5]
sitting = {"id": "Q19 (q series: J18 protocol, sessions in src/qc_exclude.py left out, 2026-11-13)",
  "host": socket.gethostname(), "cores": 64,
  "valid": summ["valid"], "max_outside_cores": summ["max_outside_cores"], "bench_seen": summ["bench_seen"],
  "median_1min_load": loads[len(loads) // 2], "max_1min_load": loads[-1],
  "precheck": json.load(open(L / "Q19_precheck.json")), "watch_log": "build/provenance/q19_logs/Q19_watch.jsonl",
  "samples_over_0p5_core": [{"t": r["t"], "outside_cores": r["outside_cores"], "busy": [b["cmd"][:60] for b in r["busy"]]} for r in over],
  "parts": "one part, no resume; per-call gate T_MAX_OTHER_CORES=0.5",
  "watch_allow_extra": "TW_ALLOW=claude daemon run --origin transient (name match only; its CPU counted)",
  "idle_rule": "nothing of ours running, measured from /proc by figs/timing_watch.py (outside <= 0.5 core) "
               "and per call by fig3f_timing_clean.py other_cores (<= 0.5 core); host load from other "
               "containers logged, not removable",
  "thread_env": "NUMBA/OMP/OPENBLAS/MKL_NUM_THREADS=1, UV_THREADPOOL_SIZE=1"}
for name in ("fig3_runtime_clean_b3_q19.json", "fig3_headtohead_clean_b3_q19.json", "fig3_runtime_clean_b3_q19_a4c6.json"):
    p = O / name; d = json.load(open(p))
    if "lucid_commit" in d:
        assert d["lucid_commit"] == "2dc001fe527c4dbaad17fa171da132cfdfe010e6", d["lucid_commit"]
    assert "qc_excluded" in d, name
    d["sitting"] = sitting
    json.dump(d, open(p, "w"), indent=1)
a = pathlib.Path("/root/vast/eric/sleap-3d-gui/scratch/repos/lucid/figs/out/tmp/timing_clean_b3_j18")
b = O / "tmp/timing_clean_b3_q19"
fl = sorted(b.glob("*/*/frames.json"))
same = sum((a / f.relative_to(b)).exists() and json.load(open(a / f.relative_to(b))) == json.load(open(f)) for f in fl)
print(f"frames identical to J18: {same}/{len(fl)}")
old = {m["key"]: m for m in json.load(open(O / "fig3_runtime_clean_j18_b.json"))["measured"]}
old.update({m["key"]: m for m in json.load(open(O / "fig3_runtime_clean_j18_a4c6_b.json"))["measured"]})
new = json.load(open(O / "fig3_runtime_clean_b3_q19.json"))["measured"] + json.load(open(O / "fig3_runtime_clean_b3_q19_a4c6.json"))["measured"]
for m in new:
    o = old[m["key"]]
    e = m.get("exhaustive_seconds_per_frame"); eo = o.get("exhaustive_seconds_per_frame")
    print(f'{m["key"]:>18} {m["n_sessions"]} sessions  greedy {m["seconds_per_frame"]*1e3:.4f} ms (J18 {o["seconds_per_frame"]*1e3:.4f}, {o["n_sessions"]})'
          + (f'  exhaustive {e*1e3:.3f} ms (J18 {eo*1e3:.3f})' if e else ""))
print("valid", summ["valid"], "max_outside_cores", summ["max_outside_cores"], "load median/max", loads[len(loads)//2], loads[-1])
PY
t "Q19 DONE $(tail -1 $L/Q19_watch.jsonl)"
