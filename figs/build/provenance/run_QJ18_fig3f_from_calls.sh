#!/usr/bin/env bash
# QJ18 (2026-11-14): the q series' Fig 2g timing = the J18 sitting (run_J18_fig3f_timing.sh,
# the placed b timing) with the calls of the sessions in src/qc_exclude.py left out. Nothing
# is re-timed: fig3f_timing_clean.py T_FROM_CALLS re-aggregates J18's calls.jsonl through the
# code the sitting used (with no exclusion it reproduces J18's deposits exactly). Both arms
# of a call were timed on the same frames, so dropping a call drops it from both arms.
#
# Why not the Q19 re-time (run_Q19_fig3f_timing.sh, kept with its logs): Q19 was valid=false
# under timing_watch.py (7 of 1,760 samples at 0.50-0.66 core, mostly the VS Code server) and
# ran on a different, busier host (load ~24 vs ~6), which moved every bar by itself
# (exhaustive ~5% slower in every cell). J18 minus 183921 moves Mouse-Dyad greedy 0.2802 ->
# 0.2805 ms and exhaustive 3.948 -> 3.952 ms; the SLAP-2M cells do not change.
#
# Writes out/fig3_{runtime,headtohead}_clean_b3_j18q.json and _b3_j18q_a4c6.json (new names,
# refuses to overwrite), then the q links out/fig3_{runtime,headtohead}_clean_j18{,_a4c6}_q.json.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
P=build/provenance; C=$P/j18_calls; O=out
PY=/root/vast/eric/aniposelib413_env/bin/python
J18=/root/vast/eric/sleap-3d-gui/scratch/repos/lucid/figs/out/tmp/timing_clean_b3_j18

# 1. J18's per-call logs, kept with the branch so the q timing can be re-derived from it
mkdir -p $C
for f in calls.jsonl calls_a4c6.jsonl; do
  [ -e $C/$f ] || cp $J18/$f $C/$f
  cmp -s $J18/$f $C/$f || { echo "QJ18: $C/$f differs from $J18/$f" >&2; exit 1; }
done

# 2. re-aggregate without the qc_exclude sessions
LUCID_ROOT=/root/vast/eric/lucid-main-md56 FIG3_VARIANT=b3 SCORE_NOTAIL=1 QC_EXCLUDE=1 \
  T_TAG=_j18q T_FROM_CALLS=$C $PY fig3f_timing_clean.py

# 3. J18's sitting record, the 4x6 deposit, and the q links
$PY - <<'PY'
import json, pathlib, sys
sys.path.insert(0, ".")
from src.qc_exclude import excluded, provenance
O, C = pathlib.Path("out"), pathlib.Path("build/provenance/j18_calls")
derived = ("q series: J18's calls re-aggregated by fig3f_timing_clean.py T_FROM_CALLS with "
           "the sessions in src/qc_exclude.py left out (both arms of each dropped call); "
           "nothing re-timed. build/provenance/run_QJ18_fig3f_from_calls.sh")
j18 = json.load(open(O / "fig3_runtime_clean_j18_b.json"))
for name in ("fig3_runtime_clean_b3_j18q.json", "fig3_headtohead_clean_b3_j18q.json"):
    p = O / name; d = json.load(open(p))
    assert d.get("qc_excluded"), name
    d["sitting"] = {**j18["sitting"], "derived": derived}
    json.dump(d, open(p, "w"), indent=1)
# 4x6: J18's greedy-only sitting. None of its sessions is excluded, so its numbers stand;
# the q copy records that it was checked.
a4 = json.load(open(O / "fig3_runtime_clean_j18_a4c6_b.json"))
calls = [json.loads(l) for l in open(C / "calls_a4c6.jsonl")]
bad = sorted({r["session"] for r in calls if excluded(r["session"])}
             | {s for m in a4["measured"] for s in m.get("eligible_frames", {}) if excluded(s)})
assert not bad, f"4x6 sitting holds excluded sessions {bad}: re-aggregate it too"
a4["qc_excluded"] = provenance()
a4["sitting"] = {**a4["sitting"], "derived": "q series: no session of this sitting is on "
                 "src/qc_exclude.py's list, numbers unchanged. "
                 "build/provenance/run_QJ18_fig3f_from_calls.sh"}
p = O / "fig3_runtime_clean_b3_j18q_a4c6.json"
assert not p.exists(), f"{p} exists"
json.dump(a4, open(p, "w"), indent=1)
for src, link in (("fig3_runtime_clean_b3_j18q.json", "fig3_runtime_clean_j18_q.json"),
                  ("fig3_headtohead_clean_b3_j18q.json", "fig3_headtohead_clean_j18_q.json"),
                  ("fig3_runtime_clean_b3_j18q_a4c6.json", "fig3_runtime_clean_j18_a4c6_q.json")):
    (O / link).write_text((O / src).read_text())
    print(f"[QJ18] {link} <- {src}")
new = {m["key"]: m for m in json.load(open(O / "fig3_runtime_clean_j18_q.json"))["measured"]}
for m in j18["measured"]:
    n = new[m["key"]]
    print(f'{m["key"]:>16}: {m["n_sessions"]} -> {n["n_sessions"]} sessions  greedy '
          f'{m["seconds_per_frame"]*1e3:.4f} -> {n["seconds_per_frame"]*1e3:.4f} ms  exhaustive '
          f'{m["exhaustive_seconds_per_frame"]*1e3:.3f} -> {n["exhaustive_seconds_per_frame"]*1e3:.3f} ms')
PY
