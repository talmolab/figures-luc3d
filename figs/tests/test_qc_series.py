"""The q series: the b deposits with the src/qc_exclude.py sessions left out.

Each q deposit must be complete (one per b deposit), and wherever it differs from b it
must record the exclusion and no longer contain the excluded sessions. The placed data
tables on this branch must not contain them either.
"""
import json
import re
import sys
from pathlib import Path

import pytest

FIGS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(FIGS))
from src.qc_exclude import EXCLUDE, excluded, provenance  # noqa: E402

NAMES = [s for s in EXCLUDE] + [ts for ts, _ in EXCLUDE.values()]
PATTERN = re.compile("|".join(re.escape(n) for n in NAMES))


def b_names():
    with open(FIGS / "build" / "B_links.tsv") as f:
        return [line.split("\t")[0] for line in f if line.strip()]


def series_file(name, series):
    stem, ext = name.rsplit(".", 1)
    return FIGS / "out" / f"{stem}_{series}.{ext}"


def test_excluded_matches_folder_name_and_video_timestamp():
    assert excluded("20250904_183921")
    assert excluded("20250904183930")
    assert excluded("/root/vast/eric/BMimica/20250904_183921/x.h5")
    assert not excluded("20250904_140306")
    assert not excluded("20250905_203607")
    assert set(provenance()) == set(EXCLUDE)


@pytest.mark.parametrize("name", b_names())
def test_q_deposit_exists(name):
    assert series_file(name, "q").exists(), f"missing out/{series_file(name, 'q').name}"


@pytest.mark.parametrize("name", [n for n in b_names() if n.endswith(".json")])
def test_changed_q_deposit_records_and_applies_the_exclusion(name):
    b, q = series_file(name, "b"), series_file(name, "q")
    if b.read_bytes() == q.read_bytes():
        # Unchanged from b: only allowed if b never mentions an excluded session, or no
        # placed panel reads it (fig3_trackers: its Mouse-Dyad block is unplaced).
        if name != "fig3_trackers.json":
            assert not PATTERN.search(b.read_text()), f"{q.name} is a b copy but b names an excluded session"
        return
    d = json.loads(q.read_text())
    assert d.get("qc_excluded") == provenance(), f"{q.name} does not record qc_excluded"
    rest = {k: v for k, v in d.items() if k != "qc_excluded"}
    assert not PATTERN.search(json.dumps(rest)), f"{q.name} still names an excluded session"


def test_placed_data_tables_exclude_the_sessions():
    hits = [p.relative_to(FIGS) for p in sorted(FIGS.glob("data/fig[1-7]/*.csv"))
            if PATTERN.search(p.read_text())]
    assert not hits, f"data tables still list an excluded session: {hits}"


# --- Fig 2g timing (build/provenance/run_QJ18_fig3f_from_calls.sh) ---------------------
# A deposit that names no session (fig3_runtime_clean_j18) passes the check above as a b
# copy, so the timing gets its own tests: the q deposits must come from J18's calls with
# the excluded sessions' calls left out, through fig3f_timing_clean.py's own aggregation.

def _from_calls(tmp_path, qc):
    """fig3f_timing_clean.py write_deposits() on build/provenance/j18_calls into tmp_path."""
    import subprocess
    # the producer's input name -> its series link in out/
    hh_in, src = (("fig3_headtohead_q.json", "fig3_headtohead_q.json") if qc else
                  ("fig3_headtohead_b3.json", "fig3_headtohead_b.json"))
    (tmp_path / hh_in).symlink_to(FIGS / "out" / src)
    code = ("import sys, pathlib; sys.path.insert(0, '.'); import fig3f_timing_clean as m; "
            f"m.OUT = pathlib.Path({str(tmp_path)!r}); "
            "m.write_deposits(m.calls_from('build/provenance/j18_calls/calls.jsonl'), 3, "
            "'2dc001fe527c4dbaad17fa171da132cfdfe010e6')")
    env = {"FIG3_VARIANT": "b3", "SCORE_NOTAIL": "1", "T_TAG": "_x",
           **({"QC_EXCLUDE": "1"} if qc else {}), "PATH": "/usr/bin:/bin"}
    subprocess.run([sys.executable, "-c", code], cwd=FIGS, env=env, check=True, capture_output=True)
    return {n: json.loads((tmp_path / f"fig3_{n}_clean_b3_x.json").read_text())
            for n in ("runtime", "headtohead")}


def _no_sitting(d):
    return {k: v for k, v in d.items() if k != "sitting"}


def test_j18_calls_reproduce_the_b_timing(tmp_path):
    out = _from_calls(tmp_path, qc=False)
    for n in ("runtime", "headtohead"):
        b = json.loads(series_file(f"fig3_{n}_clean_j18.json", "b").read_text())
        assert out[n] == _no_sitting(b), f"J18 calls do not reproduce fig3_{n}_clean_j18_b.json"


def test_q_timing_is_j18_minus_the_excluded_sessions(tmp_path):
    out = _from_calls(tmp_path, qc=True)
    for n in ("runtime", "headtohead"):
        q = json.loads(series_file(f"fig3_{n}_clean_j18.json", "q").read_text())
        assert out[n] == _no_sitting(q), f"fig3_{n}_clean_j18_q.json is not J18 minus the exclusion"
        assert q["qc_excluded"] == provenance()
    md = next(m for m in out["runtime"]["measured"] if m["key"] == "A2_C5_bmimica")
    assert md["n_sessions"] == 55 and md["frames_per_round"] == 55_000
    a4 = json.loads(series_file("fig3_runtime_clean_j18_a4c6.json", "q").read_text())
    assert a4["qc_excluded"] == provenance()


def test_benchmark_subset_table_is_the_q_corpus():
    import csv
    rows = {r["subset"]: r for r in csv.DictReader(open(FIGS / "data/fig5/fig5s_benchmark50_subset.csv"))}
    assert set(rows) == {"all_55", "benchmark_49", "outside_49_only"}
    ev = json.loads((FIGS / "out" / "fig5_upright_q.json").read_text())["events"]
    assert int(rows["all_55"]["displays"]) == len(ev)
    assert int(rows["all_55"]["sessions"]) == len({e["session"] for e in ev})
    assert int(rows["benchmark_49"]["displays"]) + int(rows["outside_49_only"]["displays"]) == len(ev)
