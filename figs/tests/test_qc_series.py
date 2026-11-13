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
