"""The inputs run_FINALBUILD.sh needs are in the repo, and the build stays as placed.

These check the checkout itself, in seconds, so a missing render or deposit shows up
before a full rebuild. The full proof is `bash figs/build/run_FINALBUILD.sh --verify`.
"""
import csv
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

FIGS = Path(__file__).resolve().parents[1]
BUILD = FIGS / "build"
WRAPPER = BUILD / "run_FINALBUILD.sh"
sys.path.insert(0, str(FIGS))


def links(name):
    with open(BUILD / name, newline="") as f:
        return [row for row in csv.reader(f, delimiter="\t") if row and row[0].strip()]


def linked_file(name, variant):
    stem, ext = name.rsplit(".", 1)
    return FIGS / "out" / f"{stem}_{variant}.{ext}"


def preflight_inputs():
    """The paths in the wrapper's step-0 `for f in ...; do` list."""
    m = re.search(r"^for f in (.*?); do", WRAPPER.read_text(), re.S | re.M)
    assert m, "step-0 input list not found in run_FINALBUILD.sh"
    return m.group(1).replace("\\\n", " ").split()


@pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")
def test_wrapper_parses():
    subprocess.run(["bash", "-n", str(WRAPPER)], check=True)


def test_preflight_list_is_found():
    assert len(preflight_inputs()) >= 10


@pytest.mark.parametrize("path", preflight_inputs())
def test_preflight_input_exists(path):
    assert (FIGS / path).exists(), f"missing build input {path}"


@pytest.mark.parametrize("variant,table", [("b", "B_links.tsv"), ("x7", "X7_links.tsv")])
def test_link_tables_are_well_formed(variant, table):
    rows = links(table)
    assert rows, f"{table} is empty"
    names = [r[0] for r in rows]
    assert len(names) == len(set(names)), f"{table} lists a deposit twice"
    for r in rows:
        assert len(r) == 3, f"{table}: expected 3 tab-separated columns in {r}"
        assert "." in r[0], f"{table}: {r[0]} has no extension"


@pytest.mark.parametrize("name", [r[0] for r in links("B_links.tsv")])
def test_b_link_exists(name):
    # FIG_VARIANT=b + FIG_VARIANT_REQUIRE make the loader refuse a fallback to the base file.
    assert linked_file(name, "b").exists(), f"missing out/{linked_file(name, 'b').name}"


@pytest.mark.parametrize("name", [r[0] for r in links("X7_links.tsv")])
def test_x7_link_exists(name):
    # Fig 3a and Fig 7 are built with FIG_VARIANT=x7.
    assert linked_file(name, "x7").exists(), f"missing out/{linked_file(name, 'x7').name}"


def test_placed_composites_are_committed():
    for n in range(1, 8):
        assert (FIGS / "figures" / f"fig{n}" / f"fig{n}.pdf").exists()


def test_within_vs_cross_default_run_is_skipped():
    # Its default run wrote the stale 50-session data/fig6/fig6a_within_vs_cross.csv.
    import make_figures
    scripts = [p.name for _, p in make_figures.panel_scripts(None)]
    assert "fig6_05_within_vs_cross.py" not in scripts
    assert any(s.startswith("fig6_") for s in scripts)
    assert (make_figures.PANELS / "fig6_05_within_vs_cross.py").exists()


def test_panel_scripts_filter_by_figure():
    import make_figures
    assert {n for n, _ in make_figures.panel_scripts({4})} == {4}


def test_stale_within_vs_cross_table_is_gone():
    assert not (FIGS / "data" / "fig6" / "fig6a_within_vs_cross.csv").exists()
    assert (FIGS / "data" / "fig6" / "fig6a_within_vs_cross_variant.csv").exists()
