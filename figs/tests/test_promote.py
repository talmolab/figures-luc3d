"""build/promote.py: figures/fig{N}<tag>/fig{N}<tag><rest> -> figures/fig{N}/fig{N}<rest>,
and data/fig{N}<tag>/* -> data/fig{N}/*. The tagged composite is never promoted, since the
main composite is re-assembled afterwards.
"""
import subprocess
import sys
from pathlib import Path

import pytest

PROMOTE = Path(__file__).resolve().parents[1] / "build" / "promote.py"


def promote(figs, tag, nums):
    return subprocess.run([sys.executable, str(PROMOTE), str(figs), tag, nums],
                          capture_output=True, text=True, check=True).stdout


def touch(path, text=""):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


@pytest.fixture
def figs(tmp_path):
    for n in (1, 2):
        (tmp_path / "figures" / f"fig{n}").mkdir(parents=True)
    return tmp_path


def test_panels_are_renamed_into_the_main_folder(figs):
    touch(figs / "figures/fig1b/fig1ba_pipeline.pdf", "new a")
    touch(figs / "figures/fig1b/fig1bd_render.png", "new d")
    out = promote(figs, "b", "1")
    assert (figs / "figures/fig1/fig1a_pipeline.pdf").read_text() == "new a"
    assert (figs / "figures/fig1/fig1d_render.png").read_text() == "new d"
    assert "fig1: 2 figure files, 0 data files promoted" in out


def test_tagged_composite_is_not_promoted(figs):
    touch(figs / "figures/fig1/fig1.pdf", "placed composite")
    touch(figs / "figures/fig1b/fig1b.pdf", "tagged composite")
    touch(figs / "figures/fig1b/fig1b.png", "tagged composite")
    promote(figs, "b", "1")
    assert (figs / "figures/fig1/fig1.pdf").read_text() == "placed composite"
    assert not (figs / "figures/fig1/fig1.png").exists()


def test_existing_panel_is_overwritten(figs):
    touch(figs / "figures/fig2/fig2c_timing.pdf", "old")
    touch(figs / "figures/fig2b/fig2bc_timing.pdf", "new")
    promote(figs, "b", "2")
    assert (figs / "figures/fig2/fig2c_timing.pdf").read_text() == "new"


def test_untagged_name_is_copied_as_is(figs):
    touch(figs / "figures/fig1b/axes.json", "{}")
    promote(figs, "b", "1")
    assert (figs / "figures/fig1/axes.json").read_text() == "{}"


def test_data_is_copied_and_folder_created(figs):
    touch(figs / "figures/fig2b/fig2ba_x.pdf")
    touch(figs / "data/fig2b/fig2a_x.csv", "x\n1\n")
    out = promote(figs, "b", "2")
    assert (figs / "data/fig2/fig2a_x.csv").read_text() == "x\n1\n"
    assert "fig2: 1 figure files, 1 data files promoted" in out


def test_several_figures_in_one_call(figs):
    touch(figs / "figures/fig1b/fig1ba.pdf", "1")
    touch(figs / "figures/fig2b/fig2ba.pdf", "2")
    promote(figs, "b", "1,2")
    assert (figs / "figures/fig1/fig1a.pdf").read_text() == "1"
    assert (figs / "figures/fig2/fig2a.pdf").read_text() == "2"


def test_other_figures_are_left_alone(figs):
    touch(figs / "figures/fig1b/fig1ba.pdf", "1")
    touch(figs / "figures/fig2b/fig2ba.pdf", "2")
    promote(figs, "b", "1")
    assert not (figs / "figures/fig2/fig2a.pdf").exists()
