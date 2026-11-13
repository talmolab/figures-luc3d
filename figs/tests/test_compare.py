"""build/compare.py, the pass/fail rule behind run_FINALBUILD.sh --verify.

It must fail on anything the paper shows (a composite that is not pixel-identical, a
rebuilt CSV that is not byte-identical, a composite that was not built) and only report
on the rest (unplaced panels, CSVs the panel build does not write).
"""
import os
import shutil
import sys
from pathlib import Path

import fitz
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "build"))
import compare  # noqa: E402


def make_pdf(path, pages=1, size=(200, 100), rect=None):
    """A small PDF; `rect` draws one filled black rectangle on every page."""
    path.parent.mkdir(parents=True, exist_ok=True)
    doc = fitz.open()
    for _ in range(pages):
        page = doc.new_page(width=size[0], height=size[1])
        if rect:
            page.draw_rect(fitz.Rect(*rect), color=(0, 0, 0), fill=(0, 0, 0))
    doc.save(path)
    return path


def make_tree(root):
    """A figs/ tree with the seven composites, one placed panel and two data CSVs."""
    for n in range(1, 8):
        make_pdf(root / "figures" / f"fig{n}" / f"fig{n}.pdf")
    make_pdf(root / "figures" / "fig2" / "fig2a_panel.pdf")
    (root / "data" / "fig2").mkdir(parents=True)
    (root / "data" / "fig2" / "fig2a_panel.csv").write_text("x,y\n1,2\n")
    (root / "data" / "fig2" / "fig2s_measured.csv").write_text("x\n3\n")
    return root


@pytest.fixture
def trees(tmp_path):
    ref = make_tree(tmp_path / "ref")
    built = tmp_path / "built"
    shutil.copytree(ref, built)
    return built, ref


def run(built, ref, *extra):
    return compare.main(["compare.py", str(built), str(ref), *extra])


# --- pdf_diff

def test_pdf_diff_same_drawing_is_pixel_identical(tmp_path):
    # Two separately written PDFs differ in bytes but not in pixels.
    a = make_pdf(tmp_path / "a.pdf", rect=(10, 10, 50, 50))
    b = make_pdf(tmp_path / "b.pdf", rect=(10, 10, 50, 50))
    assert compare.pdf_diff(a, b) == "PIXEL-IDENTICAL"


def test_pdf_diff_reports_changed_pixels(tmp_path):
    a = make_pdf(tmp_path / "a.pdf")
    b = make_pdf(tmp_path / "b.pdf", rect=(10, 10, 50, 50))
    r = compare.pdf_diff(a, b)
    assert r.startswith("max 255/255 on ") and r.endswith(" px")
    assert int(r.split(" on ")[1].split()[0]) > 0


def test_pdf_diff_reports_page_count(tmp_path):
    a = make_pdf(tmp_path / "a.pdf", pages=1)
    b = make_pdf(tmp_path / "b.pdf", pages=2)
    assert compare.pdf_diff(a, b) == "pages 1 vs 2"


def test_pdf_diff_reports_page_size(tmp_path):
    a = make_pdf(tmp_path / "a.pdf", size=(200, 100))
    b = make_pdf(tmp_path / "b.pdf", size=(300, 100))
    assert compare.pdf_diff(a, b).startswith("size ")


# --- main: what passes

def test_identical_trees_pass(trees, capsys):
    assert run(*trees) == 0
    assert "VERIFY PASS" in capsys.readouterr().out


def test_redrawn_but_pixel_identical_composite_passes(trees):
    built, ref = trees
    make_pdf(built / "figures" / "fig3" / "fig3.pdf")  # new bytes, same pixels
    assert run(built, ref) == 0


def test_unplaced_panel_difference_is_info_only(trees, capsys):
    built, ref = trees
    make_pdf(built / "figures" / "fig2" / "fig2a_panel.pdf", rect=(0, 0, 20, 20))
    assert run(built, ref) == 0
    assert "panel differs (unplaced): figures/fig2/fig2a_panel.pdf" in capsys.readouterr().out


def test_missing_panel_is_info_only(trees, capsys):
    built, ref = trees
    (built / "figures" / "fig2" / "fig2a_panel.pdf").unlink()
    assert run(built, ref) == 0
    assert "panel not rebuilt (unplaced)" in capsys.readouterr().out


def test_csv_not_written_since_marker_is_info_only(trees, tmp_path, capsys):
    # A measurement-script CSV left over from the copy (older than the build start)
    # is reported, not failed, even if it differs.
    built, ref = trees
    old = built / "data" / "fig2" / "fig2s_measured.csv"
    old.write_text("x\n999\n")
    os.utime(old, (1_000_000, 1_000_000))
    marker = tmp_path / ".build_start"
    marker.touch()
    assert run(built, ref, "--since", str(marker)) == 0
    assert "csv not written by the panel build: data/fig2/fig2s_measured.csv" in capsys.readouterr().out


# --- main: what fails

def test_changed_composite_fails(trees, capsys):
    built, ref = trees
    make_pdf(built / "figures" / "fig5" / "fig5.pdf", rect=(10, 10, 50, 50))
    assert run(built, ref) == 1
    out = capsys.readouterr().out
    assert "FAIL: figures/fig5/fig5.pdf: max 255/255" in out
    assert "VERIFY FAIL: 1 problem(s)" in out


def test_missing_composite_fails(trees, capsys):
    built, ref = trees
    (built / "figures" / "fig7" / "fig7.pdf").unlink()
    assert run(built, ref) == 1
    assert "FAIL: figures/fig7/fig7.pdf: not built" in capsys.readouterr().out


def test_rebuilt_csv_with_different_bytes_fails(trees, tmp_path, capsys):
    built, ref = trees
    marker = tmp_path / ".build_start"
    marker.touch()
    os.utime(marker, (1_000_000, 1_000_000))  # every built file is newer: all rebuilt
    (built / "data" / "fig2" / "fig2a_panel.csv").write_text("x,y\n1,3\n")
    assert run(built, ref, "--since", str(marker)) == 1
    assert "FAIL: data/fig2/fig2a_panel.csv: bytes differ" in capsys.readouterr().out


def test_every_problem_is_counted(trees):
    built, ref = trees
    make_pdf(built / "figures" / "fig1" / "fig1.pdf", rect=(0, 0, 10, 10))
    (built / "figures" / "fig4" / "fig4.pdf").unlink()
    (built / "data" / "fig2" / "fig2a_panel.csv").write_text("changed\n")
    assert run(built, ref) == 1
