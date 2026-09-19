# figures-luc3d

The figures of the LUC3D manuscript, with every generation script, plot-ready
table, and measurement deposit needed to rebuild them. Mirrored from
`talmolab/luc3d` branch `eric/figs`, plus the `figs/out/` deposit subset the panel
scripts read, which the source repo keeps machine-local.

## The seven figures

Five main figures and two supplementary ones. The numbering in this repo is the
**figs tree** numbering, which is not the manuscript's supplementary numbering.

| This repo | Manuscript | Content |
|---|---|---|
| `figs/figures/fig1/fig1.pdf` | Figure 1 | LUC3D overview: rigs, pipeline, re-identification, triangulation, tool table |
| `figs/figures/fig2/fig2.pdf` | Figure 2 | Reprojection-aided labeling protocol and triangulation solvers |
| `figs/figures/fig3/fig3.pdf` | Figure 3 | Cross-view tracking: greedy and exhaustive, accuracy, sweep |
| `figs/figures/fig4/fig4.pdf` | Figure 4 | Social rearing, female-led leader-follower dynamics |
| `figs/figures/fig5/fig5.pdf` | Figure 5 | SLAP-2M datasets: difficulty, IDF1, recovery |
| `figs/figures/fig6/fig6.pdf` | Supplementary Figure 1 | Cross-view identity comparison and the fresh-anchor sweep |
| `figs/figures/fig7/fig7.pdf` | Supplementary Figure 3 | Calibration accuracy of calibrat3 and Anipose on three camera rigs |

Supplementary Figure 2 in the manuscript is a GUI screenshot and is not built by
this pipeline. The supplementary datasheet table is
`figs/figures/datasheets/datasheet_combined.pdf`.

Each `figs/figures/figN/` also holds the per-panel PDFs and PNGs, and a 300 dpi
PNG proof of the composite.

## Rebuilding

```bash
cd figs
python make_figures.py            # every figure
python make_figures.py 3 7        # only these
python make_figures.py --panels   # panels only, no assembly
python lint_text.py 7             # check one figure for clipped or overlapping text
```

Each panel script runs in its own subprocess, because they set rcParams globally
through `src/style.py` and a panel that leaves state behind must not be able to
change how the next one renders. One broken panel reports and the rest still build.

Needs Python 3.12 with numpy, pandas, scipy, matplotlib, seaborn and PyMuPDF.

This has been verified: a clean build of all seven figures from the contents of
this repo alone produces 54 panels with 0 failures.

## What is here

| Path | What it is |
|---|---|
| `figs/figures/figN/` | the composites and every panel, PDF and PNG |
| `figs/panels/` | one script per panel, the only things that draw |
| `figs/src/` | shared style, palette, layout grid and the deposit loader |
| `figs/out/` | the measurement deposits the panels read, plus the app screenshots and exported frames they place |
| `figs/data/figN/` | the plot-ready numbers each panel deposits as it draws, one CSV per panel |
| `figs/assemble.py` | lays panels out into composites; holds the layout grid and the panel titles |
| `figs/*.py`, `*.mjs`, `*.sh` | the measurement pipeline that produced `figs/out/` |
| `figs/DATA-LOCATIONS.md` | where the raw corpora live |
| `figs/PANEL-SOURCES.md` | which script produced which panel |

`figs/out/` holds only the deposits the panels actually read. The measurement
scripts at the top level are kept for provenance: they document how those deposits
were produced, but they cannot be re-run from this repo because they need the raw
video corpora described in `DATA-LOCATIONS.md`.

## A note on the calibration figure

Supplementary Figure 3 compares calibrat3 with Anipose. Both tools are run on
**OpenCV 4.13.0**, which is forced: calibrat3 vendors OpenCV.js 4.13.0 and cannot
run another version, and OpenCV 5 moved the ChArUco corner origin by half a pixel
in both axes. A calibration scored across the two conventions pays a penalty it
does not owe. Anything that re-runs this benchmark must keep both detectors on one
OpenCV or the comparison is invalid.
