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
| `figs/figures/fig1/fig1.pdf` | Figure 1 | LUC3D overview: cross-view association and triangulated 3D, dataset arenas and camera rigs, pipeline |
| `figs/figures/fig2/fig2.pdf` | Figure 2 | Cross-view tracking: greedy and exhaustive, accuracy, sweep |
| `figs/figures/fig3/fig3.pdf` | Figure 3 | Reprojection-aided labeling protocol and triangulation solvers |
| `figs/figures/fig4/fig4.pdf` | Figure 4 | SLAP-2M datasets: difficulty, keypoints that cannot be triangulated, disagreement across cameras, identity tracking accuracy |
| `figs/figures/fig5/fig5.pdf` | Figure 5 | Social rearing, female-led leader-follower dynamics |
| `figs/figures/fig6/fig6.pdf` | Supplementary Figure 4 | Within-view and cross-view identity against SLEAP and ByteTrack, the staleness (anchor eviction) sweep, and s-DANNCE |
| `figs/figures/fig7/fig7.pdf` | Supplementary Figure 2 | Calibration accuracy of calibrat3 and Anipose on three camera rigs |
| `figs/fig9.pdf` (+ 300 dpi `fig9.png`) | Supplementary Figure 1 | Panopticon: hardware-synchronized nine-camera acquisition, and its calibration solver against calibrat3 |

The tool capability table that used to be Fig 1e is held out of the first
submission (it is meant to become a supplementary table later). It still builds,
as `figs/figures/fig1/fig1s1_tool_table.pdf` with its CSV in `figs/data/fig1/`, but
`assemble.py` does not place it.

Supplementary Figure 3 in the manuscript is a GUI screenshot and is not built by
this pipeline. Neither is Supplementary Figure 1, which is a finished PDF (`figs/fig9.pdf`). The supplementary datasheet table is
`figs/figures/datasheets/datasheet_combined.pdf`.

The four supplementary videos are in `figs/figures/supp_videos/` (see the README
there): 1, the SLAP-2M home-cage flythrough; 2, Mouse-Dyad-10M male and female 3D
poses; 3, SLAP-2M difficulty across animal count and enrichment; 4, the full LUC3D
walkthrough. They are rendered artwork, not rebuilt by `make_figures.py`.

Each `figs/figures/figN/` also holds the per-panel PDFs and PNGs, and a 300 dpi
PNG proof of the composite.

**SLAP-2M input (Figs 2, 4 and 6).** The SLAP-2M arms use SLEAP's pre-proofreading
predictions with none of sleap-nn's instance filters, minus detections with fewer
than five visible body keypoints, and the four tail nodes are excluded from all
scoring. `figs/PANEL-SOURCES.md` lists the commands that produced each deposit.

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

No fonts need installing. The figures are set in Liberation Sans 2.1.5, which is
bundled in `figs/src/fonts/` (SIL Open Font License 1.1) and registered by
`src/style.py`. It has the same advance widths as Arial, which the panels were
designed in, so layout is identical. It is loaded explicitly, not left to the
system, because a fallback to DejaVu Sans is wider and fails the label-fit checks,
and a machine-local Arial would produce slightly different glyphs.

This has been verified: a clean build of all seven figures from the contents of
this repo alone runs 57 panel scripts with 0 failures, and all seven composites are
pixel-identical to the committed PDFs. The datasheets (`python dataset_sheets.py`)
also reproduce pixel-identically, but read the Mouse-Dyad-10M corpus described in
`DATA-LOCATIONS.md` to measure its values.

As of the 2026-10-29 sync, the repo follows the manuscript's new order: `fig2` is
cross-view tracking and `fig3` is labeling and triangulation. Fig 1 is A, cross-view
association and triangulated 3D; B, dataset arenas and camera rigs; C, the pipeline.
Panel files carry the letters printed in the figures, and Figs 1, 2, 3 and 6 are
placed by `composites.py`. The composites, panels, `figs/data/` CSVs, panel scripts,
`assemble.py` and Supplementary Videos 2 and 4 come from `eric/figs` and match the
project page. The `figs/out/` deposits have not been re-synced, so a rebuild from this
repo reproduces Fig 7 exactly but Figs 1 to 6 only approximately until they are.

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

Supplementary Figure 2 compares calibrat3 with Anipose. Both tools are run on
**OpenCV 4.13.0**, which is forced: calibrat3 vendors OpenCV.js 4.13.0 and cannot
run another version, and OpenCV 5 moved the ChArUco corner origin by half a pixel
in both axes. A calibration scored across the two conventions pays a penalty it
does not owe. Anything that re-runs this benchmark must keep both detectors on one
OpenCV or the comparison is invalid.
