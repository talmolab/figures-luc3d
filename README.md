# figures-luc3d

The figures of the LUC3D manuscript, with every generation script, plot-ready
table, and measurement deposit needed to rebuild them. **This repository is the
canonical source of the manuscript's figures.** It began as a mirror of
`talmolab/luc3d` branch `eric/figs` (plus the `figs/out/` deposit subset the panel
scripts read); that branch is superseded by this repo.

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
bash figs/build/run_FINALBUILD.sh            # rebuild the seven placed figures in place
bash figs/build/run_FINALBUILD.sh --verify   # rebuild a clean copy and pixel-compare it to this checkout
PYTHON=/path/to/python3.12 bash figs/build/run_FINALBUILD.sh   # pick the interpreter
python figs/lint_text.py 7                   # check one figure for clipped or overlapping text
```

**Use the wrapper, not a bare `make_figures.py`.** The published figures are the
**b series** (2026-10-20/21), not plain `x7`. A plain `FIG_VARIANT=x7` build reproduces
Fig 4 and Supplementary Figure 2 but not the 3D renders, Fig 2, or Supplementary Figure 4e.
b is x7's deposit set plus eight overrides, listed in `figs/build/B_links.tsv` (every name
required):

- the LUC3D v1.0.5 app captures in `out/b_app/`;
- the LUC3D-triangulated HardFight and two-anchor scenes;
- the J18 timing sitting for Fig 2g;
- the NumPy-free Fig 4 detection deposit.

It is built with `FIG_TAG=b`, so `src/style.tagged_render` places the `*_b.png` Blender
renders made from LUC3D v1.0.5 output. The panels are then promoted into `figures/figN/`.

Two figures keep the settings they were placed with:

- **Fig 3a** was redrawn at the Fig 2/3 swap with `FIG_VARIANT=x7`, so its reprojection
  overlays come from the August app export `out/fig2-protocol.json`.
- **Supplementary Figure 2** (`fig7`) keeps its x7 build.

`figs/build/run_FINALBUILD.sh` encodes all of this. Its header is the reference. The
original launch scripts, link manifests and handoff notes are in `figs/build/provenance/`.
Of those, `run_FINALBUILD.sh` (the 2026-10-15 c series) and `run_B_numpyfree.sh` (an aborted
first try at b) produced no placed figure.

The 3D renders are inputs, not rebuilt. Re-rendering them needs Blender 5.2 (bpy) and
the recipes in `provenance/run_B_luc3d_renders.sh`.

Each panel script runs in its own subprocess, because they set rcParams globally
through `src/style.py` and a panel that leaves state behind must not be able to
change how the next one renders. One broken panel reports and the rest still build.

Needs Python 3.12 with numpy, pandas, scipy, matplotlib, seaborn and PyMuPDF.

**The q series (this branch): Mouse-Dyad session 20250904_183921 left out.** That
session's reference 3D is corrupted over frames ~42k-115k: camera 21369048's identities
are swapped in the lab's SLEAP predictions and were triangulated into the 3D. The
exclusion list is `figs/src/qc_exclude.py`. `FIG_SERIES=q bash figs/build/run_FINALBUILD.sh`
builds the figures from `out/<stem>_q.*`, which are the b deposits with that session
dropped and every cross-session value recomputed. No LUC3D was re-run:

- Fig 2 tracking (`figs/qc/reagg_fig2.py`), Fig 3 triangulation (`figs/qc/reagg_fig3.py`)
  and Supp Fig 4 (`figs/qc/reagg_supp4.py`) re-aggregate the per-session values the b
  deposits already hold, with the producers' own formulas. Each script's `--check` re-runs
  it with nothing excluded and requires it to reproduce the b deposit exactly. In
  `fig4_q.json` the blocks pooled over individual LUC3D keypoints can't be recomputed
  (the keypoints aren't stored) and are `null`; no placed panel reads them.
  `figs/qc/anipose_linear_rerun.py` recovers the pooled Anipose samples (aniposelib only).
- Fig 5 re-runs its Python measurement scripts (`fig5_upright.py`, `fig5_rear_coupling.py`,
  `fig5_controls.py`, `fig5_slot_swap_control.py`) with `QC_EXCLUDE=1`. Without it they
  reproduce the b deposits exactly.

Figs 1 and 4 and Supp Fig 2 come out pixel-identical to b. Figs 2, 3 and 5 and Supp Fig 4
change. On this branch `figs/figures/fig1`-`fig7` hold the q series; the published b
composites are kept in `figs/figures/published-b/`, and the b deposits are unchanged, so
the default (b) build still reproduces them exactly. Check this branch with
`FIG_SERIES=q bash figs/build/run_FINALBUILD.sh --verify`.

Unit tests (seconds, no rebuild): `python -m pytest figs/tests` (needs pytest). They cover
`build/compare.py` (what `--verify` fails on and what it only reports), `build/promote.py`,
and the checkout itself: every input and `B_links.tsv`/`X7_links.tsv` deposit the wrapper
needs is present, the wrapper parses, and the stale `fig6a_within_vs_cross.csv` stays gone.
They do not rebuild anything; `run_FINALBUILD.sh --verify` is the end-to-end check.

No fonts need installing. The figures are set in Liberation Sans 2.1.5, which is
bundled in `figs/src/fonts/` (SIL Open Font License 1.1) and registered by
`src/style.py`. It has the same advance widths as Arial, which the panels were
designed in, so layout is identical. It is loaded explicitly, not left to the
system, because a fallback to DejaVu Sans is wider and fails the label-fit checks,
and a machine-local Arial would produce slightly different glyphs.

This has been verified (2026-11-12): `run_FINALBUILD.sh --verify` copies this checkout,
empties every `figs/figures/fig[1-7]/` so no committed panel can stand in, rebuilds, and
compares the result with the checkout. All seven composites come out pixel-identical, and
every `figs/data/` CSV the panel build writes is byte-identical. The CSVs it does not
write (`fig5s_*`, `fig7s*`, `fig4e_*`) come from the measurement scripts. The composites
also match the figures placed in the manuscript. The datasheets (`python dataset_sheets.py`)
reproduce pixel-identically too, but they read the Mouse-Dyad-10M corpus described in
`DATA-LOCATIONS.md` to measure their values.

As of the 2026-10-29 sync, the repo follows the manuscript's new order: `fig2` is
cross-view tracking and `fig3` is labeling and triangulation. Fig 1 is A, cross-view
association and triangulated 3D; B, dataset arenas and camera rigs; C, the pipeline.
Panel files carry the letters printed in the figures, and Figs 1, 2, 3 and 6 are
placed by `composites.py`. The composites, panels, `figs/data/` CSVs, panel scripts,
`assemble.py` and Supplementary Videos 2 and 4 come from `eric/figs` and match the
project page. The `figs/out/` deposits, the `_b` renders and the v1.0.5 s-DANNCE aggregate
(`fig10-bench/results_v105/agg`) were completed on 2026-11-12, so `run_FINALBUILD.sh`
reproduces all seven figures.

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
| `figs/fig3-bench/`, `figs/fig6-bench/` | the headless Node benchmark drivers for cross-view tracking and exhaustive enumeration (Figs 2, 4, Supp Fig 4) |
| `figs/blender-images/` | the Blender (`bpy`) scene scripts behind every 3D render |
| `figs/legacy/` | the earlier figure scripts that a few current scripts still import helpers from |
| `figs/DATA-LOCATIONS.md` | where the raw corpora live |
| `figs/PANEL-SOURCES.md` | which script produced which panel |

`figs/out/` holds only the deposits the panels actually read. The measurement
scripts at the top level are kept for provenance: they document how those deposits
were produced, but they cannot be re-run from this repo because they need the raw
video corpora described in `DATA-LOCATIONS.md`.

The benchmark drivers (`fig3-bench/`, `fig6-bench/`, and anything that loads
`lucid_v105.mjs`) run the LUC3D application modules themselves. Point them at a clean
checkout of [talmolab/luc3d](https://github.com/talmolab/luc3d) at tag `v1.0.5`
(commit `2dc001fe527c4dbaad17fa171da132cfdfe010e6`) with `LUCID_ROOT=<checkout>` (or
`POSE_DIR=<checkout>/pose`). `lucid_v105.mjs` refuses any other commit, and every
deposit records the commit it was measured on. `fig6-bench/hooks6.mjs` also loads that
checkout's `scripts/bench/hooks.mjs`, and `fig6-bench/xv_experimental.js` is meant to
sit beside the app's `pose/triangulation.js` and `pose/pose-data.js`.

The bench drivers, the Blender scene scripts, and the other producer scripts these
reference (48 files) were copied on 2026-11-12 from the working tree the figures were
built in. Some were last edited after the 2026-10-20/21 figure build, so they are the
latest versions, not guaranteed to be byte-identical to what produced every deposit.
The per-figure build wrappers (`run_FINALBUILD.sh`, `run_B_numpyfree.sh`) are not yet
included.

## A note on the calibration figure

Supplementary Figure 2 compares calibrat3 with Anipose. Both tools are run on
**OpenCV 4.13.0**, which is forced: calibrat3 vendors OpenCV.js 4.13.0 and cannot
run another version, and OpenCV 5 moved the ChArUco corner origin by half a pixel
in both axes. A calibration scored across the two conventions pays a penalty it
does not owe. Anything that re-runs this benchmark must keep both detectors on one
OpenCV or the comparison is invalid.

## License

Code is released under the BSD 3-Clause License (see `LICENSE`). The bundled
Liberation Sans fonts in `figs/src/fonts/` are under the SIL Open Font License 1.1.
