# figures-luc3d

The figures of the LUC3D manuscript ("LUC3D: Browser-based 3D annotation and
proofreading GUI for pose estimation and multi-animal tracking"), with every
generation script, plot-ready table, and measurement deposit needed to rebuild
them identically. Mirrored from `talmolab/luc3d` branch `eric/figs` (commit
`b7aece1`, 2026-08-26), plus the `figs/out/` deposit subset the panel scripts
read, which the source repo keeps machine-local.

## The six figures

Figure numbering here matches the manuscript exactly. The final composites are:

| Figure | File | Content |
|---|---|---|
| 1 | `figs/figures/fig1/fig1.pdf` | LUC3D overview — rigs, pipeline, re-identification, triangulation, tool table |
| 2 | `figs/figures/fig2/fig2.pdf` | Reprojection-aided labeling protocol + triangulation solvers (e–g) |
| 3 | `figs/figures/fig3/fig3.pdf` | Cross-view tracking — greedy vs exhaustive, accuracy, sweep |
| 4 | `figs/figures/fig4/fig4.pdf` | Social rearing — female-led leader–follower dynamics |
| 5 | `figs/figures/fig5/fig5.pdf` | SLAP-2M datasets — difficulty, IDF1, recovery |
| 6 | `figs/figures/fig6/fig6.pdf` | Supplementary — cross-view identity comparison + fresh-anchor sweep |

Each `figs/figures/figN/` also holds the per-panel PDFs/PNGs and a 300 dpi PNG
proof of the composite. The supplementary datasheet table is
`figs/figures/datasheets/datasheet_combined.pdf` (built by
`figs/dataset_sheets.py`).

Before 2026-08-26 the source repo numbered these directories 1, 2, 13, 5, 6,
11; everything (panel scripts, `data/` CSVs, syncs, assembler config) was
renamed to the manuscript's 1–6. **One deliberate exception:** the
`figs/out/*.json` measurement deposits keep their historical names
(`fig5_upright.json` backs Figure 4, `fig7_variant_best.json` and
`fig8_methods_50.json` back Figure 6, `fig9_slap2m.json` backs Figure 5b, the
`fig4_*.json` solver deposits back Figure 2e–g) — they key ~100 GB of raw
measurement records on the lab machine and are named by the harnesses that
wrote them. `figs/PANEL-SOURCES.md` maps every panel to its deposit.

## Rebuilding the figures

The plotting pass is self-contained in this repo:

```bash
cd figs
python3 -m venv .venv
.venv/bin/pip install numpy pandas scipy matplotlib pymupdf pillow
.venv/bin/python make_figures.py          # every panel, both syncs, assemble 1-6
.venv/bin/python make_figures.py 4        # just Figure 4
```

`make_figures.py` runs each panel script in its own subprocess, then the two
pre-merge syncs (`fig3_sync.py` builds Figure 3's a/b stack and column
composites; `fig6_sync.py` builds Figure 6's 2×2 tracker block, including the
`--variant` arm of `panels/fig6_05_within_vs_cross.py`), then
`assemble.py 1 2 3 4 5 6`. Rebuilt composites are pixel-identical to the
committed ones (verified at 100/300 dpi with PyMuPDF).

Per-panel provenance — which script draws each panel, which deposit it reads,
which measurement script wrote that deposit, and which `data/figN/*.csv`
plot-ready table it re-deposits — is generated into `figs/PANEL-SOURCES.md`
(`.venv/bin/python figs/make_docs.py`). Do not edit that file by hand.

## What is here and what is not

- `figs/panels/` — one script per panel (plus `_src`-suffixed code donors the
  merged panels import).
- `figs/data/` — the small plot-ready CSVs each panel deposits, committed so
  every number on the artwork is auditable against a table.
- `figs/out/` — the measurement deposits (JSON) and app-export frames (PNG)
  the panel scripts read. This is the replication boundary: with these files,
  every panel rebuilds bit-identically without the corpora.
- `figs/fig*_*.py`, `figs/*.mjs` — the measurement pass that produced
  `figs/out/`. These need the raw corpora (Mouse-Dyad-10M, SLAP-2M,
  HardFight), the LUC3D app checkout, and in places a Playwright/bench
  environment; they are included for provenance, not as part of the rebuild.
- `figs/blender-images/` — Blender scene scripts and their rendered PNGs
  (fig1a rigs, fig1d/fig2a pose renders, fig3c/fig3d illustrations, fig4a
  display render, fig5a difficulty grid). Re-rendering needs the `bpy`
  environment described in each script; the committed renders are what the
  panels consume.
- `figs/figures/drafts/` — the manuscript tex (`new_luc3d.tex` is the current
  six-figure draft) and the diagram assets (`figs/` PNGs) some panels embed.
- The raw video corpora and the full `out/` measurement records are **not**
  here — see the Data Availability statement in the manuscript
  (huggingface.co/datasets/talmolab/slap-2m and .../mouse-dyad-10M).

Everything was produced by driving the real LUC3D app over real data — no
mock-ups; see `figs/README.md` for the house style and per-figure history.
