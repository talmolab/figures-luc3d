# luc3d-bench scripts

A copy of the `scripts/` folder of `luc3d-bench`, the benchmark workspace that built
the Mouse-Dyad-10M and SLAP-2M benchmark inputs and the other-tool baselines. The
deposits in `figs/out/` name these scripts by their original path,
`/root/vast/eric/luc3d-bench/scripts/...`. That folder maps to `figs/luc3d-bench/scripts/...` here.

Copied on 2026-11-13 from the working tree. `luc3d-bench` is a local git repository
with no remote, at commit `9804406` (2026-05-21) plus uncommitted edits, so these are
the latest working-tree versions. They are not guaranteed to be byte-identical to the
version that produced every deposit. Only the scripts were copied (plus
`pyproject.toml`, `package.json` and `main.py`). The Python environments, models,
`outputs/` and data stay on the lab server (see `figs/DATA-LOCATIONS.md`).

## The scripts the paper's deposits depend on

| Script | What it produced |
|---|---|
| `scripts/bartul/bmimica_retrack_all.py` | sleap-nn filter+track ("keeptrack") on every Mouse-Dyad-10M session and camera |
| `scripts/bartul/bmimica_build_master.py`, `bmimica_build_inputs.py`, `build_inputs*.py` | the Mouse-Dyad-10M session table (`outputs/bmimica/bmimica_master.tsv`) and the per-session canonical detection files |
| `scripts/bartul/build_gt_reproj.py`, `bmimica_build_gt.py` | the 2D ground truth: the dataset's proofread 3D reprojected into each camera (`outputs/bmimica/gt/<session>/<serial>/proofread.analysis.h5`) |
| `scripts/bartul/bmimica_build_sleap_ref.py`, `bmimica_bytetrack.py` | the per-camera SLEAP and ByteTrack baselines on Mouse-Dyad-10M |
| `scripts/bartul/bmimica_eval_crossview_all.py`, `cross_view_metric.py`, `eval_bartul.py` | cross-view identity scoring on Mouse-Dyad-10M |
| `scripts/bartul/muppet_run*.py`, `bmimica_muppet*.py` | the 3D-MuPPET baseline (an earlier comparison, not in the submitted figures) |
| `scripts/luc3d_track_all.mjs`, `scripts/bartul/luc3d_track_bartul.mjs` | headless LUC3D tracking drivers used by the early benchmark runs |
| `scripts/evaluate.py`, `scripts/sweep_bytetrack.py`, `scripts/lib/` | SLAP-2M scoring, the ByteTrack sweep, and the shared SLAP-2M paths and camera lists |
| `scripts/fig8/` | the earlier "Fig 8" experiments: LUC3D triangulators against Lightning Pose 3D, and the first s-DANNCE runs (the submitted s-DANNCE panel is `figs/fig10-bench/`) |

The rest (`lp3d/`, `udmt/`, `liezl/`, `reid/`, `track_c_reid/`, `sleap_nn/` and the
top-level sweeps) are the other-tool and identity experiments run in the same
workspace. They are kept so the record is complete. Some fed comparisons that are not in
the submitted figures.

The code imports its siblings by path (`from lib.slap2m_paths import ...`,
`from evaluate import ...`), so run it from `scripts/`.
