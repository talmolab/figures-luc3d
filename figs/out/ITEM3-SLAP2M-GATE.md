# Item 3 — the SLAP-2M harness gate

**Question.** My Fig 8/9 harness reproduces `figs/out/fig3_trackers.json`'s BMimica LUC3D
numbers to 16 digits but not its SLAP-2M ones: within-view IDF1 **0.7520** (mine) against
**0.7360** (`slap2m.within_view.luc3d.mean` = `0.7360353065988466`), a **+0.0160** gap of the
same order as the effect Fig 7 b–g would claim.

## VERDICT — (a). The exact difference is identified, and the reference is reproducible.

**`fig3_trackers.json`'s SLAP-2M LUC3D is a different tracker, not a different harness.** It
was produced by `matchFrameInstances` — the pre-module-refactor, pre-#131 **per-frame**
matcher — driven by `luc3d-bench/scripts/luc3d_track_all.mjs` against a **flat** LUCID
snapshot. My 0.7520 is the **shipped** `runCrossViewTracker` (`pose/cross-view-tracker.js`,
commit `625351a` "Implementation of Liezl's cross view tracker", **2026-07-06**) — merged
**seven weeks after** `outputs/luc3d_results/` was written (2026-05-15) and evaluated
(2026-05-19).

Everything else is identical, and that is measured rather than argued: **scoring the
reference's own outputs with my scorer reproduces `_eval_baseline.csv` to all sixteen digits
(`0.7360353065988466`), with a maximum per-session difference of `0.000e+00` across all 74
sessions.** Detection pool, scoring window, camera list and order, per-session animal counts,
and the aggregation convention are all bit-identical between the two harnesses.

The +0.0160 decomposes additively and completely:

```
tracker generation (pre-#131 matchFrameInstances -> shipped runCrossViewTracker):  +0.013563  (85%)
tail-node exclusion (my driver zeroes TailTip/Tail_0/Tail_1/Tail_2; the old code
  had no node weighting at all):                                                   +0.002425  (15%)
                                                                                   ---------
                                                                                   +0.015988
```

The reference number was the shipped tracker *as of May 15* (the driver's own comment:
"mirrors `trackAll()` in `index.html:16532`", and `lucid_flat/index.html:16575` does call
`matchFrameInstances` inside `trackAll()`). It is not the shipped tracker now. The anchor is
**stale, not wrong-at-the-time.**

### Reproduction

```bash
# 1. The reference arm, reproduced exactly (assignment-for-assignment):
cd /root/vast/eric/luc3d-bench
LUCID_FLAT_DIR=/root/vast/eric/luc3d-bench/outputs/bartul_smoke/lucid_flat \
  node scripts/_item3_track_all_flat.mjs \
    --session-idx 49 --num-animals 2 \
    --calibration /root/talmolab-smb/eric/slap_2m/2022-10-30/10302022035407/calibration.toml \
    --pred-h5-dir outputs/predictions_h5s --out /tmp/ref/10302022035407.json
# → 0 of 9,042 frames differ from outputs/luc3d_results/10302022035407.json
#   (`_item3_track_all_flat.mjs` is byte-identical to `luc3d_track_all.mjs` except that
#    LUCID_DIR is overridable by env, because `lucid -> the repo` no longer has flat
#    tracker.js/pose-data.js/triangulation.js/kalman.js at all.)

# 2. The 0.7360 mean, from the reference outputs, through MY scorer (all 74 sessions):
#    → n=74  mean=0.7360353065988466  max_abs_session_diff=0.000e+00
```

```python
# self-contained form of (2) — no scratch files needed. Run with the bench interpreter
# /root/vast/eric/luc3d-bench/liezl_env/bin/python (motmetrics + h5py).
import sys; sys.path.insert(0, "figs")
import fig3_score as fs, pandas as pd, numpy as np
from pathlib import Path
B = Path("/root/vast/eric/luc3d-bench/outputs")
CAMS = ["back", "backL", "mid", "midL", "top", "topL"]
df = pd.read_csv(B/"predictions_master_sheet.tsv", sep="\t", index_col=0).reset_index(drop=True)
vals = []
for i, row in df.iterrows():                      # i IS the pooled-H5 session axis index
    s = str(row["session"])
    r = fs.score_session(str(B/"luc3d_results"/f"{s}.json"), str(B/"predictions_h5s"),
                         None, CAMS, int(row["num_animals"]),
                         max_frames=int(row["frames"]), det_session_idx=int(i),
                         gt_paths={c: row[f"{c}_proofread_h5"] for c in CAMS})
    vals.append(r["within_idf1"])
print(f"{np.mean(vals):.16f}")                    # 0.7360353065988466
```

## Candidate causes, one at a time

### 1. `num_animals` and the target-count cap — RULED OUT

* Both harnesses read the same per-session animal count. `outputs/predictions_master_sheet.tsv`
  (reference) and `outputs/sleap_nn_master_sheet.tsv` (mine) agree on **session order**, and on
  `animals`, `num_animals`, `frames` and `calibration_toml` for **all 74 rows** — so
  `--session-idx`, `--num-animals` and the calibration path are the same values in both.
  (`animals == num_animals` within each sheet too.)
* Reference: `--num-animals` → `matchFrameInstances(..., {numAnimals})`, which caps groups per
  frame. Mine: `--num-animals` → `maxTargets` in `runCrossViewTracker`. Same value, analogous
  role.
* **`--no-cap` ablation** (session `10302022035407`, current tracker): within-view IDF1
  `0.3867890619411236`, cross `0.4006209624918624`, 284 switches — **bit-identical to the capped
  run in the deposit**. The cap is not binding on this session, so it cannot be carrying the gap.

### 2. `max_frames` clipping — RULED OUT (provably inert on this pool)

* Windows: reference `min(pool_frames, gt_frames)`; mine `min(gt, det, master frames)`.
* Checked all **74 × 6 = 444** (session, camera) pairs from HDF5 shapes alone: the extra
  master-`frames` clip changes the window in **0 of 444** cases. The pool is padded to 108,133
  frames, but each GT `.analysis.h5` is its own session's length and never exceeds the master
  sheet's `frames`.
* The padding is also inert for *tracking*: it is all-NaN, so `usedFrames` skips it. The
  reference driver's own log line for session 49 reads `frames-with-data=9042/108133` — exactly
  the window my `--max-frames 9042` produces.
* Strongest form: `_eval_baseline.csv` records motmetrics' own `num_frames` per camera-session.
  It equals `min(gt_frames, pool_frames, master frames)` — my window — for **444 of 444**
  camera-sessions. The reference scored exactly the frames I score, by its own record.

### 3. Camera list and order — RULED OUT

Both drivers default to `back,backL,mid,midL,top,topL`; the reference run passed no
`--cameras`, mine passes that exact list. Scoring uses `BENCH_CAMERAS =
("back","backL","mid","midL","top","topL")` in both. Order therefore cannot differ, and the
per-camera IDF1 values line up name-for-name (see Candidate 5).

### 4. Tail-node exclusion — REAL, but second-order (±0.01), and not the cause

* The reference **could not** exclude tail nodes: `lucid_flat/tracker.js` has no `nodeWeight`
  function at all, and the driver builds `new Session(..., {nodes: []}, ...)` — every keypoint
  weighted 1. My driver zeroes `TailTip`, `Tail_0`, `Tail_1`, `Tail_2` unless
  `--no-exclude-tail`.
* Ablation with the **current** tracker, same sessions, same scorer:

  | session | animals | tail EXCLUDED (mine) | tail INCLUDED | reference (old tracker) |
  |---|---|---|---|---|
  | `10192022181152` | 2 | **0.9088** (50 sw) | 0.8945 (111 sw) | 0.6147 (82 sw) |
  | `10302022035407` | 2 | **0.3868** (284 sw) | 0.3946 (366 sw) | 0.5563 (108 sw) |

  Tail exclusion moves the number by −0.008 to +0.014 and does not have a fixed sign; the
  residual distance to the reference on these two sessions is **+0.294** and **−0.170**.
* At corpus scale (all 74 sessions, both arms) tail exclusion is worth **+0.0024** of the
  +0.0160 — 15%. See "Corpus-scale tail arm" below for the exact additive decomposition.

### 5. The scorer — RULED OUT, bit-exact

I scored the **reference's own** `outputs/luc3d_results/*.json` with `figs/fig3_score.py`
(clipped exactly as `fig9_slap2m.py` does) and compared per-camera `idf1` against
`_eval_baseline.csv`:

| session | animals | mine (mean of 6 cams) | `_eval_baseline.csv` | max abs per-camera diff |
|---|---|---|---|---|
| `10192022181152` | 2 | 0.614703612423905 | 0.614703612423905 | **0.0** |
| `10302022035407` | 2 | 0.5563005919121208 | 0.5563005919121208 | **0.0** |
| `10192022175248` | 1 | 0.9987762431636659 | 0.9987762431636659 | **0.0** |

**All 74 sessions**, `figs/fig3_score.py` over `outputs/luc3d_results/`:

```
n=74  mean_mine=0.7360353065988466  mean_ref=0.7360353065988466  max_abs_session_diff=0.000e+00
```

That is `fig3_trackers.json`'s `slap2m.within_view.luc3d.mean` to all sixteen digits, with zero
per-session deviation across the corpus. **The scorer accounts for none of the 0.0160.**

The conventions coincide for concrete reasons, not by luck: IoU gate 0.5 in both
(`evaluate.py` passes `max_iou = 1.0 - 0.5`, `fig3_score.py` passes `max_iou=0.5`); both drop
`ident < 0` detections from the prediction list and keep the GT box either way; both take the
detection count from the pool's slot axis; and both compute within-view IDF1 as the mean over
a session's six cameras first and the mean over sessions second (`fig3_trackers.slap2m()`'s
`sess_mean` then `summarise`, versus `score_session`'s `np.mean(percam_idf1)` then `agg`).
`auto_id=True` versus an explicit `frameid=fi` makes no difference because neither loop skips
frames — and the 0.0 differences settle that empirically.

### 6. The detection pool — RULED OUT, same file

Both read `outputs/predictions_h5s/{cam}_predictions.h5`, shape `(74, 108133, 4, 15, 2)`. The
proof is stronger than a path comparison: re-running the reference driver over that pool
reproduced the stored `luc3d_results` output **assignment-for-assignment** (0 of 9,042 frames
differ), so that pool is what produced the stored file.

## What actually differs

| | reference (`fig3_trackers.json`, Fig 7 b–g) | mine (`fig9_slap2m_predictions.json`) |
|---|---|---|
| driver | `luc3d-bench/scripts/luc3d_track_all.mjs` | `figs/fig8-bench/fig8_bench.mjs` |
| LUCID code | **flat, pre-refactor** snapshot (`outputs/bartul_smoke/lucid_flat/`) | current `pose/` modules |
| entry point | `matchFrameInstances()` **per frame**, with `prevAssignments` / `prevTargets3d` | `runCrossViewTracker()` (`CrossViewTracker` + `Target` objects) |
| tracker vintage | pre-#131 (shipped as of 2026-05-15) | post-#131 (`625351a`, 2026-07-06) |
| skeleton passed | `{nodes: []}` → no node weighting possible | `{nodes: NODE_NAMES}` → tail nodes weighted 0 |
| `Session` tracks | `nAnimals` from the pool's slot axis (4) | `numAnimals` from the master sheet |
| output schema | carries per-frame `targets3d` | no `targets3d`; carries `excludeTail`, `capped`, `params` |
| run date | 2026-05-15 (eval 2026-05-19) | 2026-08-13 |

Supporting facts:

* Root `tracker.js`/`pose-data.js`/`triangulation.js` were **deleted from LUCID on 2026-05-06**
  (`fbec22f`, "Refactor: split `index.html` into ES module tree (#81)"). `luc3d_track_all.mjs`
  reads exactly those flat filenames from `LUCID_DIR = luc3d-bench/lucid`, which is a symlink to
  the repo — so **the reference driver cannot run against the repo today**. The snapshot it needs
  survives at `outputs/bartul_smoke/lucid_flat/` (671-line `tracker.js`, `matchFrameInstances` at
  line 233, **no** `CrossViewTracker`, **no** `nodeWeight`; plus bench-only temporal-PAF additions
  that stay inert without `--params`, which the reference run did not pass).
* The reference JSONs carry per-frame `targets3d` and no `params`/`excludeTail`/`capped` keys —
  the output schema only `luc3d_track_all.mjs` writes.
* **My arm is genuinely the shipped tracker.** `fig8_bench.mjs` with `method: {}` produced output
  **bit-identical** (0 of 8,996 frames differing, same identity set) to `figs/fig3-bench/fig3_bench.mjs`,
  which imports the real `pose/cross-view-tracker.js` through `scripts/bench/hooks.mjs`. The app's
  "Track All" uses the async sibling of `runCrossViewTracker` with, per its own docstring,
  "identical association".

## The gap has the signature of an association change, not a harness change

Split by animal count — the single strongest piece of evidence in this report:

| stratum | n | mine | reference | mean Δ | mean \|Δ\| | max \|Δ\| |
|---|---|---|---|---|---|---|
| **1 animal** (nothing to associate) | 32 | 0.8279 | 0.8270 | +0.0009 | **0.0010** | 0.0035 |
| **2–4 animals** | 42 | 0.6942 | 0.6667 | +0.0275 | **0.0549** | 0.2941 |

A harness difference — scoring window, detection pool, camera set, animal count, tail weights —
would perturb the single-animal sessions too. They agree to a thousandth (three of them
bit-exactly, the only exact matches in the corpus), while the multi-animal sessions move by up to
0.29. The difference lives entirely in cross-view/temporal **association**.

Also: detector recall is 0.7268 (mine) against 0.7285 (reference) — the same detections, as it
must be for a shared pool. Fragmentations, by contrast, are a tracker property and move a lot:
**10.4** per camera-session (mine) against **28.1** (reference).

### Which tracker is actually better, per stratum (this changes what Fig 7 would say)

Within-view IDF1 (mean over sessions) and total within-view switches, on the identical pool and
scorer. `winner` = the Fig 8 fresh-anchor config (`sync`, `stale 20`, `distanceThreshold 25`).

| animals | n | reference (pre-#131) | shipped (post-#131) | fresh anchor | ref sw | shipped sw | fresh sw |
|---|---|---|---|---|---|---|---|
| 1 | 32 | 0.8270 | 0.8279 | 0.8279 | 0 | 0 | 0 |
| 2 | 35 | 0.7293 | **0.7691** | **0.7829** | 3,206 | 1,744 | **1,113** |
| 3 | 4 | **0.3728** | 0.3512 | 0.3604 | 205 | **744** | 54 |
| 4 | 3 | **0.3292** | 0.2774 | 0.3162 | 299 | **606** | 145 |
| ALL | 74 | 0.7360 | 0.7520 | 0.7606 | 3,710 | 3,094 | 1,312 |

Two things follow, and neither should be averaged away:

1. The +0.0160 corpus gain is **entirely** the 35 two-animal sessions. On the 7 sessions with
   3–4 animals the **pre-#131 tracker is better** (0.3728/0.3292 against 0.3512/0.2774) and the
   shipped tracker's switch count **blows up** — 744 against 205, and 606 against 299. That is a
   real, reproducible regression in the shipped tracker on ≥3 animals, on a pool where the
   detector already misses a third of GT.
2. The Fig 8 fresh anchor removes most of that blow-up (744 → 54, 606 → 145) and still does not
   quite recover the old tracker's IDF1 on 3 animals (0.3604 vs 0.3728). So "the fresh anchor
   fixes the ≥3-animal regression" is true of switches and only partly true of IDF1.

## The gap is not a small systematic offset

Per session the difference ranges **−0.170 to +0.294**; only **3 of 74** sessions agree to 1e-12.
Switch totals differ the same way: **3,710** within-view switches (reference, summed over
camera-sessions) versus **3,094** (mine). Biggest movers, mine vs reference:
`10072022143153` 679 vs 44, `10072022151549` 500 vs 178, `10212022175626` 83 vs 277,
`10072022184618` 50 vs 234. Both arms have 0 switches on all 32 single-animal sessions.

Cross-view IDF1 **cannot** be compared against the shipped figure at all: `_eval_baseline.csv`
has no cross-view columns (`fig3_trackers.py` says so and sources every cross-view number from
BMimica). I therefore measured it from the reference's own outputs with my scorer — 0.7494 over
74 sessions, 0.6777 over the 42 multi-animal ones, against the shipped tracker's 0.7648 / 0.7040.
So the third leg of item 3's verification standard (within IDF1, cross IDF1, total switches) is
now measurable on both arms, and all three move the same way and for the same reason.

## Corollary, larger than the gate itself: Fig 7 mixes two tracker generations

BMimica's LUC3D series (`outputs/bmimica/bmimica_crossview_all_eval.csv`, written **2026-07-21**)
came from `scripts/bench/bench_crossview.mjs`, which calls **`runCrossViewTracker`** with
`excludeTail` on and `maxTargets = num_animals` — i.e. the same configuration as my SLAP-2M arm.
So Fig 7 as shipped labels **two different trackers** "LUC3D": post-#131 on BMimica, pre-#131 on
SLAP-2M. That is exactly the transcription-versus-measurement failure mode item 3 was written to
catch, and it also explains why the BMimica gate passed to 16 digits while the SLAP-2M gate could
not.

## Consequence for Fig 7

A clean SLAP-2M substitution **is** possible, under one condition: the SLAP-2M LUC3D anchor must
be **replaced wholesale**, never mixed. Specifically —

* The `shipped` arm of `figs/out/fig9_slap2m_predictions.json` (within 0.7520, cross 0.7648,
  3,094 switches over 74 sessions; 0.6942 / 0.7040 / 3,094 over the 42 multi-animal sessions) is
  the honest current-code baseline on the pool Fig 7 b–g uses, and it supersedes 0.7360 rather
  than contradicting it.
* Any SLAP-2M panel that keeps a 0.7360-derived number while showing a post-#131 LUC3D arm is
  comparing two tracker generations. Same for the SLEAP/ByteTrack columns of that CSV: they are
  unaffected by the LUCID change (they do not use LUCID's tracker), so they can stay — but the
  LUC3D column cannot.
* The 0.7704 "identity-fix ceiling" quoted for this pool is a property of the **detections**, not
  of the tracker, so it survives the substitution.
* **Mechanically**, `fig3_trackers.slap2m()` consumes far more than a mean: paired
  LUC3D−SLEAP differences, `by_animals` (camera-session weighted), `by_bedding`, the error
  decomposition (`num_false_positives` / `num_misses` / `num_switches` / `num_objects`),
  `recall_correlation.per_session` (a `[recall, luc3d, sleap, animals, bytetrack]` row per
  session), camera-session argmax/pairwise win counts, and survival curves. All of those need
  LUC3D at **per-(session, camera)** granularity, which `fig9_slap2m_predictions.json` does not
  carry (it stores `per_camera_idf1` but session-level sums for the counts).
  The clean route is to re-score the fresh arm with luc3d-bench's **own** scorer so the output is
  a drop-in replacement for the `luc3d` rows of `_eval_baseline.csv`: `evaluate.eval_camera(...,
  no_sleap=True)` over `figs/out/tmp/fig9slap_predictions/shipped/` (whose JSON schema
  `luc3d_assignments_for_cam` already accepts). That leaves the `sleap` and `bytetrack` rows
  untouched — they do not use LUCID's tracker — and changes exactly one column.
  **Do not** call `evaluate.main()` to do it: that overwrites
  `luc3d-bench/outputs/metrics/by_difficulty.csv`, a historical artefact `fig3_trackers.py`'s
  provenance note reasons about. Use `eval_camera` directly.
  Status of that artefact: see "Per-camera-session CSV" below.

Not reconciled to bit-identical, because that is impossible: the two arms are different
algorithms. Per item 3's own verification standard, this document is the "explicit written
account of why not".

## Corpus-scale results

All three arms on the same pool (`outputs/predictions_h5s`), the same window, and the same scorer
(`figs/fig3_score.py`). Cross-view IDF1 does not exist in `_eval_baseline.csv`, so the reference
row's cross-view is measured here from the reference's own per-frame outputs — the first time that
number exists for this corpus.

| arm | tracker | within IDF1 (74) | cross IDF1 (74) | within sw | within IDF1 (42 multi) | cross IDF1 (42) |
|---|---|---|---|---|---|---|
| reference = Fig 7 b–g today | pre-#131 `matchFrameInstances` | **0.7360353065988466** | 0.7494 | 3,710 | 0.6667 | 0.6777 |
| shipped (mine) | post-#131 `runCrossViewTracker` | **0.7520235409209860** | 0.7648 | 3,094 | 0.6942 | 0.7040 |
| fresh anchor | + `sync`, `stale 20`, `dist 25` | 0.7606112337840365 | 0.7745 | 1,312 | 0.7093 | 0.7212 |

Detector recall is 0.7285 / 0.7268 / 0.7270 across the three arms — the same detections, as it
must be. Fragmentations per camera-session: 28.11 (reference), 10.37 (shipped), 7.33 (fresh).

### Corpus-scale tail arm

The current tracker re-run over **all 74 sessions with `--no-exclude-tail`** — i.e. with the
reference's own node treatment, every keypoint weighted 1 — isolates the two contributions and
closes the gap exactly:

| arm | within IDF1 (74) | cross IDF1 | within sw |
|---|---|---|---|
| reference: **old** tracker, all nodes | 0.7360353065988466 | 0.7494 | 3,710 |
| **new** tracker, all nodes (`--no-exclude-tail`) | 0.7495986772099323 | 0.7629 | 3,455 |
| **new** tracker, tail excluded (my shipped arm) | 0.7520235409209860 | 0.7648 | 3,094 |

```
tracker generation (old -> new, node treatment held fixed):  +0.013563   84.8% of the gap
tail-node exclusion (new tracker, tail 1 -> 0):              +0.002425   15.2% of the gap
                                                              ---------
total                                                         +0.015988 = the 0.0160 observed
```

Nothing is left unexplained. The tail term is also a wash per session rather than a uniform
gain — 26 sessions better, 21 worse, 27 unchanged, worst −0.0917, best +0.1115, and **exactly
0.0000 on all 32 single-animal sessions** — which is the signature of a cost-function tweak, not
of a scoring artefact.

### Per-camera-session CSV — the substitution artefact, and the equivalence proved the other way

`figs/data/fig7/slap2m_luc3d_shipped_percam_ITEM3.csv` — the **fresh shipped arm** scored by
**luc3d-bench's own scorer** (`evaluate.eval_camera`, `no_sleap=True`, no ByteTrack), 444 rows of
74 sessions × 6 cameras, header **byte-identical** to `_eval_baseline.csv`. Drop-in replacement
for that file's `luc3d` rows; the `sleap` and `bytetrack` rows are unaffected by the LUCID change
and can be carried over unmodified.

It also closes the harness equivalence from the opposite direction:

| | scorer | within IDF1 |
|---|---|---|
| reference arm, my scorer | `figs/fig3_score.py` | 0.7360353065988466 = `_eval_baseline.csv` |
| fresh arm, bench scorer | `evaluate.eval_camera` | 0.7520235409209860 = `fig9_slap2m_predictions.json` |

Each scorer reproduces the *other* harness's published number on the *other* harness's tracker
output, to all sixteen digits. Switches 3,094, fragmentations 10.3671 per camera-session, recall
0.726828 — identical to the `fig3_score.py` values. There is no scorer difference left to hide in.

## Files

* Report: `figs/out/ITEM3-SLAP2M-GATE.md` (this file).
* New deposit: `figs/data/fig7/slap2m_luc3d_shipped_percam_ITEM3.csv` (see above).
* New file written outside the repo: `luc3d-bench/scripts/_item3_track_all_flat.mjs` — a copy of
  `luc3d_track_all.mjs` whose only change is `LUCID_DIR` being overridable via
  `$LUCID_FLAT_DIR`, needed because the original hard-codes a path that no longer contains the
  flat modules. Nothing under `luc3d-bench/outputs/` was modified.
* Working artefacts (scratch, not deposits): reference rescoring + ablation runs under
  the session scratchpad; nothing under `figs/out/` was touched except this report.
