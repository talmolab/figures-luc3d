# Item 6 — audit of the remaining competitor baselines (2026-08-13)

Scope: FIX-PLAN-8-13.md item 6 only. ByteTrack on BMimica, SLAP-2M's SLEAP series,
3D-MuPPET on BMimica, and the 1/C = 0.20 ceiling claim in
`figs/panels/fig7_05_within_vs_cross.py`. Item 1's SLEAP correction is being run
separately and is NOT duplicated here — but the trace below contradicts its stated
mechanism, so §0 records what the SLEAP number actually is.

Everything measured with `/root/vast/eric/luc3d-bench/liezl_env/bin/python`, 4 worker
processes, first 20,000 frames per camera-session unless stated. Sessions are
180,200 frames long, so "first 20,000" is a 11% window, chosen for CPU discipline;
whole-session quantities (instance counts, track-id counts, coverage) were computed
over the FULL session because they are cheap.

## The coherence test used throughout

Per identity label (slot or track id) per camera-session: keypoint centroid
displacement between consecutive frames in which that label is present in both
frames; p95 over all such displacements. "Catastrophic jump" = a label displacing
further in one frame than the two GT animals are apart in that frame (proofread GT
centroid separation). Reported as the fraction of frames containing at least one
such jump. Script: `scratchpad/coherence.py` (temporary, not committed).

BMimica, all 50 sessions x 5 cameras, first 20,000 frames, median GT separation
175.6 px:

| identity source | file | p95 displacement | frames with a jump > GT separation | detections present per frame |
|---|---|---|---|---|
| detection-pool **slot index** | `outputs/bmimica/det_h5/` | **313.1 px** | **9.95%** | 1.790 |
| sleap-nn's own **track id** | `outputs/bmimica/sleap_h5/` | **4.0 px** | **0.01%** | 0.878 |
| **ByteTrack** track id | `outputs/bmimica/results/bytetrack/` | **4.2 px** | **0.01%** | 1.784 |
| **3D-MuPPET** global id | `outputs/bmimica/results/muppet/` | **3.4 px** | **0.17%** | 0.102 |

The det-pool row reproduces the 498 px / 8.78% figure class from the item-1 finding
(the differences are the 20,000-frame window and the resulting lower median GT
separation, 175.6 vs 322 px). SLAP-2M pool for comparison, 6 multi-animal sessions x
6 cameras (`scratchpad/coh_slap2m.py`): **p95 18.4 px** (max 33.6), **0.09%**
catastrophic, median nearest-pair GT separation 165.8 px — coherent, matching the
27 px / 0.00% figure in the plan.

---

## 0. What Fig 7a's SLEAP number actually is (item 1's premise is wrong)

Not part of my assignment, but it changes item 1's fix, so it is recorded here.

**`bmimica_crossview_all_eval.csv`'s `sleap_idf1` does NOT come from the
detections-only pool.** `scripts/bartul/bmimica_eval_crossview_all.py` reads
`SLEAP = outputs/bmimica/sleap_h5` for the sleap arm (`sleap_det[c]`), and only
LUC3D / MuPPET / ByteTrack read `det_h5`. `sleap_h5` is built by
`scripts/bartul/bmimica_build_sleap_ref.py` from the **retracked** `.slp` sliced by
`inst.track` — sleap-nn's own tracker identity. Measured above: p95 4.0 px, 0.01%
catastrophic. It is a genuinely tracked source. The older Jul 3 eval
(`bmimica_eval_per_session.csv`) has **no sleap column at all**, i.e. the
`--no-sleap` caution in `evaluate.py` *was* respected on BMimica; the sleap arm was
added on Jul 21 from the tracked file. Timestamps agree: `sleap_h5` 05:06, CSV 07:03,
aggregate 07:04, all Jul 21.

**The real defect is a truncation, not a slot index.** `bmimica_build_sleap_ref.py`
allocates `N_ANIMALS = 2` slots and keeps an instance only `if ti < N_ANIMALS`, where
`ti` indexes `L.tracks`. sleap-nn's `--tracking_target_instance_count 2` caps
instances **per frame**, not the number of tracks in the file. Measured over all 250
retracked `.slp` files (instance table read directly):

- distinct sleap-nn tracks per camera-session: min 5, **median 47**, max 309
- instances on tracks 0/1: **7,917,132 of 79,267,632 = 10.0%** overall; per
  camera-session retained fraction min 0.0001, p25 0.0112, **median 0.0366**, p75
  0.1207, max 0.9902

So 90% of SLEAP's own tracked detections were discarded before scoring. The 0.1154
is a recall artefact of that slicing, not a property of SLEAP's tracker.

**Warning for item 1's sanity band.** With ALL track ids retained, IDF1 is still hard
bounded: with 2 GT animals the optimal 1-1 assignment can match at most the two
largest track ids, so `IDF1 <= 2 * (mass of top-2 ids) / (N_gt + N_pred)`. Measured
per camera-session and averaged per session over all 50 sessions:
**mean ceiling 0.2702**, median 0.2501, min 0.100, max 0.549; 9/50 sessions above
0.40. The honest BMimica SLEAP within-view mean therefore lands near **0.25, not
0.66** — FIX-PLAN item 1's sanity band ("should land in the same regime as SLAP-2M's
0.66") is not attainable on this corpus, and a corrected corpus mean of ~0.2–0.3 is
the expected result, not a sign the fix failed. The 0.3996 quoted from one session is
consistent with one of the 9 high-concentration sessions and must not be read as the
corpus mean.

**Verdict: INVALID (confirmed), different mechanism than stated.** The fix is not
"score `retracked/` instead of the pool" — the pool was never the sleap source. The
fix is "score `retracked/` keeping ALL track ids instead of only the first two".

---

## 1. ByteTrack on BMimica (within 0.1574, cross 0.0457) — **VALID**

**Traced.** `figs/fig3_trackers.py` reads `bmimica_crossview_all_eval.csv` columns
`byte_idf1` / `byte_xview_idf1` into `bmimica_50_sessions["ByteTrack"]`.
`bmimica_eval_crossview_all.py` reads `outputs/bmimica/results/bytetrack/{sid}/{cam}.h5`,
column 0 of each row as the predicted id (`pl.append(int(row[0]))`) and columns 1:5 as
the box — i.e. **ByteTrack's own track id**, not a slot. Those files were written by
`scripts/bartul/bmimica_bytetrack.py` -> `run_bytetrack_bench.run_camera_session` over
the shared det pool (`bytetrack_run.log`: `DONE bytetrack 250/250`). Independently
cross-checked: the older Jul 3 `bmimica_eval_per_session.csv`, a different script,
reports the identical `byte_idf1` for a shared session (0.10528784907650009), so the
number reproduces across two evaluators.

**Coherence.** p95 frame-to-frame displacement **4.2 px**; **0.01%** of frames contain
a jump longer than the inter-animal distance — i.e. two orders of magnitude better
than the det-pool slot's 313 px / 9.95%, and better than the SLAP-2M pool that the
plan calls coherent (18.4 px / 0.09%). Coverage 1.784 detections/frame against the
pool's 1.790, so ByteTrack labels essentially every detection it is given.

**Why the score is nonetheless low, and why that is a real result.** ByteTrack does
not reuse ids after a gap. Over the full 180,200-frame sessions it emits a median of
**85 distinct ids per camera-session** (p25 57, p75 116, max 267), and its two largest
ids hold a median of **18.7%** of its detections. The resulting arithmetic ceiling
(top-2 mass / detections, as above) is **0.1903** mean over the 50 sessions; the
reported 0.1574 is 83% of it. The measurement is therefore fully explained by
long-horizon id fragmentation of a coherent tracker, which is what the panel claims.
Corroborated two ways: on a 20,000-frame window the same files score within-view IDF1
**0.7212 / 0.5397 / 0.3888 / 0.6040** on four sessions (mean 0.563) against a
full-session 0.157; and the same ByteTrack on SLAP-2M, whose sessions are 10x shorter
(median **18,138** frames vs 180,200), scores 0.5274.

**Caveat that should be stated, not a defect.** LUC3D's arm is scored from a global-id
JSON constrained to `numAnimals = 2` and so cannot fragment by construction, while
ByteTrack (and SLEAP) allocate ids freely. On BMimica's 30-minute horizon that
asymmetry, not per-frame association quality, is the dominant driver of 0.749 vs
0.157. The 20,000-frame window numbers above are the cleanest evidence.

---

## 2. SLAP-2M's SLEAP series (within-view mean 0.6614) — **VALID**

**Traced to file and column.** `figs/fig3_trackers.py` `SLAP2M =
outputs/PAF_3d_kalman/_eval_baseline.csv`; rows with `tracker == "sleap"` (444 rows =
74 sessions x 6 cameras), column `idf1`, per-camera rows averaged per session then
over sessions. Recomputed: **0.661392** (luc3d 0.736035, bytetrack 0.527389) — matches
the deposit's 0.6614.

**Was the source tracked? Yes.** Those rows come from `scripts/evaluate.py`'s
`acc_sleap`, whose identity is the slot `a` of
`outputs/predictions_h5s/{cam}_predictions.h5`. That pool is built by
`scripts/aggregate_predictions.py`, which calls `sio.load_file(...).numpy()` on each
raw `.predictions.slp` — `Labels.numpy()`'s animal axis **is** the track axis, so
slot = SLEAP's own track id from inference. Confirmed directly: the raw
`.predictions.slp` files carry populated `instances['track']` columns (e.g. session
10072022131531/back: 2 tracks holding 17,552 and 18,036 instances).

**Coherence.** p95 displacement **18.4 px** (max over 36 camera-sessions 33.6 px),
**0.09%** of frames with a jump beyond the nearest-pair GT separation. Coherent;
reproduces the plan's 27 px / 0.00%.

**Two caveats worth putting on the record (neither invalidates the number).**
1. The pool truncates to `max_animals = 4` (`t_end = min(nt, max_animals)`), so
   instances on tracks with index >= 4 never enter the pool. On sampled
   camera-sessions the retained instance fraction is 1.00, 1.00, 1.00, 1.00, 0.357,
   0.557, 0.024, 0.023. This is a **shared** loss — every SLAP-2M tracker reads the
   same pool — and it surfaces as the already-documented detector recall (session-mean
   0.729–0.737 across trackers) and the r = 0.975 recall/IDF1 coupling caveat.
2. Consequence for cross-corpus reading: because the SLAP-2M pool is *indexed by
   SLEAP's tracks*, SLEAP's fragments are outside the pool and cost it nothing beyond
   shared recall; on BMimica the pool is re-slotted by score and holds every instance,
   so fragments are inside the other trackers' pool. The two corpora therefore do NOT
   treat SLEAP the same way, and SLAP-2M's 0.661 must not be used as the expected band
   for a corrected BMimica SLEAP (see §0's 0.2702 ceiling).

---

## 3. 3D-MuPPET on BMimica (0.0112 / 0.0112) — **INVALID as an identity measurement**

**Traced.** `bmimica_eval_crossview_all.py` takes MuPPET's identity from
`outputs/bmimica/results/muppet/{sid}.json` via `ev.luc3d_assignments_for_cam`, applied
to the shared det pool. Frames absent from that JSON get identity −1 and are dropped
from the prediction set entirely — they become misses.

**The JSONs cover almost none of the session.** All 50 files, frames with at least one
assignment divided by session length: min **0.17%**, **median 1.31%**, max 7.22%.
Frame indices start at 0 and stop after a few thousand (e.g. 20250827_141755: 4,478
frames emitted, max frame index 4,497, session length 180,200).

**Mechanism, from `scripts/bartul/muppet_run.py`.** `global_by_track` is built **once**,
at the init frame, mapping SORT track id -> global id per camera. Per-frame emission
does `gid = global_by_track[c].get(tid)` and `continue`s when it is missing. SORT runs
with `max_age = 10`, so as soon as a tracklet dies its successor gets a fresh id that
is in no map, and that camera goes silent permanently. Within a few thousand frames
every init-frame tracklet has died in every camera.

**Conditional score — the decisive measurement.** Re-scoring exactly as the shipped
scorer does, but restricted to the frames where MuPPET emits anything (first 20,000
frames, 4 sessions; `scratchpad/muppet_cond.py`):

| session | emitted | full within / cross | conditional within / cross |
|---|---|---|---|
| 20250829_155407 | 7.60% | 0.0776 / 0.0780 | **0.6718 / 0.6964** |
| 20250904_153332 | 20.72% | 0.1313 / 0.1334 | **0.4839 / 0.5060** |
| 20250827_141755 | 22.39% | 0.1388 / 0.1418 | **0.4747 / 0.5084** |
| 20250905_200953 | 21.55% | 0.0561 / 0.0588 | **0.2082 / 0.2467** |

Identity is also coherent where present (p95 3.4 px). So 0.0112 is a **coverage**
number: it measures that the port's one-shot cross-view initialisation does not
survive a 30-minute recording, not that its identities are wrong. Stating "3D-MuPPET
IDF1 0.011" alongside trackers that predict on every frame compares a coverage
collapse to an association quality.

**Verdict: INVALID as stated.** Either re-run with periodic re-initialisation (or
re-linking of new SORT ids) and re-score, or state the finding as what it is: "MuPPET's
first-frame cross-view matching yields no identity at all for ~98.7% of a 180,200-frame
session (median coverage 1.3%); on the frames it does label, within-view IDF1 is
0.21–0.67." Flatness from within- to cross-view (0.0112 -> 0.0112) is meaningless at
that coverage and should not be offered as evidence of cross-view consistency.

---

## 4. The 1/C = 0.20 ceiling in `fig7_05_within_vs_cross.py` — claim does **NOT** survive

The docstring's argument: "A tracker that labels each camera independently can have its
labelling matched to the truth in at most ONE camera; the other C − 1 cameras carry
labels that cannot simultaneously be right. So 1/C = 0.20 is the BEST such a tracker
could score." Two independent problems.

**(a) The bound is a property of the SCORING CONVENTION, not of per-camera trackers.**
`bmimica_eval_crossview_all.py` deliberately makes each camera's labels disjoint before
pooling: `pg = ci * 10 + a` for SLEAP and `pg = ci * 100000 + tid` for ByteTrack. Under
that construction a label physically cannot appear in two cameras, which is what makes
the argument go through. `figs/fig3_score.py` — the scorer the correction is going
through — pools the tracker's own id **unscoped** as a global id
(`pooled.update(gti, pid_global, ...)`). Then `track_0` in camera A and `track_0` in
camera B are the SAME label and can be matched in all C cameras at once, and 1/C bounds
nothing. Measured on the same files both ways (`scratchpad/xview_scoping.py`, 4
sessions, first 20,000 frames, C = 5):

| source | session | within | cross, camera-scoped | cross, shared numbering |
|---|---|---|---|---|
| SLEAP (`sleap_h5`) | 20250829_155407 | 0.2292 | 0.0727 | **0.2243** |
| SLEAP | 20250905_200953 | 0.5275 | **0.2022** | **0.4980** |
| SLEAP | 20250904_153332 | 0.6982 | **0.2151** | **0.6227** |
| SLEAP | 20250827_141755 | 0.8059 | 0.1992 | **0.5304** |
| ByteTrack | 20250827_141755 | 0.7212 | 0.1969 | **0.3444** |
| ByteTrack | 20250904_153332 | 0.5397 | 0.1333 | **0.4933** |
| ByteTrack | 20250829_155407 | 0.3888 | 0.1150 | 0.1807 |
| ByteTrack | 20250905_200953 | 0.6040 | 0.1614 | **0.3044** |
| means | | 0.565 / 0.563 | 0.172 / 0.152 | **0.469 / 0.331** |

Shared numbering exceeds 0.20 in 7 of 8 cases and reaches 0.6227 — 3.1x the "ceiling".
Per-camera track numbering coincides across cameras far better than the argument
assumes, exactly as the item-6 brief suspected: both trackers allocate ids in order of
first appearance, so id 0/1 tend to be the same two animals in every view.

**(b) Even camera-scoped, 1/C is an approximation, not a bound.**
IDF1 = 2·IDTP/(N_gt + N_pred). "One camera's worth of IDTP" only equals 1/C of the
denominator when the tracker's predicted mass is spread evenly over cameras and roughly
equals GT mass. When a per-camera tracker's predictions are sparse or unevenly
distributed, N_pred shrinks the denominator while IDTP concentrates, and the pooled
score rises above 1/C. Two of the four scoped SLEAP sessions above do exactly that:
**0.2022** and **0.2151**, on the shipped scorer's own convention.

**What survives.** The empirical fact that Fig 7a's per-camera series sit at
0.046–0.062 while LUC3D sits at 0.749, and the correct explanation that a per-camera
tracker has no mechanism to make its labels mean the same thing in two views. What does
not survive is calling 0.20 a *ceiling*, and the docstring's claim that landing below it
is "the stronger claim".

**Correct statement to use instead** (wording, not artwork — I did not touch the panel):
"Cross-view IDF1 is scored here with each camera's labels kept in a separate label
space, so a per-camera tracker's score is *approximately* limited to the share of the
pooled ground truth that one camera contributes, ~1/C = 0.20 at C = 5. It is an
approximate reference line, not a hard bound: a tracker whose predictions are sparse or
concentrated in a few cameras can exceed it (measured up to 0.215 here), and if a
tracker's raw id numbering happens to agree across cameras — scored without
camera-scoping — there is no bound at all (measured up to 0.62)."

Corollary the correction must handle: **0.2237 and 0.20 are not comparable numbers.**
The corrected SLEAP cross-view comes from `fig3_score.py`'s unscoped pooling while the
0.062 it replaces came from camera-scoped pooling. Either re-score the corrected SLEAP
with camera scoping (keeping the panel's series internally consistent and its reference
line meaningful) or drop the 0.20 rule from the panel — do not plot an unscoped number
against a scoped rule.

---

## Manuscript-figure claims that need changing

1. **Fig 7a — SLEAP per-camera series (0.115 within / 0.062 cross).** INVALID.
   Mechanism is the 2-slot truncation in `bmimica_build_sleap_ref.py` (10.0% of
   sleap-nn's tracked detections retained), not a slot-as-identity error. Expect the
   corrected corpus mean near 0.25, ceiling 0.2702 — not 0.66.
2. **Fig 7a — 3D-MuPPET series (0.011 / 0.011).** INVALID as an identity measurement:
   median 1.31% frame coverage; conditional within-view IDF1 0.21–0.67. Also invalidates
   the panel's "3D-MuPPET is flat but at 0.011 — height and flatness read together"
   sentence and the matching `caveats` entry in `figs/out/fig3_trackers.json`.
3. **Fig 7a — the 0.20 rule and its on-artwork label** ("1/C = 0.20: ceiling for a
   per-camera tracker (SLEAP, ByteTrack)"), plus the identical `caveats` entry in
   `figs/fig3_trackers.py` ("bounded near 1/C ... land far BELOW that bound ... a
   ceiling on what they could achieve"). Not a bound; see §4 for replacement wording.
4. **Fig 7a — ByteTrack series (0.157 / 0.046).** VALID, keep the numbers, but the
   figure should say the comparison is over 180,200-frame sessions where ByteTrack
   emits ~85 ids per camera (its own top-2-id ceiling is 0.190), and that LUC3D's arm is
   constrained to 2 global ids. Same files score 0.56 within-view on a 20,000-frame
   window.
5. **Deposit-only, not on artwork:** `bmimica_50_sessions["SLEAP per-camera"].switches_2d_total
   = 856` is invalid for the same truncation reason (few labels, 10% of detections, so
   few switches are even possible). It is not plotted in 7a — do not start plotting it.
6. **Unaffected:** every SLAP-2M panel (Fig 7b–g: `fig7_01_survival.py`,
   `fig7_03_error_decomposition.py`, `fig7_04_trackers.py`, `fig7_06_bedding.py`,
   `fig7_07_recall.py`) — their SLEAP and ByteTrack series trace to a coherent tracked
   source (§2). `fig8_05_all50.py` reads only LUC3D's BMimica cross-view mean.

## What I could not determine

- Whether 3D-MuPPET as *published* re-establishes its cross-view mapping after a
  tracklet death. I audited the port (`muppet_run.py`) and the emitted JSONs, not the
  upstream repository, so I cannot say whether the coverage collapse is the method's
  or the port's. Either way the plotted number is not an identity-accuracy measurement.
- The exact corrected BMimica SLEAP corpus mean. I computed its hard ceiling (0.2702
  session-mean) but did not score `retracked/`, which is item 1's job.
- Whether `figs/fig3_score.py`'s unscoped pooling is what produced the quoted 0.3996 /
  0.2237 — inferred from reading the scorer, not confirmed against that run's
  invocation.
