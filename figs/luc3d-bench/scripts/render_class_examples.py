#!/usr/bin/env python3
"""Render one demo video PER ERROR CLASS, each containing MULTIPLE examples.

The whole point is to SEE THE SWAP HAPPEN: every clip opens on a stretch where the
focal animal is correctly tracked (GREEN boxes), then transitions into the swap (RED).
Many raw episodes in error_episodes.tsv are *continuations* of an earlier swap (a
maximal wrong-run that got split by a dropout gap), so their `start` is mid-swap. We
therefore ignore the raw `start` and instead walk each timeline back to find the true
clean->wrong TRANSITION, and anchor the clip on that.

For every class we keep the longest-N episodes that (a) have such a clean->wrong
transition within reach, (b) recover, and (c) come from a variety of sessions. Each
example is concatenated into one video per class, separated by a title card.

Companion to render_swap_grid.py (single clip).

Usage:
  PYTHONPATH=. .venv/bin/python scripts/render_class_examples.py \
      --results-dir outputs/PAF_3d_kalman/_liezlsweep/G_keeptrack_3d6 \
      --pred-h5-dir outputs/keeptrack_h5s \
      --out-dir outputs/PAF_3d_kalman/_diag/grids \
      [--classes CROWD,CLOSE_CROSSING] [--n 4] [--lead 60] [--show 400]
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.render_swap_grid import build_session, open_writer, render_clip, title_card

REPO = Path(__file__).resolve().parents[1]
EPISODES = REPO / "outputs/PAF_3d_kalman/_diag/error_episodes.tsv"
TL = REPO / "outputs/PAF_3d_kalman/_diag/timelines"

# one-line definition shown on each class's title cards
CLASS_DEF = {
    "CLOSE_CROSSING": "two animals detected & physically close in 3D through the swap",
    "CROSSVIEW_LOCK": "id flips in >=3 cameras at once and stays locked (worst/longest)",
    "CROWD": ">=3 animals: a multi-animal pileup drives the mislabel",
    "DROPOUT_REENTRY": "focal animal vanishes (occlusion), then re-enters with the wrong id",
}

_TL_CACHE = {}


def load_timeline(session):
    if session not in _TL_CACHE:
        _TL_CACHE[session] = np.load(TL / f"{session}.npz", allow_pickle=True)
    return _TL_CACHE[session]


def clean_mask(tl, track):
    """Per-frame: True where the focal is detected in a majority of cams AND wears the correct id."""
    matched = tl["matched"]; correct = tl["correct"]
    ncams = matched.shape[0]
    mt = matched[:, :, track]
    det = mt >= 0
    wrong = det & (mt != correct[track])
    ncam_det = det.sum(0)
    ncam_wrong = wrong.sum(0)
    wf = np.where(ncam_det > 0, ncam_wrong / np.maximum(ncam_det, 1), 1.0)
    return (ncam_det >= ncams / 2.0) & (wf < 0.5)


def clean_transition(tl, track, start, min_run=20, search_back=700):
    """Last frame `g` ending a clean run of >= min_run frames before `start` (the swap).

    This is where good tracking ends. For CLOSE_CROSSING/CROSSVIEW_LOCK it sits right at
    the swap; for DROPOUT_REENTRY it's where the animal drops out (the actual wrong-id
    reentry comes later, at `start`). Returns None if nothing clean within search_back.
    """
    clean = clean_mask(tl, track)
    floor = max(0, start - search_back)
    f = start - 1
    while f >= floor:
        if clean[f]:
            r = 0; k = f
            while k >= 0 and clean[k] and r < min_run:
                r += 1; k -= 1
            if r >= min_run:
                return f
        f -= 1
    return None


def pick_episodes(df, cls, n, lead, show, tail, max_per_session):
    """Longest-N recovered episodes for a class that open on clean tracking, varied & deduped.

    Each clip spans [g-lead, wrong_onset+show]: it opens on correct tracking (g = end of the
    last clean run), runs through the transition (and any dropout), and shows the wrong id
    for `show` frames after it actually begins (`start`). Near-duplicate events (same session
    within 150f) are dropped so the examples are distinct.
    """
    g = df[(df.cls == cls) & (df.recovered == 1)].copy().sort_values("duration", ascending=False)
    chosen = []
    per_sess = {}
    picked_events = []  # (session, frame) of already-chosen transitions
    for _, r in g.iterrows():
        if r.correct_id < 0:
            continue
        if per_sess.get(r.session, 0) >= max_per_session:
            continue
        gframe = clean_transition(load_timeline(r.session), int(r.gt_track), int(r.start))
        if gframe is None:
            continue
        if any(s == r.session and abs(f - gframe) < 150 for s, f in picked_events):
            continue  # same event, different focal track — skip the duplicate
        clip_start = max(0, gframe - lead)
        clip_end = min(int(r.end) + tail, int(r.start) + show)
        chosen.append((r, clip_start, clip_end, gframe))
        per_sess[r.session] = per_sess.get(r.session, 0) + 1
        picked_events.append((r.session, gframe))
        if len(chosen) >= n:
            break
    return chosen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", required=True)
    ap.add_argument("--pred-h5-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--classes", default="CLOSE_CROSSING,CROSSVIEW_LOCK,CROWD,DROPOUT_REENTRY")
    ap.add_argument("--n", type=int, default=4, help="examples per class")
    ap.add_argument("--lead", type=int, default=60, help="frames shown before the clean->wrong transition")
    ap.add_argument("--show", type=int, default=400, help="max frames of the swap shown after onset")
    ap.add_argument("--tail", type=int, default=45, help="extra frames after the episode ends (recovery)")
    ap.add_argument("--max-per-session", type=int, default=2)
    ap.add_argument("--fps", type=int, default=30)
    a = ap.parse_args()

    df = pd.read_csv(EPISODES, sep="\t"); df["session"] = df["session"].astype(str)
    out_dir = Path(a.out_dir)

    for cls in a.classes.split(","):
        cls = cls.strip()
        eps = pick_episodes(df, cls, a.n, a.lead, a.show, a.tail, a.max_per_session)
        if not eps:
            print(f"[{cls}] no episodes with a clean->wrong transition — skipping"); continue
        out = str(out_dir / f"class_{cls}.mp4")
        print(f"\n=== {cls}: {len(eps)} examples -> {out} ===")
        vw = None; cell_h = None
        sess_cache = {}
        for i, (ep, clip_start, clip_end, gframe) in enumerate(eps, 1):
            sid = ep.session
            if sid not in sess_cache:
                for s in sess_cache.values():
                    for c in s["caps"].values():
                        c.release()
                sess_cache = {sid: build_session(sid, a.results_dir, a.pred_h5_dir)}
            sess = sess_cache[sid]
            if vw is None:
                cell_h = sess["cell_h"]
                vw = open_writer(out, a.fps, cell_h)
            print(f"  [{i}/{len(eps)}] {sid} gt{ep.gt_track} clean-opens f{gframe} "
                  f"swap-onset f{ep.start} clip f{clip_start}-{clip_end} (dur={ep.duration}) "
                  f"inter={ep.onset_inter_mm:.0f}mm dropout={ep.dropout_pre:.2f} animals={ep.animals}")
            title_card(vw, cell_h, [
                (f"{cls}  -  example {i}/{len(eps)}", 1.0, (255, 255, 255)),
                (CLASS_DEF.get(cls, ""), 0.55, (180, 220, 180)),
                (f"session {sid}   focal gt{ep.gt_track}   swap onset f{ep.start}", 0.6, (200, 200, 255)),
                ("watch the GREEN (correct) box turn RED (swapped)", 0.55, (160, 200, 255)),
            ])
            render_clip(vw, sess, clip_start, clip_end, focal=int(ep.gt_track),
                        swap_start=int(ep.start), swap_end=int(ep.end),
                        label=f"{cls} ({i}/{len(eps)})")
        if vw is not None:
            vw.release()
        for s in sess_cache.values():
            for c in s["caps"].values():
                c.release()
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
