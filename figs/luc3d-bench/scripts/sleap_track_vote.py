#!/usr/bin/env python3
"""Regularize Liezl's cross-view identity using sleap-nn's RELIABLE per-camera tracks.

Insight (Eric, 2026-06-22): geometric 3D trajectories are unreliable at close
encounters (every swap-corrector built on them failed). But sleap-nn's per-camera
2D tracks ARE reliable (the SLEAP single-view reference had few sustained switches).
The keeptrack pipeline feeds Liezl those tracked .slp's but STRIPS the track ids
(canonical slot). This post-process recovers them: for each (camera, sleap_track)
we vote every member detection to that track's MODAL Liezl identity. Where Liezl's
cross-view identity flickered/swapped while the per-camera SLEAP track stayed on one
animal, the vote fixes it. Unlike the geometric tracklet vote, the segmentation here
is the reliable SLEAP track, so the modal is trustworthy.

Input : <in-dir>/<session>.json (Liezl on keeptrack; keys "{cam}:{det_pos}")
        the tracked .slp's (sleap_nn_retracked) for the per-camera track ids
Output: <out-dir>/<session>.json (same schema, identities regularized)

Usage:
  PYTHONPATH=. .venv/bin/python scripts/sleap_track_vote.py \
      --in-dir   outputs/PAF_3d_kalman/keeptrack_liezl_results \
      --slp-dir  outputs/sleap_nn_retracked \
      --master   outputs/sleap_nn_master_sheet.tsv \
      --out-dir  outputs/PAF_3d_kalman/keeptrack_trackvote_results \
      [--min-frac 0.0] [--sessions S1 ...]
"""
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd
import sleap_io as sio

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "lib"))
from canonical_order import sort_instances  # noqa: E402

CAMS = ["back", "backL", "mid", "midL", "top", "topL"]


def slp_path(slp_dir, sid, cam):
    return Path(slp_dir) / sid / f"{cam}.slp"


def build_detpos_to_track(slp_file):
    """Map (frame_idx, det_pos) -> sleap track id for one camera .slp, using the
    SAME canonical ordering Liezl/the H5 aggregator used (so det_pos lines up)."""
    labels = sio.load_slp(str(slp_file))
    out = {}
    for lf in labels.labeled_frames:
        for det_pos, inst in enumerate(sort_instances(lf.instances)):
            tname = inst.track.name if inst.track is not None else None
            out[(lf.frame_idx, det_pos)] = tname
    return out


def process(in_json, slp_dir, sid, min_frac):
    d = json.load(open(in_json))
    # per cam: (frame, det_pos) -> sleap track
    detpos_track = {}
    for cam in CAMS:
        p = slp_path(slp_dir, sid, cam)
        if p.exists():
            detpos_track[cam] = build_detpos_to_track(p)

    # gather Liezl identity votes per (cam, sleap_track)
    votes = defaultdict(Counter)
    for fr in d["frames"]:
        f = fr["frame"]
        for k, ident in fr["assignments"]:
            cam, pos = k.split(":")
            tname = detpos_track.get(cam, {}).get((f, int(pos)))
            if tname is not None:
                votes[(cam, tname)][ident] += 1
    # modal identity per (cam, sleap_track), with optional majority threshold
    modal = {}
    for key, c in votes.items():
        ident, n = c.most_common(1)[0]
        if n / sum(c.values()) >= min_frac:
            modal[key] = ident

    n_changed = 0
    for fr in d["frames"]:
        f = fr["frame"]
        new = []
        for k, ident in fr["assignments"]:
            cam, pos = k.split(":")
            tname = detpos_track.get(cam, {}).get((f, int(pos)))
            nid = modal.get((cam, tname), ident) if tname is not None else ident
            if nid != ident:
                n_changed += 1
            new.append([k, int(nid)])
        fr["assignments"] = new
    return d, n_changed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", required=True)
    ap.add_argument("--slp-dir", required=True)
    ap.add_argument("--master", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--min-frac", type=float, default=0.0,
                    help="only apply vote if modal identity covers >= this fraction of the track")
    ap.add_argument("--sessions", nargs="*", default=None)
    a = ap.parse_args()
    out_dir = Path(a.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    in_dir = Path(a.in_dir)
    targets = a.sessions or [p.stem for p in sorted(in_dir.glob("*.json"))]
    tot = 0
    for sid in targets:
        ij = in_dir / f"{sid}.json"
        if not ij.exists():
            continue
        d, nch = process(ij, a.slp_dir, sid, a.min_frac)
        tot += nch
        json.dump(d, open(out_dir / f"{sid}.json", "w"))
        print(f"  {sid}: revoted {nch} assignments")
    print(f"DONE — {tot} assignments revoted -> {out_dir}")


if __name__ == "__main__":
    main()
