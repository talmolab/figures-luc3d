"""Is s-DANNCE's channel identity actually consistent across cameras? Audited
against HUMAN clicks, with no geometry and no LUC3D anywhere in the measurement.

Why this exists: `triads_adjudicate.py` compares LUC3D's grouping against the
channel grouping by reprojection residual, and that is CIRCULAR -- reprojection
error is LUC3D's own association objective, so it disagrees with the channel
precisely when it has found a lower-residual grouping. That test cannot establish
that the channel is wrong. This one can.

The measurement: `handLabeled2D` in the COM Label3D file is (animals, cameras, 2,
frames) of the human's ACTUAL clicks, NaN where the human did not click that view
(`data_2d` is the reprojection-filled version and is useless for this -- it
reproduces `data_3d` to 0.0000 px). Where a human clicked animal `a` in camera `c`,
we know which physical animal sits at that pixel. Match it to the nearest COM
network detection in the same camera and we recover, per camera, the map
channel -> animal that the network actually produced.

s-DANNCE's identity assumes that map is the SAME in every camera -- that is the
whole "identity by construction" mechanism. So: collect the map from every camera
in a frame that has a matched human click, and check whether any channel is claimed
by two different animals. Any such frame is one where s-DANNCE's per-animal 3D COM
is built from a mixture of two animals.

Limits, stated because they bound the claim:
  * a human clicked ~2 of 6 cameras per animal, so most frames contribute only a
    couple of (camera, channel, animal) observations and a conflict can only be
    seen where two cameras happen to constrain the same channel;
  * an unmatched or ambiguous click (nearest detection further than --max-dist) is
    dropped rather than guessed;
  * this measures the CHANNEL, not LUC3D. It says nothing about whether LUC3D got
    those frames right -- it only establishes whether the reference LUC3D is being
    scored against is itself reliable.
"""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io as sio


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--session-dir", type=Path, required=True)
    ap.add_argument("--labels", type=Path, required=True)
    ap.add_argument("--max-dist", type=float, default=40.0,
                    help="px; a human click further than this from every detection "
                         "is treated as unmatched rather than assigned")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()

    m = sio.loadmat(args.labels)
    hl = np.asarray(m["handLabeled2D"])              # (A, C, 2, F)
    names = [str(c[0]) for c in m["camnames"][0]]
    sids = np.asarray(m["labelData"][0, 0]["data_sampleID"][0, 0]).ravel()
    A, C, _, F = hl.shape

    d = pickle.load(open(args.session_dir / "COM" / "predict00" / "com3d.pickle", "rb"))
    # keys look like '<exp>_<frame>'; index by frame id
    by_frame = {}
    for k in d:
        by_frame.setdefault(int(k.split("_")[1]), k)

    rows = []
    n_obs = 0
    frames_checked = 0
    frames_conflict = 0
    matched_dists = []
    for fi in range(F):
        sid = int(sids[fi])
        key = by_frame.get(sid)
        if key is None:
            continue
        s = d[key]
        # (camera, channel) -> animal, as established by the human
        claim = {}
        for ci, cam in enumerate(names):
            com = np.asarray(s[cam]["COM"], dtype=float)     # (channels, 2)
            for a in range(A):
                click = hl[a, ci, :, fi]
                if not np.all(np.isfinite(click)):
                    continue
                dist = np.linalg.norm(com - click[None, :], axis=1)
                ch = int(np.nanargmin(dist))
                if not np.isfinite(dist[ch]) or dist[ch] > args.max_dist:
                    continue
                matched_dists.append(float(dist[ch]))
                claim[(ci, ch)] = a
                n_obs += 1

        # A channel claimed by two different animals in two cameras = the channel
        # identity is not the same animal in every view on this frame.
        per_channel = {}
        for (ci, ch), a in claim.items():
            per_channel.setdefault(ch, set()).add(a)
        checkable = {ch: v for ch, v in per_channel.items()
                     if sum(1 for (c2, c3) in claim if c3 == ch) >= 2}
        if not checkable:
            continue
        frames_checked += 1
        conflict = any(len(v) > 1 for v in checkable.values())
        frames_conflict += int(conflict)
        rows.append({"sample": sid, "channels_checkable": len(checkable),
                     "conflict": int(conflict)})

    md = np.asarray(matched_dists)
    print(f"human clicks matched to a detection: {n_obs} "
          f"(median match distance {np.median(md):.2f} px, p95 {np.percentile(md, 95):.2f})"
          if md.size else "no matches")
    print(f"frames where >=1 channel is constrained by >=2 cameras: {frames_checked}")
    print(f"  of those, frames where a channel maps to DIFFERENT animals: "
          f"{frames_conflict} ({100 * frames_conflict / max(1, frames_checked):.1f}%)")
    if args.out and rows:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(args.out, index=False)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
