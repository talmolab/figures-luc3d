"""Turn the projected 2D pose labels + extracted frames into a SLEAP `.slp` training
file, so sleap-nn can train a real 2D pose detector on the TRIADS footage.

One Labels file covering all six cameras rather than one per camera: the views are
very different, but a rig is deployed as a unit and a single model that handles every
view is both what a user would train and more data for the same labelling effort.

Instances carry all 23 nodes. A node whose projection fell outside the image was
already NaN'd upstream and is written as a missing point. Self-occlusion is NOT
modelled (see triads_make_pose_labels.py) -- some labels sit on the far side of the
animal -- so expect node scores on hidden parts to be poorly calibrated. That matters
later, because the sleap-nn filter stack thresholds exactly those scores.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels-npz", type=Path, required=True)
    ap.add_argument("--frames-root", type=Path, required=True,
                    help="directory holding <Camera>/%%06d.png")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--min-nodes", type=int, default=6,
                    help="drop an instance with fewer visible nodes than this")
    args = ap.parse_args()

    import sleap_io as sio

    d = np.load(args.labels_npz, allow_pickle=True)
    nodes = [str(n) for n in d["nodes"]]
    cameras = [str(c) for c in d["cameras"]]
    frame_idx = d["frame_idx"]

    # A minimal chain skeleton: sleap-nn needs edges for bottom-up PAFs, and the
    # rat23 connectivity is not shipped with the predictions, so use an anatomically
    # sensible spine + limbs rather than inventing a full graph.
    E = [("snout", "earL"), ("snout", "earR"), ("earL", "spineF"), ("earR", "spineF"),
         ("spineF", "spineM"), ("spineM", "spineL"), ("spineL", "tailBase"),
         ("tailBase", "tailMid"), ("tailMid", "tailTip"),
         ("spineF", "shoulderL"), ("shoulderL", "elbowL"), ("elbowL", "wristL"),
         ("wristL", "handL"),
         ("spineF", "shoulderR"), ("shoulderR", "elbowR"), ("elbowR", "wristR"),
         ("wristR", "handR"),
         ("spineL", "hipL"), ("hipL", "kneeL"), ("kneeL", "footL"),
         ("spineL", "hipR"), ("hipR", "kneeR"), ("kneeR", "footR")]
    edges = [(a, b) for a, b in E if a in nodes and b in nodes]
    skel = sio.Skeleton(nodes=nodes, edges=edges)

    lfs = []
    videos = []
    for cam in cameras:
        cam_dir = args.frames_root / cam
        imgs = sorted(cam_dir.glob("*.png"))
        if not imgs:
            print(f"  {cam}: no frames, skipping")
            continue
        if len(imgs) != len(frame_idx):
            print(f"  {cam}: {len(imgs)} images vs {len(frame_idx)} labels — "
                  f"using the first {min(len(imgs), len(frame_idx))}")
        n = min(len(imgs), len(frame_idx))
        video = sio.Video.from_filename([str(p) for p in imgs[:n]])
        videos.append(video)
        pts = d[f"pts_{cam}"]                       # (F, A, K, 2)
        kept = dropped = 0
        for j in range(n):
            insts = []
            for a in range(pts.shape[1]):
                xy = pts[j, a]
                vis = np.isfinite(xy).all(axis=1)
                if vis.sum() < args.min_nodes:
                    dropped += 1
                    continue
                insts.append(sio.Instance.from_numpy(xy.astype(np.float32), skeleton=skel))
                kept += 1
            if insts:
                lfs.append(sio.LabeledFrame(video=video, frame_idx=j, instances=insts))
        print(f"  {cam}: {n} frames, {kept} instances kept, {dropped} dropped")

    labels = sio.Labels(labeled_frames=lfs, videos=videos, skeletons=[skel])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    labels.save(str(args.out))
    print(f"wrote {args.out}: {len(lfs)} labeled frames, {len(videos)} videos, "
          f"{len(nodes)} nodes, {len(edges)} edges")


if __name__ == "__main__":
    main()
