"""Build 2D pose training labels for the TRIADS videos by projecting s-DANNCE's own
3D predictions into each camera.

Why this step exists. The TRIADS release has no per-view 2D POSE -- only per-camera
COM centroids (one point per animal), because s-DANNCE goes image-volume straight to
3D and has no 2D pose stage. A single point gives LUC3D's association nothing to work
with and gives the sleap-nn filter stack nothing to filter (MIN_VISIBLE_NODES=7 alone
would reject every detection). So to run the real pipeline -- SLEAP tracking, the
sleap-nn filters, then LUC3D -- we first need a 2D pose detector on their footage.

The labels here are DISTILLED: `save_data_AVG0.mat` holds s-DANNCE's 23-node 3D pose
for each of 3 animals on all 60,000 frames, and projecting it through the verified
calibration gives 2D pose in every view for free. Two things to be clear about:

  * As TRAINING TARGETS these are fine. They are accurate positions -- the calibration
    round-trips the humans' own labels to 0.000 px -- and a detector trained on them
    learns to find rat body parts in images.
  * The resulting DETECTIONS are what we actually want, and they are NOT oracle: a
    trained network run on pixels produces its own errors, misses, duplicates and
    scores, independently in each camera. That is exactly the property the projected
    2D lacked and the whole reason for this detour.

Ceiling worth stating up front: a detector distilled from s-DANNCE cannot exceed
s-DANNCE, and it inherits its systematic errors. This measures LUC3D's association on
realistic multi-node detections of these animals -- it is not a claim about s-DANNCE's
own accuracy.

Occlusion handling: a projected point is marked not-visible when it falls outside the
image. Self-occlusion is NOT modelled -- s-DANNCE reports every node in 3D whether or
not it is visible in a given view -- so some labels sit on top of the animal's own
body. That is a real limitation of distillation and is the main reason to expect the
detector's node scores to be poorly calibrated on occluded parts.
"""
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import numpy as np
import scipy.io as sio

from sdannce_prepare import load_dannce_cam, project

# s-DANNCE rat23 skeleton node names, in the order the 3D array stores them.
RAT23 = ["snout", "earL", "earR", "spineF", "spineM", "spineL", "tailBase",
         "tailMid", "tailTip", "shoulderL", "elbowL", "wristL", "handL",
         "shoulderR", "elbowR", "wristR", "handR", "hipL", "kneeL", "footL",
         "hipR", "kneeR", "footR"]


def extract_frames(video: Path, indices: np.ndarray, out_dir: Path, size=None):
    """Decode the requested frames to PNG with ffmpeg's select filter."""
    out_dir.mkdir(parents=True, exist_ok=True)
    expr = "+".join(f"eq(n\\,{int(i)})" for i in indices)
    cmd = ["ffmpeg", "-v", "error", "-y", "-i", str(video),
           "-vf", f"select='{expr}'", "-vsync", "0",
           str(out_dir / "%06d.png")]
    subprocess.run(cmd, check=True)
    return sorted(out_dir.glob("*.png"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--n-frames", type=int, default=600,
                    help="labelled frames per camera")
    ap.add_argument("--stride", type=int, default=0,
                    help="0 = spread n-frames evenly over the session")
    ap.add_argument("--cameras", nargs="+", default=None)
    ap.add_argument("--extract", action="store_true", help="also decode the images")
    args = ap.parse_args()

    sess = args.session.resolve()
    out = args.out_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)

    cal = sorted((sess / "calibration").glob("hires_cam*_params.mat"),
                 key=lambda p: int("".join(c for c in p.stem if c.isdigit())))
    cams = {f"Camera{i+1}": load_dannce_cam(p) for i, p in enumerate(cal)}
    names = args.cameras or list(cams)

    m = sio.loadmat(sess / "SDANNCE" / "predict00" / "save_data_AVG0.mat") \
        if (sess / "SDANNCE" / "predict00" / "save_data_AVG0.mat").exists() else None
    if m is None:
        cand = list(sess.glob("**/save_data_AVG0.mat"))
        if not cand:
            raise SystemExit(f"no save_data_AVG0.mat under {sess}")
        m = sio.loadmat(cand[0])
    pred = np.asarray(m["pred"])                  # (T, A, 3, K)
    T, A, _, K = pred.shape
    if K != len(RAT23):
        print(f"  note: {K} nodes, RAT23 list has {len(RAT23)} -- naming n0..n{K-1}")
    nodes = RAT23 if K == len(RAT23) else [f"n{i}" for i in range(K)]

    idx = (np.arange(0, T, args.stride)[: args.n_frames] if args.stride
           else np.linspace(0, T - 1, args.n_frames).astype(int))
    print(f"{sess.name}: {T} frames, {A} animals, {K} nodes -> {len(idx)} labelled frames")

    W, H = 1920, 1200
    npz = {"frame_idx": idx, "nodes": np.array(nodes, dtype=object), "cameras": np.array(names, dtype=object)}
    for cam in names:
        pts = np.full((len(idx), A, K, 2), np.nan)
        for j, fi in enumerate(idx):
            for a in range(A):
                X = pred[fi, a].T                  # (K,3)
                if not np.all(np.isfinite(X)):
                    continue
                xy = project(X, cams[cam])
                # off-image points are not visible; self-occlusion is NOT modelled
                bad = (xy[:, 0] < 0) | (xy[:, 0] >= W) | (xy[:, 1] < 0) | (xy[:, 1] >= H)
                xy[bad] = np.nan
                pts[j, a] = xy
        npz[f"pts_{cam}"] = pts
        vis = np.isfinite(pts).all(axis=3)
        print(f"  {cam}: {100 * vis.mean():.1f}% of nodes in frame, "
              f"{100 * vis.all(axis=2).mean():.1f}% of instances fully visible")
        if args.extract:
            vid = sess / "videos" / cam / "0.mp4"
            d = out / "frames" / cam
            if not d.exists() or len(list(d.glob('*.png'))) < len(idx):
                print(f"    decoding {len(idx)} frames from {vid.name} ...", flush=True)
                extract_frames(vid, idx, d)
    np.savez_compressed(out / f"{sess.name}_pose_labels.npz", **npz)
    print(f"wrote {out / (sess.name + '_pose_labels.npz')}")


if __name__ == "__main__":
    main()
