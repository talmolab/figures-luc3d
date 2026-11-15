"""Crop mouse-body detection patches from slap2m video for appearance ReID.

The detections-only pool H5 (`outputs/detections_only_h5s/<cam>_predictions.h5`,
dataset `tracks` shape (n_sessions, n_frames, n_dets=5, n_nodes=15, xy=2)) is ALREADY
in canonical slot order (see scripts/lib/canonical_order.py), so the slot index here is
exactly the `detSlot` referenced by `<cam>:<slot>` assignment keys and by the evaluator.

Mouse-specific cropping (agreed with Eric):
  * NO torso/"t-shirt" crop — meaningless for mice.
  * DROP the tail nodes so the box is the body (coat-color signal), not a thin tail.

The crop here only feeds the embedding network; the benchmark's scoring bbox is computed
independently inside evaluate.py, so the body-only crop does not affect eval correctness.
"""
from __future__ import annotations

import glob
import os
from pathlib import Path

import cv2
import h5py
import numpy as np

# Mouse skeleton node order (15):
# 0 Nose 1 Ear_R 2 Ear_L 3 TTI 4 TailTip 5 Head 6 Trunk 7 Tail_0 8 Tail_1 9 Tail_2
# 10 Shoulder_left 11 Shoulder_right 12 Haunch_left 13 Haunch_right 14 Neck
TAIL_NODES = (4, 7, 8, 9)          # TailTip, Tail_0, Tail_1, Tail_2 — always dropped
TTI_NODE = 3                       # tail-base / hip junction — optional (INCLUDE_TTI)
BODY_NODES = (0, 1, 2, 5, 6, 10, 11, 12, 13, 14)

SLAP2M_ROOT = "/root/talmolab-smb/eric/slap_2m"
BENCH_CAMERAS = ("back", "backL", "mid", "midL", "top", "topL")


def body_node_indices(include_tti: bool = True) -> list[int]:
    nodes = list(BODY_NODES)
    if include_tti:
        nodes.append(TTI_NODE)
    return sorted(nodes)


def resolve_video_path(rel_path: str, root: str = SLAP2M_ROOT) -> Path:
    """Resolve a master-sheet `{cam}_video` relative path to an on-disk file.

    Falls back to globbing the containing dir for any *.mp4 if the exact path (which
    encodes a CRF level that can drift) is missing.
    """
    p = Path(root) / rel_path
    if p.exists():
        return p
    cands = sorted(glob.glob(os.path.join(os.path.dirname(str(p)), "*.mp4")))
    if not cands:
        raise FileNotFoundError(f"no video for {rel_path} (looked in {os.path.dirname(str(p))})")
    return Path(cands[0])


def bbox_from_body(kpts: np.ndarray, node_idx: list[int], pad: float, min_valid: int):
    """Axis-aligned bbox from the body keypoints of one detection, or None."""
    pts = kpts[node_idx]                       # (n_body, 2)
    valid = np.isfinite(pts[:, 0]) & np.isfinite(pts[:, 1])
    if int(valid.sum()) < min_valid:
        return None
    xs, ys = pts[valid, 0], pts[valid, 1]
    return xs.min() - pad, ys.min() - pad, xs.max() + pad, ys.max() + pad


def iter_crops(
    session_idx: int,
    cam: str,
    h5_path: Path,
    video_path: Path,
    *,
    frame_filter=None,           # optional set/iterable of frame indices to keep
    include_tti: bool = True,
    pad: float = 8.0,
    size=(256, 128),             # (H, W) — standard ReID input
    min_valid: int = 3,
    min_box: int = 12,
):
    """Yield (frame_idx, slot, crop_rgb) for every valid detection.

    Reads the video sequentially (one decode per frame, all slots cropped) for speed.
    `crop_rgb` is an (H, W, 3) uint8 RGB array resized to `size`.
    """
    node_idx = body_node_indices(include_tti)
    keep = set(frame_filter) if frame_filter is not None else None

    with h5py.File(h5_path, "r") as f:
        tracks = f["tracks"][session_idx]      # (n_frames, n_dets, n_nodes, 2)
    n_frames, n_dets = tracks.shape[0], tracks.shape[1]

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video {video_path}")
    n_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    n = min(n_frames, n_video) if n_video > 0 else n_frames

    # Always decode sequentially — per-frame random seeks on h265 are pathologically
    # slow (re-seek to keyframe + decode-forward each time). When only a subset of
    # frames is wanted, stop early after the highest needed frame so we never decode
    # the whole video unnecessarily.
    stop = (max(keep) + 1) if keep else n
    stop = min(stop, n)
    try:
        fi = 0
        while fi < stop:
            ok, frame = cap.read()
            if not ok:
                break
            if keep is None or fi in keep:
                yield from _crops_in_frame(frame, tracks[fi], fi, n_dets, node_idx,
                                           pad, min_valid, min_box, size)
            fi += 1
    finally:
        cap.release()


def _crops_in_frame(frame, frame_tracks, fi, n_dets, node_idx, pad, min_valid, min_box, size):
    h, w = frame.shape[:2]
    for slot in range(n_dets):
        kpts = frame_tracks[slot]              # (n_nodes, 2)
        if not np.isfinite(kpts).any():
            continue
        box = bbox_from_body(kpts, node_idx, pad, min_valid)
        if box is None:
            continue
        x0, y0, x1, y1 = box
        x0 = max(0, int(x0)); y0 = max(0, int(y0))
        x1 = min(w, int(x1)); y1 = min(h, int(y1))
        if (x1 - x0) < min_box or (y1 - y0) < min_box:
            continue
        crop = frame[y0:y1, x0:x1]
        crop = cv2.resize(crop, (size[1], size[0]))            # cv2 wants (W, H)
        yield fi, slot, cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
