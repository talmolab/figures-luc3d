"""Filter-only detection export — NO tracking, NO per-frame cap, NO top-N drop.

This is the linchpin of the detections-only ID-tracking benchmark
(docs/superpowers/plans/2026-06-21-detections-only-bench-plan.md, Task 1).

We reuse the exact quality/overlap filters from `retrack_one.py`
(`_retrack_via_library`) but:
  - DO NOT run `tracker.track(...)`  → no per-camera track continuity is baked in.
  - DO NOT apply `kept[:target_instance_count]` → keep EVERY surviving detection.
  - Order the surviving detections deterministically (canonical_order.sort_instances)
    so the on-disk per-frame instance order matches the canonical slot order.
  - Strip track identity (inst.track = None) so nothing downstream re-uses
    sleap-nn's track names as a continuity signal.

The output `.slp` therefore contains the full filtered per-frame detection pool,
which is fed identically to every ID tracker.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import sleap_io as sio

REPO = Path(__file__).resolve().parents[2]
# Add scripts/lib directly (NOT scripts/) so we don't shadow the installed
# `sleap_nn` package with the local `scripts/sleap_nn/` directory.
sys.path.insert(0, str(REPO / "scripts" / "lib"))
from canonical_order import sort_instances  # noqa: E402


def filter_one(
    src: Path,
    dst: Path,
    # Same SLAP-2M-calibrated thresholds as retrack_one.retrack_one.
    min_instance_score: float = 0.85,
    min_mean_node_score: float = 0.55,
    min_visible_nodes: int = 8,
    overlapping_threshold: float = 0.50,
) -> dict:
    """Filter one camera's raw predictions; keep all surviving detections.

    Returns dict with input/output counts and the per-file max detections/frame
    (feeds the dataset-wide n_dets computation in aggregate_detections_h5.py).
    """
    from sleap_nn.inference import postprocessing as pp

    src, dst = Path(src), Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)

    labels_in = sio.load_slp(str(src))
    n_in = sum(len(lf.instances) for lf in labels_in.labeled_frames)

    n_kept = 0
    max_dets_per_frame = 0
    for lf in labels_in.labeled_frames:
        kept = []
        for inst in lf.instances:
            # 1. instance score filter
            if inst.score is not None and inst.score < min_instance_score:
                continue
            # 2./3. mean-node-score + visible-node filters
            pts = inst.points
            if hasattr(pts, "dtype") and pts.dtype.names and "score" in pts.dtype.names:
                node_scores = pts["score"]
                mean_score = float(np.nanmean(node_scores)) if len(node_scores) > 0 else 0.0
                nvis = int(np.sum(pts["visible"]))
            else:
                mean_score = 0.0
                nvis = 0
            if mean_score < min_mean_node_score:
                continue
            if nvis < min_visible_nodes:
                continue
            kept.append(inst)

        # OKS-overlap prune (same as retrack_one.py) — dedupes double-detections.
        if len(kept) > 1:
            tmp_lf = sio.LabeledFrame(
                video=lf.video, frame_idx=lf.frame_idx, instances=kept
            )
            tmp_labels = sio.Labels(
                labeled_frames=[tmp_lf],
                skeletons=labels_in.skeletons,
                videos=labels_in.videos,
            )
            tmp_labels = pp.filter_overlapping_instances(
                tmp_labels, method="oks", threshold=overlapping_threshold,
            )
            kept = list(tmp_labels.labeled_frames[0].instances)

        # NO cap: keep EVERY surviving detection (this is the whole point).
        # Canonical per-frame ordering → this is the slot order on disk.
        kept = sort_instances(kept)
        # Strip track identity so no sleap-nn continuity leaks downstream.
        for inst in kept:
            inst.track = None
        lf.instances = kept

        n_kept += len(kept)
        if len(kept) > max_dets_per_frame:
            max_dets_per_frame = len(kept)

    # No tracks at all in the detections-only export.
    labels_in.tracks = []
    labels_in.save(str(dst))

    return {
        "src": str(src),
        "dst": str(dst),
        "n_input": int(n_in),
        "n_kept": int(n_kept),
        "max_dets_per_frame": int(max_dets_per_frame),
    }


if __name__ == "__main__":
    import argparse
    import json

    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--min-instance-score", type=float, default=0.85)
    ap.add_argument("--min-mean-node-score", type=float, default=0.55)
    ap.add_argument("--min-visible-nodes", type=int, default=8)
    ap.add_argument("--overlapping-threshold", type=float, default=0.50)
    a = ap.parse_args()
    info = filter_one(
        Path(a.src), Path(a.dst),
        a.min_instance_score, a.min_mean_node_score,
        a.min_visible_nodes, a.overlapping_threshold,
    )
    print(json.dumps(info))
