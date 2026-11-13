"""Apply sleap-nn filter + tracking to a single .slp file.

Branch A (preferred): shell out to `sleap-nn cli track --data_path <slp>`
                       (no --model_paths — relies on existing predictions).
Branch B (fallback):  use sleap_nn library directly to filter+track an
                       sio.Labels in memory.

Returns dict with input/output instance counts so callers can log progress.
"""
from __future__ import annotations
import numpy as np
import subprocess
import sys
from pathlib import Path
from typing import Optional
import sleap_io as sio

REPO = Path(__file__).resolve().parents[2]
SLEAP_NN_PY = REPO / "sleap_nn_env" / "bin" / "python"

# Set FILTER_ONLY_WORKS from the probe outcome (Task 2). Update after probe.
FILTER_ONLY_WORKS: Optional[bool] = None  # None → auto-detect on first call


def _detect_filter_only_works() -> bool:
    """Cache result after first try. Read from PROBE_RESULT.txt."""
    global FILTER_ONLY_WORKS
    if FILTER_ONLY_WORKS is not None:
        return FILTER_ONLY_WORKS
    probe = REPO / "scripts" / "sleap_nn" / "PROBE_RESULT.txt"
    if probe.exists():
        last = probe.read_text().strip().splitlines()[-1]
        FILTER_ONLY_WORKS = "works" in last and "does NOT" not in last
    else:
        FILTER_ONLY_WORKS = False  # safe default — use adapter
    return FILTER_ONLY_WORKS


def retrack_one(
    src: Path, dst: Path, target_instance_count: int,
    # Defaults calibrated for SLAP-2M (15-node skeleton). Distribution analysis
    # (2026-05-20, n=2000 instances on session 10072022120554/back):
    #   instance_score:   p10=0.92, p50=0.94, p90=0.97  → 0.85 retains 99.5%
    #   mean_node_score:  p10=0.55, p50=0.67, p90=0.73  → 0.80 retains 0%
    #   visible_nodes:    always 15 (any threshold up to 15 passes)
    # Stefan's recipe (0.85 / 0.80 / 12) was tuned on a different skeleton and
    # over-culls SLAP-2M by ~99%. Loosen mean_node_score to 0.55 (just below
    # median) so the filter trims the lowest-quality tail without nuking real
    # detections; loosen visible_nodes to 8 for consistency with the probe.
    min_instance_score: float = 0.85,
    min_mean_node_score: float = 0.55,
    min_visible_nodes: int = 8,
    overlapping_threshold: float = 0.50,
    window_size: int = 15,
) -> dict:
    src, dst = Path(src), Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    labels_in = sio.load_slp(src)
    n_in = sum(len(lf.instances) for lf in labels_in.labeled_frames)

    if _detect_filter_only_works():
        cmd = [
            str(SLEAP_NN_PY), "-m", "sleap_nn.cli", "track",
            "--data_path", str(src),
            "--filter_min_instance_score", str(min_instance_score),
            "--filter_min_mean_node_score", str(min_mean_node_score),
            "--filter_min_visible_nodes", str(min_visible_nodes),
            "--filter_overlapping",
            "--filter_overlapping_method", "oks",
            "--filter_overlapping_threshold", str(overlapping_threshold),
            "--tracking",
            "--tracking_window_size", str(window_size),
            "--tracking_target_instance_count", str(target_instance_count),
            "--post_connect_single_breaks",
            "--output_path", str(dst),
        ]
        # Some sessions (especially long videos with many overlapping
        # detections) take 15-25 min to retrack. 1800s = 30 min cap.
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
        if res.returncode != 0:
            raise RuntimeError(f"sleap-nn cli track failed:\n{res.stderr[-2000:]}")
    else:
        _retrack_via_library(
            labels_in, dst, target_instance_count,
            min_instance_score, min_mean_node_score, min_visible_nodes,
            overlapping_threshold, window_size,
        )
    labels_out = sio.load_slp(dst)
    n_out = sum(len(lf.instances) for lf in labels_out.labeled_frames)
    return {"src": str(src), "dst": str(dst),
            "n_input_instances": n_in, "n_output_instances": n_out}


def _retrack_via_library(labels_in, dst, target_instance_count,
                         min_instance_score, min_mean_node_score,
                         min_visible_nodes, overlapping_threshold,
                         window_size):
    """Fallback adapter: replicate sleap-nn's filter+track in Python.

    The exact module paths may shift between sleap-nn point releases — adjust
    imports if pytest fails with ImportError.

    Tracker.track() takes (instances, frame_idx, image=None) — image is optional,
    so this path works without a real video backend.
    """
    from sleap_nn.inference import postprocessing as pp
    from sleap_nn.tracking.tracker import Tracker

    tracker = Tracker(tracking_target_instance_count=target_instance_count)

    for lf in labels_in.labeled_frames:
        # Apply score / visibility filters.
        kept = []
        for inst in lf.instances:
            if inst.score is not None and inst.score < min_instance_score:
                continue
            # PredictedPointsArray is a structured numpy array with fields:
            # 'xy' (n,2), 'score' (n,), 'visible' (n,), 'name' (n,)
            pts = inst.points
            if hasattr(pts, "dtype") and "score" in pts.dtype.names:
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
        # OKS overlap pruning — filter_overlapping_instances takes a Labels object.
        # Wrap the kept instances into a temporary single-frame Labels, filter,
        # then unwrap.
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
        kept = kept[:target_instance_count]
        # Track — image=None is fine for keypoint-based OKS scoring.
        tracked = tracker.track(kept, frame_idx=lf.frame_idx, image=None)
        lf.instances = tracked

    # Collect all Track objects created by the tracker and register them in Labels.
    all_tracks = []
    seen_track_names = set()
    for lf in labels_in.labeled_frames:
        for inst in lf.instances:
            if inst.track is not None and inst.track.name not in seen_track_names:
                seen_track_names.add(inst.track.name)
                all_tracks.append(inst.track)
    labels_in.tracks = all_tracks
    labels_in.save(dst)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--target-instance-count", type=int, required=True)
    ap.add_argument("--min-instance-score", type=float, default=0.85)
    ap.add_argument("--min-mean-node-score", type=float, default=0.55)
    ap.add_argument("--min-visible-nodes", type=int, default=8)
    ap.add_argument("--overlapping-threshold", type=float, default=0.50)
    ap.add_argument("--window-size", type=int, default=15)
    a = ap.parse_args()
    info = retrack_one(
        Path(a.src), Path(a.dst), a.target_instance_count,
        a.min_instance_score, a.min_mean_node_score, a.min_visible_nodes,
        a.overlapping_threshold, a.window_size,
    )
    print(info)
