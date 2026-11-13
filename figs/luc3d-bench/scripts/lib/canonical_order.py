"""Canonical per-frame detection ordering — the linchpin of the detections-only
benchmark.

Within each (session, camera, frame), the surviving detections are ordered
DETERMINISTICALLY so that every consumer (the widened H5 aggregator, the LUC3D
driver, Liezl's CrossViewTracker, the evaluator) references the *same* detection
at the *same* slot index. The ordering is:

    1. descending instance score
    2. ties broken by keypoint-centroid x (ascending)
    3. further ties broken by keypoint-centroid y (ascending)

This module is imported by both the `.venv`/`sleap_nn_env` aggregator path and
the `liezl_env` session builder, so it must depend ONLY on numpy + sleap_io
(both present in every relevant env). Keep it dependency-light.

Crucially, downstream consumers RE-SORT instances with `sort_instances` rather
than trusting the on-disk `.slp` instance order. That makes the slot mapping a
pure function of the detection geometry, independent of serialization order.
"""
from __future__ import annotations

import numpy as np


def instance_xy(inst) -> np.ndarray:
    """Return an (n_nodes, 2) float array of keypoint xy for a sleap_io instance.

    Missing/invisible keypoints are NaN. Works for both the structured
    PredictedPointsArray (`points["xy"]`) and the generic `inst.numpy()` path.
    """
    pts = getattr(inst, "points", None)
    if pts is not None and hasattr(pts, "dtype") and pts.dtype.names and "xy" in pts.dtype.names:
        xy = np.asarray(pts["xy"], dtype=np.float64)
        if "visible" in pts.dtype.names:
            vis = np.asarray(pts["visible"], dtype=bool)
            xy = xy.copy()
            xy[~vis] = np.nan
        return xy
    # Fallback: sleap_io Instance.numpy() → (n_nodes, 2) with NaN for missing.
    return np.asarray(inst.numpy(), dtype=np.float64)


def instance_centroid(inst) -> tuple[float, float]:
    """Mean of the visible keypoints (NaN-safe). Returns (+inf, +inf) if none."""
    xy = instance_xy(inst)
    if xy.size == 0 or not np.isfinite(xy).any():
        return (np.inf, np.inf)
    with np.errstate(invalid="ignore"):
        cx = float(np.nanmean(xy[:, 0]))
        cy = float(np.nanmean(xy[:, 1]))
    if not np.isfinite(cx):
        cx = np.inf
    if not np.isfinite(cy):
        cy = np.inf
    return (cx, cy)


def canonical_sort_key(inst):
    """Deterministic sort key for one detection.

    Ascending sort with this key yields: highest score first, ties broken by
    smallest centroid-x then smallest centroid-y. Detections with no score sort
    after scored ones; detections with no visible keypoints sort last.
    """
    score = getattr(inst, "score", None)
    score = float(score) if score is not None and np.isfinite(score) else -np.inf
    cx, cy = instance_centroid(inst)
    return (-score, cx, cy)


def sort_instances(instances) -> list:
    """Return the instances sorted into canonical per-frame slot order.

    `sorted` is stable, so any residual ties (identical score + centroid) keep
    their original relative order — deterministic given a fixed input order.
    """
    return sorted(list(instances), key=canonical_sort_key)
