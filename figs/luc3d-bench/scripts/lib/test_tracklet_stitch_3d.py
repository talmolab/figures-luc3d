"""Tests for the identity-consistent 3D stitch."""
from __future__ import annotations
import numpy as np
import pytest
from scripts.lib.tracklet_stitch_3d import (
    stitch_tracklets,
    Tracklet,
)


def _t(start, end, identity, last_pos, last_vel):
    return Tracklet(
        start_frame=start, end_frame=end,
        identity=identity,
        last_pos3d=np.asarray(last_pos, dtype=float),
        last_vel3d=np.asarray(last_vel, dtype=float),
        first_pos3d=np.asarray(last_pos, dtype=float),
    )


def test_stitch_close_gap_same_predicted_position():
    # Tracklet A ends at frame 100, moving +10 mm/frame in x.
    a = _t(0, 100, identity=1, last_pos=[100, 0, 0], last_vel=[10, 0, 0])
    # Tracklet B starts at frame 105; KF predicts [150, 0, 0] which is exactly
    # B's first_pos3d => stitch.
    b = _t(105, 200, identity=2, last_pos=[300, 0, 0], last_vel=[10, 0, 0])
    b.first_pos3d = np.array([150.0, 0, 0])
    out = stitch_tracklets([a, b], max_gap=60, sigma=50.0, time_decay=0.05,
                           threshold=0.5)
    assert len(out) == 1, "expected stitched into one"
    assert out[0].identity == 1


def test_no_stitch_outside_sigma():
    a = _t(0, 100, identity=1, last_pos=[100, 0, 0], last_vel=[10, 0, 0])
    b = _t(105, 200, identity=2, last_pos=[300, 0, 0], last_vel=[10, 0, 0])
    b.first_pos3d = np.array([1000.0, 0, 0])  # far from KF prediction
    out = stitch_tracklets([a, b], max_gap=60, sigma=50.0, time_decay=0.05,
                           threshold=0.5)
    assert len(out) == 2


def test_no_stitch_when_gap_exceeds_max():
    a = _t(0, 100, identity=1, last_pos=[100, 0, 0], last_vel=[10, 0, 0])
    b = _t(500, 600, identity=2, last_pos=[300, 0, 0], last_vel=[10, 0, 0])
    b.first_pos3d = np.array([5000.0, 0, 0])  # would match KF
    out = stitch_tracklets([a, b], max_gap=60, sigma=50.0, time_decay=0.05,
                           threshold=0.5)
    assert len(out) == 2
