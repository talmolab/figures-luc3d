"""Aggregator produces H5s matching the existing schema."""
from __future__ import annotations
import h5py
import numpy as np
from pathlib import Path
from scripts.sleap_nn.aggregate_to_h5 import aggregate_cam


def test_aggregate_one_cam_one_session(tmp_path):
    # Reuse the tiny.slp fixture from earlier tasks.
    src_root = tmp_path / "retracked"
    (src_root / "S1").mkdir(parents=True)
    import shutil
    shutil.copy(Path(__file__).parent / "fixtures" / "tiny.slp",
                src_root / "S1" / "back.slp")
    out = tmp_path / "back_predictions.h5"
    aggregate_cam(retrack_dir=src_root, cam="back", out_h5=out,
                  session_ids=["S1"], n_animals=2, n_nodes=3, n_frames=5)
    with h5py.File(out) as f:
        tracks = f["tracks"][...]
        assert tracks.shape == (1, 5, 2, 3, 2), tracks.shape
