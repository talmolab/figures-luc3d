"""Tests for retrack_one — filter+retrack a tiny .slp."""
from __future__ import annotations
import sleap_io as sio
from pathlib import Path
import scripts.sleap_nn.retrack_one as ro
from scripts.sleap_nn.retrack_one import retrack_one

FIX = Path(__file__).parent / "fixtures" / "tiny.slp"


def test_retrack_one_writes_output(tmp_path, monkeypatch):
    # Force Branch B (library adapter) so the test doesn't need a real video.
    monkeypatch.setattr(ro, "FILTER_ONLY_WORKS", False)
    out = tmp_path / "tiny_retracked.slp"
    info = retrack_one(
        src=FIX, dst=out, target_instance_count=2,
        min_instance_score=0.85, min_mean_node_score=0.80,
        min_visible_nodes=3,  # tiny skeleton has 3 nodes
        overlapping_threshold=0.50, window_size=15,
    )
    assert out.exists()
    labels = sio.load_slp(out)
    assert len(labels.labeled_frames) == 5
    assert info["n_input_instances"] == 10
    assert info["n_output_instances"] >= 1
    # Every kept instance must have been assigned a Track.
    for lf in labels.labeled_frames:
        for inst in lf.instances:
            assert inst.track is not None, "tracking did not assign a Track"
