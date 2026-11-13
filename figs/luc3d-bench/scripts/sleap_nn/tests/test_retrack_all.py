"""Smoke test for the multi-session driver — uses 2 fake rows."""
from __future__ import annotations
import pandas as pd
from pathlib import Path
import scripts.sleap_nn.retrack_one as ro
from scripts.sleap_nn.retrack_all import retrack_all_rows


def test_driver_dispatches_per_row(tmp_path, monkeypatch):
    # Force Branch B so the test doesn't need a real video file.
    monkeypatch.setattr(ro, "FILTER_ONLY_WORKS", False)

    fake_master = tmp_path / "master.tsv"
    fake_src = Path(__file__).parent / "fixtures" / "tiny.slp"
    pd.DataFrame([
        {"session": "S1", "session_path": "s1", "num_animals": 2,
         "back_raw_pred_slp": str(fake_src)},
        {"session": "S2", "session_path": "s2", "num_animals": 2,
         "back_raw_pred_slp": str(fake_src)},
    ]).to_csv(fake_master, sep="\t", index=False)

    out_dir = tmp_path / "out"
    info = retrack_all_rows(
        master=fake_master, out_dir=out_dir,
        cams=["back"], target_col="num_animals",
        min_visible_nodes=3,
    )
    assert info["n_processed"] == 2
    assert (out_dir / "S1" / "back.slp").exists()
    assert (out_dir / "S2" / "back.slp").exists()
