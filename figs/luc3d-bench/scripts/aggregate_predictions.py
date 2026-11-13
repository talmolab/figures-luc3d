"""Aggregate per-session, per-camera raw .predictions.slp into camera H5s.

Output schema matches /root/vast/eric/slap-2m-h5s/{cam}_inference.h5:
  dataset `tracks`: float64 shape (num_sessions, max_frames, max_animals, num_nodes, 2)

Pads missing data with NaN.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import sleap_io as sio
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from lib.slap2m_paths import BENCH_CAMERAS, raw_pred_slp

MASTER_SHEET = Path("/root/vast/eric/slap-2m-h5s/master_sheet.tsv")
REF_H5 = Path("/root/vast/eric/slap-2m-h5s/back_inference.h5")
DEFAULT_OUT = Path("/root/vast/eric/luc3d-bench/outputs/predictions_h5s")


def reference_shape() -> tuple[int, int, int, int]:
    with h5py.File(REF_H5, "r") as f:
        s = f["tracks"].shape
    return int(s[1]), int(s[2]), int(s[3]), int(s[4])


def load_session_cam(session_path: str, camera: str, max_frames: int, max_animals: int, num_nodes: int) -> np.ndarray:
    """Return float64 (max_frames, max_animals, num_nodes, 2), NaN-padded."""
    out = np.full((max_frames, max_animals, num_nodes, 2), np.nan, dtype=np.float64)
    slp_path = raw_pred_slp(session_path, camera)
    if slp_path is None:
        return out
    try:
        labels = sio.load_file(str(slp_path))
        arr = labels.numpy()  # (frames, tracks, nodes, xy) float32, NaN missing
    except Exception as e:
        print(f"  load FAILED {session_path}/{camera}: {e}", flush=True)
        return out
    nf, nt, nn, _ = arr.shape
    f_end = min(nf, max_frames)
    t_end = min(nt, max_animals)
    n_end = min(nn, num_nodes)
    out[:f_end, :t_end, :n_end, :] = arr[:f_end, :t_end, :n_end, :].astype(np.float64)
    return out


def aggregate_camera(df: pd.DataFrame, camera: str, out_dir: Path, max_frames: int, max_animals: int, num_nodes: int) -> Path:
    n_sessions = len(df)
    data = np.full((n_sessions, max_frames, max_animals, num_nodes, 2), np.nan, dtype=np.float64)
    for i, row in enumerate(tqdm(df.itertuples(index=False), total=n_sessions, desc=camera)):
        data[i] = load_session_cam(row.session_path, camera, max_frames, max_animals, num_nodes)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{camera}_predictions.h5"
    with h5py.File(out_path, "w") as f:
        f.create_dataset(
            "tracks",
            data=data,
            dtype="float64",
            chunks=(1, min(4096, max_frames), max_animals, num_nodes, 2),
            compression="gzip",
            compression_opts=4,
        )
        f.attrs["source"] = "raw .predictions.slp (non-proofread)"
        f.attrs["camera"] = camera
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cameras", nargs="+", default=list(BENCH_CAMERAS))
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--limit", type=int, default=None, help="limit to N sessions (smoke test)")
    args = ap.parse_args()

    df = pd.read_csv(MASTER_SHEET, sep="\t", index_col=0)
    if args.limit:
        df = df.head(args.limit)
    print(f"Aggregating {len(df)} sessions × {len(args.cameras)} cameras → {args.out}")

    mf, ma, nn, _ = reference_shape()
    print(f"Target shape per camera: (n_sessions={len(df)}, max_frames={mf}, max_animals={ma}, num_nodes={nn}, 2)")

    for cam in args.cameras:
        out = aggregate_camera(df, cam, args.out, mf, ma, nn)
        print(f"  wrote {out}")


if __name__ == "__main__":
    main()
