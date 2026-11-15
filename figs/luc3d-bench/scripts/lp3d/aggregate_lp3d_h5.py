"""Aggregate per-(session, cam) LP3D prediction CSVs into per-camera H5s
matching outputs/predictions_h5s/{cam}_predictions.h5 schema:
  dataset "tracks": float64 (n_sessions, max_frames, max_animals=4, 15, 2)

LP3D MVT is single-instance: we put predictions in animal slot 0, NaN elsewhere.
The aggregated H5 must have rows for ALL sessions in the master sheet (74),
with NaN for sessions we didn't run on. session_idx is the master-sheet row index.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))
from lib.slap2m_paths import BENCH_CAMERAS  # noqa


def csv_to_kpts(csv_path: Path, num_keypoints: int = 15, conf_thresh: float = 0.0) -> np.ndarray:
    """Parse LP3D prediction CSV to (n_frames, num_keypoints, 2)."""
    df = pd.read_csv(csv_path, header=[0, 1, 2], index_col=0)
    n = len(df)
    out = np.full((n, num_keypoints, 2), np.nan, dtype=np.float64)
    kp_names = []
    for c in df.columns:
        if c[2] == "x" and c[1] not in kp_names:
            kp_names.append(c[1])
        if len(kp_names) >= num_keypoints:
            break
    for k, kp in enumerate(kp_names[:num_keypoints]):
        x_col = [c for c in df.columns if c[1] == kp and c[2] == "x"][0]
        y_col = [c for c in df.columns if c[1] == kp and c[2] == "y"][0]
        like_cols = [c for c in df.columns if c[1] == kp and c[2] == "likelihood"]
        x = df[x_col].to_numpy(dtype=np.float64)
        y = df[y_col].to_numpy(dtype=np.float64)
        if like_cols and conf_thresh > 0:
            lk = df[like_cols[0]].to_numpy(dtype=np.float64)
            mask = (lk < conf_thresh) | (~np.isfinite(lk))
            x[mask] = np.nan
            y[mask] = np.nan
        out[:, k, 0] = x
        out[:, k, 1] = y
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv-root", type=Path, required=True, help="root containing <session>/<cam>.csv")
    ap.add_argument("--master", type=Path, default=REPO / "outputs" / "predictions_master_sheet.tsv")
    ap.add_argument("--ref-h5", type=Path,
                    default=REPO / "outputs" / "predictions_h5s" / "back_predictions.h5",
                    help="reference H5 to copy shape from")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--conf-thresh", type=float, default=0.0)
    args = ap.parse_args()

    with h5py.File(args.ref_h5, "r") as f:
        ref_shape = f["tracks"].shape   # (n_sessions, max_frames, max_animals, num_nodes, 2)
    n_sessions, max_frames, max_animals, num_nodes, _ = ref_shape
    print(f"Reference shape: {ref_shape}")

    df = pd.read_csv(args.master, sep="\t", index_col=0)
    assert len(df) == n_sessions, f"master sheet {len(df)} != ref H5 {n_sessions}"

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for cam in BENCH_CAMERAS:
        out_arr = np.full((n_sessions, max_frames, max_animals, num_nodes, 2),
                          np.nan, dtype=np.float64)
        for idx, row in tqdm(df.iterrows(), total=n_sessions, desc=cam):
            session_id = str(row["session"])
            csv_path = args.csv_root / session_id / f"{cam}.csv"
            if not csv_path.exists():
                continue
            try:
                kpts = csv_to_kpts(csv_path, num_keypoints=num_nodes,
                                    conf_thresh=args.conf_thresh)
            except Exception as e:
                print(f"  fail {session_id}/{cam}: {e}", file=sys.stderr)
                continue
            nf = min(kpts.shape[0], max_frames)
            out_arr[idx, :nf, 0, :, :] = kpts[:nf]
        out_path = args.out_dir / f"{cam}_predictions.h5"
        with h5py.File(out_path, "w") as f:
            f.create_dataset(
                "tracks", data=out_arr, dtype="float64",
                chunks=(1, min(4096, max_frames), max_animals, num_nodes, 2),
                compression="gzip", compression_opts=4,
            )
            f.attrs["source"] = "lightning-pose MVT predictions"
            f.attrs["camera"] = cam
            f.attrs["conf_thresh"] = args.conf_thresh
        print(f"  wrote {out_path}")


if __name__ == "__main__":
    main()
