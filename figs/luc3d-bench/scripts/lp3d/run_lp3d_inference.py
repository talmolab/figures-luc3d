"""Run LP3D MVT inference on selected SLAP-2M sessions and write per-camera H5s
matching the schema in outputs/predictions_h5s/.

For each session in --sessions:
  - Find the 6 bench-camera CRF12 videos
  - Run `Model.predict_on_video_file_multiview(...)` once
  - This produces 6 CSVs (one per view) with columns (kp_x, kp_y, kp_likelihood)
After all sessions, aggregate per-camera CSVs into per-camera H5
matching outputs/predictions_h5s/{cam}_predictions.h5:
  dataset "tracks": float64 (n_sessions, max_frames, max_animals, 15, 2)
  with NaN for missing frames. max_animals=4, max_frames from reference H5.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))
from lib.slap2m_paths import session_dir, BENCH_CAMERAS  # noqa


def find_video(session_path: str, cam: str) -> Path | None:
    cam_dir = session_dir(session_path) / cam
    if not cam_dir.exists():
        return None
    for pat in (f"{cam}-*_h265_CRF12_denoised.mp4", f"{cam}-*_h265_10bit_CRF12_denoised.mp4"):
        ms = sorted(cam_dir.glob(pat))
        if ms:
            return ms[0]
    return None


def csv_to_kpts(csv_path: Path, num_keypoints: int = 15) -> np.ndarray:
    """Parse LP3D prediction CSV to (n_frames, num_keypoints, 2) float64, NaN where missing."""
    df = pd.read_csv(csv_path, header=[0, 1, 2], index_col=0)
    # Columns: (scorer, kp_name, [x|y|likelihood])
    n = len(df)
    out = np.full((n, num_keypoints, 2), np.nan, dtype=np.float64)
    # Walk by bodypart in order
    kp_names = []
    for c in df.columns:
        if c[2] == "x" and c[1] not in kp_names:
            kp_names.append(c[1])
        if len(kp_names) == num_keypoints:
            break
    for k, kp in enumerate(kp_names):
        x_col = [c for c in df.columns if c[1] == kp and c[2] == "x"][0]
        y_col = [c for c in df.columns if c[1] == kp and c[2] == "y"][0]
        like_col = [c for c in df.columns if c[1] == kp and c[2] == "likelihood"]
        x = df[x_col].to_numpy(dtype=np.float64)
        y = df[y_col].to_numpy(dtype=np.float64)
        if like_col:
            lk = df[like_col[0]].to_numpy(dtype=np.float64)
            # Threshold low-confidence points
            mask = (lk < 0.05) | (~np.isfinite(lk))
            x[mask] = np.nan
            y[mask] = np.nan
        out[:, k, 0] = x
        out[:, k, 1] = y
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", type=Path, required=True, help="LP3D trained model dir")
    ap.add_argument("--sessions", nargs="+", required=True)
    ap.add_argument("--master", type=Path, default=REPO / "outputs" / "predictions_master_sheet.tsv")
    ap.add_argument("--out-csv-root", type=Path, required=True,
                    help="Where to dump per-(session, cam) CSV preds")
    ap.add_argument("--skip-existing", action="store_true")
    args = ap.parse_args()

    from lightning_pose.api.model import Model
    model = Model.from_dir(args.model_dir)
    print(f"Loaded model from {args.model_dir}")

    df = pd.read_csv(args.master, sep="\t", index_col=0)
    df = df[df["session"].astype(str).isin([str(s) for s in args.sessions])]
    if df.empty:
        print(f"No sessions matched.", file=sys.stderr); sys.exit(1)

    args.out_csv_root.mkdir(parents=True, exist_ok=True)
    for _, row in df.iterrows():
        session_id = str(row["session"])
        out_session_dir = args.out_csv_root / session_id
        out_session_dir.mkdir(parents=True, exist_ok=True)
        # Check existing
        expected_csvs = [out_session_dir / f"{cam}.csv" for cam in BENCH_CAMERAS]
        if args.skip_existing and all(p.exists() for p in expected_csvs):
            print(f"  skip {session_id}: all CSVs exist")
            continue

        # Gather videos
        videos = {}
        ok = True
        for cam in BENCH_CAMERAS:
            v = find_video(row["session_path"], cam)
            if v is None:
                print(f"  skip {session_id}: missing video for {cam}")
                ok = False
                break
            videos[cam] = v
        if not ok:
            continue

        t0 = time.time()
        try:
            result = model.predict_on_video_file_multiview(
                video_file_per_view=[str(videos[c]) for c in BENCH_CAMERAS],
                output_dir=out_session_dir,
                compute_metrics=False,
                generate_labeled_video=False,
            )
        except Exception as e:
            print(f"  FAIL {session_id}: {e}")
            continue
        dt = time.time() - t0
        print(f"  done {session_id} in {dt:.1f}s")

        # Rename the CSVs to {cam}.csv (LP3D puts video stem)
        for cam, v in videos.items():
            src = out_session_dir / f"{v.stem}.csv"
            dst = out_session_dir / f"{cam}.csv"
            if src.exists():
                src.rename(dst)


if __name__ == "__main__":
    main()
