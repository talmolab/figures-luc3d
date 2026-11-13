"""Run LP3D MVT inference on ONE session. Used as a worker by parallel orchestration."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))
from lib.slap2m_paths import session_dir, BENCH_CAMERAS  # noqa


def find_video(session_path: str, cam: str):
    cam_dir = session_dir(session_path) / cam
    if not cam_dir.exists():
        return None
    for pat in (f"{cam}-*_h265_CRF12_denoised.mp4", f"{cam}-*_h265_10bit_CRF12_denoised.mp4"):
        ms = sorted(cam_dir.glob(pat))
        if ms:
            return ms[0]
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", type=Path, required=True)
    ap.add_argument("--session-id", required=True)
    ap.add_argument("--out-csv-root", type=Path, required=True)
    ap.add_argument("--master", type=Path,
                    default=REPO / "outputs" / "predictions_master_sheet.tsv")
    ap.add_argument("--skip-existing", action="store_true")
    args = ap.parse_args()

    df = pd.read_csv(args.master, sep="\t", index_col=0)
    row = df[df["session"].astype(str) == args.session_id]
    if len(row) == 0:
        print(f"[{args.session_id}] not found in master sheet"); sys.exit(1)
    row = row.iloc[0]

    out_session_dir = args.out_csv_root / args.session_id
    out_session_dir.mkdir(parents=True, exist_ok=True)
    expected = [out_session_dir / f"{cam}.csv" for cam in BENCH_CAMERAS]
    if args.skip_existing and all(p.exists() for p in expected):
        print(f"[{args.session_id}] all CSVs exist, skipping"); return

    videos = {}
    for cam in BENCH_CAMERAS:
        v = find_video(row["session_path"], cam)
        if v is None:
            print(f"[{args.session_id}] missing video for {cam}"); sys.exit(1)
        videos[cam] = v

    from lightning_pose.api.model import Model
    model = Model.from_dir(args.model_dir)

    t0 = time.time()
    print(f"[{args.session_id}] starting inference")
    try:
        model.predict_on_video_file_multiview(
            video_file_per_view=[str(videos[c]) for c in BENCH_CAMERAS],
            output_dir=out_session_dir,
            compute_metrics=False,
            generate_labeled_video=False,
        )
    except Exception as e:
        print(f"[{args.session_id}] FAILED: {e}"); raise
    dt = time.time() - t0
    print(f"[{args.session_id}] done in {dt:.1f}s")

    # Rename CSVs to {cam}.csv
    for cam, v in videos.items():
        src = out_session_dir / f"{v.stem}.csv"
        dst = out_session_dir / f"{cam}.csv"
        if src.exists():
            src.rename(dst)


if __name__ == "__main__":
    main()
