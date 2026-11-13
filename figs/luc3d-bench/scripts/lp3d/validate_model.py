"""Quick validation: run LP3D MVT on a held-out test session and compute
median pixel error vs SLAP-2M proofread labels."""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import sleap_io as sio

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


def find_proofread_slp(session_path: str, cam: str) -> Path | None:
    cam_dir = session_dir(session_path) / cam
    if not cam_dir.exists():
        return None
    matches = sorted(cam_dir.glob(f"{cam}-*.predictions.proofread.slp"))
    matches = [m for m in matches if ".analysis." not in m.name and ".static.externals." not in m.name]
    return matches[0] if matches else None


def load_gt_kpts(slp_path: Path) -> np.ndarray:
    labels = sio.load_file(str(slp_path))
    arr = labels.numpy()  # (frames, tracks, nodes, 2)
    return arr[:, 0].astype(np.float64) if arr.shape[1] >= 1 else np.full((arr.shape[0], 15, 2), np.nan)


def csv_to_kpts(csv_path: Path, num_keypoints: int = 15) -> np.ndarray:
    df = pd.read_csv(csv_path, header=[0, 1, 2], index_col=0)
    n = len(df)
    out = np.full((n, num_keypoints, 2), np.nan, dtype=np.float64)
    kp_names = []
    for c in df.columns:
        if c[2] == "x" and c[1] not in kp_names:
            kp_names.append(c[1])
        if len(kp_names) >= num_keypoints:
            break
    for k, kp in enumerate(kp_names):
        x_col = [c for c in df.columns if c[1] == kp and c[2] == "x"][0]
        y_col = [c for c in df.columns if c[1] == kp and c[2] == "y"][0]
        out[:, k, 0] = df[x_col].to_numpy(dtype=np.float64)
        out[:, k, 1] = df[y_col].to_numpy(dtype=np.float64)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", type=Path, required=True)
    ap.add_argument("--session-id", required=True, help="held-out test session ID")
    ap.add_argument("--master", type=Path, default=REPO / "outputs" / "predictions_master_sheet.tsv")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--max-frames", type=int, default=None, help="cap frames for quick val")
    args = ap.parse_args()

    df = pd.read_csv(args.master, sep="\t", index_col=0)
    row = df[df["session"].astype(str) == args.session_id]
    if len(row) == 0:
        print(f"Session {args.session_id} not in master sheet.")
        sys.exit(1)
    row = row.iloc[0]

    from lightning_pose.api.model import Model
    model = Model.from_dir(args.model_dir)

    videos = {}
    slps = {}
    for cam in BENCH_CAMERAS:
        v = find_video(row["session_path"], cam)
        s = find_proofread_slp(row["session_path"], cam)
        if v is None or s is None:
            print(f"Missing {cam} video or proofread")
            sys.exit(1)
        videos[cam] = v
        slps[cam] = s

    args.out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Running inference on {args.session_id} (6 views)...")
    t0 = time.time()
    model.predict_on_video_file_multiview(
        video_file_per_view=[str(videos[c]) for c in BENCH_CAMERAS],
        output_dir=args.out_dir,
        compute_metrics=False,
        generate_labeled_video=False,
    )
    dt = time.time() - t0
    print(f"Inference took {dt:.1f}s")

    # Compute median pixel error per cam
    print("\nPer-camera median pixel error vs proofread GT:")
    print(f"{'cam':<8} {'med_px':<10} {'p75_px':<10} {'p95_px':<10} {'n_frames':<10}")
    rows = []
    for cam in BENCH_CAMERAS:
        csv_files = list(args.out_dir.glob(f"{videos[cam].stem}.csv"))
        if not csv_files:
            csv_files = list(args.out_dir.glob(f"{cam}-*.csv"))
        if not csv_files:
            print(f"{cam:<8} no CSV found")
            continue
        pred = csv_to_kpts(csv_files[0])
        gt = load_gt_kpts(slps[cam])
        nf = min(pred.shape[0], gt.shape[0])
        if args.max_frames:
            nf = min(nf, args.max_frames)
        pred = pred[:nf]
        gt = gt[:nf]
        # pixel error
        d = np.sqrt(((pred - gt) ** 2).sum(axis=-1))
        valid = np.isfinite(d)
        if not valid.any():
            print(f"{cam:<8} no valid")
            continue
        med = float(np.median(d[valid]))
        p75 = float(np.percentile(d[valid], 75))
        p95 = float(np.percentile(d[valid], 95))
        print(f"{cam:<8} {med:<10.2f} {p75:<10.2f} {p95:<10.2f} {nf:<10}")
        rows.append({"cam": cam, "med_px": med, "p75_px": p75, "p95_px": p95, "n_frames": nf})
    pd.DataFrame(rows).to_csv(args.out_dir / "pixel_error_summary.csv", index=False)
    print(f"\nWrote {args.out_dir / 'pixel_error_summary.csv'}")


if __name__ == "__main__":
    main()
