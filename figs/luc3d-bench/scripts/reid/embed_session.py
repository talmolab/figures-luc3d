"""Extract & cache OSNet appearance embeddings for every detection in a session.

One .npz per (session, camera) under
  outputs/PAF_3d_kalman/_reid_cache/<weights_tag>/<session>/<cam>.npz
with arrays: feats (N,512) float16 (L2-normalized), frames (N,) int32, slots (N,) int16.

Idempotent: skips a (session,cam) whose .npz already exists unless --overwrite.

Usage (from repo root):
  PYTHONPATH=. reid_env/bin/python scripts/reid/embed_session.py --session 10072022131531
  PYTHONPATH=. reid_env/bin/python scripts/reid/embed_session.py --session-idx 6 --cams top back
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from reid.crop_detections import BENCH_CAMERAS, iter_crops, resolve_video_path
from reid.reid_encoder import DEFAULT_WEIGHTS, ReidEncoder

REPO = Path(__file__).resolve().parents[2]
MASTER = REPO / "outputs/detections_only_master_sheet.tsv"
H5_DIR = REPO / "outputs/detections_only_h5s"
CACHE_ROOT = REPO / "outputs/PAF_3d_kalman/_reid_cache"


def embed_cam(enc: ReidEncoder, session_idx: int, cam: str, video_path: Path,
              out_npz: Path, h5_dir: Path = H5_DIR, batch: int = 512, flush_every: int = 4096):
    h5_path = h5_dir / f"{cam}_predictions.h5"
    buf_crops, buf_fr, buf_sl = [], [], []
    feats_all, fr_all, sl_all = [], [], []

    def flush():
        if not buf_crops:
            return
        feats_all.append(enc.embed(buf_crops, batch_size=batch).astype(np.float16))
        fr_all.append(np.asarray(buf_fr, dtype=np.int32))
        sl_all.append(np.asarray(buf_sl, dtype=np.int16))
        buf_crops.clear(); buf_fr.clear(); buf_sl.clear()

    for fi, slot, rgb in iter_crops(session_idx, cam, h5_path, video_path):
        buf_crops.append(rgb); buf_fr.append(fi); buf_sl.append(slot)
        if len(buf_crops) >= flush_every:
            flush()
    flush()

    if feats_all:
        feats = np.concatenate(feats_all); frames = np.concatenate(fr_all); slots = np.concatenate(sl_all)
    else:
        feats = np.zeros((0, 512), np.float16); frames = np.zeros((0,), np.int32); slots = np.zeros((0,), np.int16)
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_npz, feats=feats, frames=frames, slots=slots)
    return len(feats)


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--session", help="session name (e.g. 10072022131531)")
    g.add_argument("--session-idx", type=int, help="session row index (0..73)")
    ap.add_argument("--cams", nargs="+", default=list(BENCH_CAMERAS))
    ap.add_argument("--weights", default=str(DEFAULT_WEIGHTS))
    ap.add_argument("--weights-tag", default="imagenet")
    ap.add_argument("--h5-dir", default=str(H5_DIR),
                    help="detection-pool H5 dir (default detections_only_h5s; use keeptrack_h5s for the best base)")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    h5_dir = Path(args.h5_dir)

    df = pd.read_csv(MASTER, sep="\t", index_col=0)
    if args.session is not None:
        sidx = int(df.index[df["session"].astype(str) == args.session][0])
    else:
        sidx = args.session_idx
    session = str(df.loc[sidx, "session"])

    enc = ReidEncoder(weights=Path(args.weights))
    out_dir = CACHE_ROOT / args.weights_tag / session
    for cam in args.cams:
        out_npz = out_dir / f"{cam}.npz"
        if out_npz.exists() and not args.overwrite:
            print(f"[skip] {session}/{cam} (cached)"); continue
        video = resolve_video_path(str(df.loc[sidx, f"{cam}_video"]))
        n = embed_cam(enc, sidx, cam, video, out_npz, h5_dir=h5_dir)
        print(f"[ok] {session}/{cam}: {n} embeddings -> {out_npz}")


if __name__ == "__main__":
    main()
