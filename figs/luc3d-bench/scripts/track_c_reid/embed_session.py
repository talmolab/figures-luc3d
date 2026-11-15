"""Precompute per-frame, per-identity pose-history embeddings for a session.

Given:
  - the session JSON (LUC3D output with 3D triangulated `targets3d.points3d`),
  - the trained ReID encoder,
output a numpy file embeddings[frame, identity_id] = 128-d L2 vector.

Used downstream by `stitch_postprocess_reid.py` to compute identity-vs-tracklet
similarities at tracklet voting time.

The pose history for identity i at frame f is the last W frames of
`targets3d[i].points3d` ending at frame f (W=10 default). We compute embeddings
for every frame in the session where the identity has at least min_valid_frac
valid frames in the window. Missing identities at a frame are NaN-filled and
then linearly interpolated; if too many NaNs we record the embedding as zero.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent))
from model import EncoderConfig, PoseHistoryEncoder


def collect_targets3d(session_json: dict) -> tuple[np.ndarray, list[int], list[int]]:
    """Return (poses, frames, ids) where poses is (n_frames, n_ids, 15, 3) NaN-filled.

    `ids` is the sorted list of identityIds seen across the session.
    """
    frames = []
    seen_ids = set()
    per_frame_targets = []
    for fr in session_json["frames"]:
        frames.append(int(fr["frame"]))
        per_id = {}
        for t in (fr.get("targets3d") or []):
            ident = t.get("identityId")
            pts = t.get("points3d")
            if ident is None or pts is None:
                continue
            # Some points3d entries can have nulls/None for missing keypoints.
            # Convert to (15, 3) with NaN for missing.
            try:
                arr = np.array([
                    [np.nan, np.nan, np.nan] if (p is None or (isinstance(p, (list, tuple)) and any(x is None for x in p)))
                    else p
                    for p in pts
                ], dtype=np.float32)
            except Exception:
                continue
            if arr.shape != (15, 3):
                continue
            per_id[int(ident)] = arr
            seen_ids.add(int(ident))
        per_frame_targets.append(per_id)
    ids = sorted(seen_ids)
    id_to_col = {i: c for c, i in enumerate(ids)}
    n_frames = len(frames)
    n_ids = len(ids)
    K = 15
    poses = np.full((n_frames, n_ids, K, 3), np.nan, dtype=np.float32)
    for fi, per_id in enumerate(per_frame_targets):
        for ident, pts in per_id.items():
            poses[fi, id_to_col[ident]] = pts
    return poses, frames, ids


def linear_impute_1d(x: np.ndarray) -> np.ndarray:
    """Linearly interpolate NaNs in x; if all NaN, leave as zeros."""
    n = len(x)
    valid = ~np.isnan(x)
    if not valid.any():
        return np.zeros_like(x)
    if valid.all():
        return x
    idx = np.arange(n)
    out = x.copy()
    out[~valid] = np.interp(idx[~valid], idx[valid], x[valid])
    return out


def embed_session(
    poses: np.ndarray,                # (n_frames, n_ids, K, 3) NaN-filled
    encoder: PoseHistoryEncoder,
    device: str,
    window_len: int = 10,
    min_valid_frac: float = 0.7,
    batch_size: int = 4096,
) -> np.ndarray:
    """Compute per-frame per-identity embeddings (n_frames, n_ids, d_embed)."""
    n_frames, n_ids, K, _ = poses.shape
    d = encoder.cfg.d_embed
    out = np.zeros((n_frames, n_ids, d), dtype=np.float32)
    valid_mask = np.zeros((n_frames, n_ids), dtype=bool)

    encoder.eval()
    # Build batches across (frame_idx, identity) where we can produce a window.
    requests = []
    inputs = []
    for ident in range(n_ids):
        # Per-identity, precompute imputed pose trajectory once.
        traj = poses[:, ident]                              # (n_frames, K, 3)
        # Imputation: linear interpolation per coord, only if at least some valid.
        present = ~np.isnan(traj).any(axis=(1, 2))           # (n_frames,)
        if present.sum() == 0:
            continue
        imputed = np.zeros_like(traj)
        for kk in range(K):
            for dd in range(3):
                imputed[:, kk, dd] = linear_impute_1d(traj[:, kk, dd])

        for fi in range(window_len - 1, n_frames):
            start = fi - window_len + 1
            end = fi + 1
            present_w = present[start:end]
            if present_w.sum() / window_len < min_valid_frac:
                continue
            win = imputed[start:end]   # (window_len, K, 3)
            if np.isnan(win).any():
                continue
            requests.append((fi, ident))
            inputs.append(win)

    if not inputs:
        return out, valid_mask

    inputs_arr = np.stack(inputs, axis=0).astype(np.float32)  # (M, T, K, 3)
    with torch.no_grad():
        for chunk_start in range(0, len(inputs_arr), batch_size):
            chunk = inputs_arr[chunk_start:chunk_start + batch_size]
            t = torch.from_numpy(chunk).to(device)
            emb = encoder(t).cpu().numpy()
            for j in range(len(chunk)):
                fi, ident = requests[chunk_start + j]
                out[fi, ident] = emb[j]
                valid_mask[fi, ident] = True

    return out, valid_mask


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="in_dir", required=True)
    p.add_argument("--out", dest="out_dir", required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--window-len", type=int, default=10)
    p.add_argument("--min-valid-frac", type=float, default=0.7)
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch-size", type=int, default=4096)
    args = p.parse_args()

    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt = torch.load(args.ckpt, map_location=args.device, weights_only=False)
    cfg = EncoderConfig(**ckpt["cfg"])
    model = PoseHistoryEncoder(cfg).to(args.device)
    model.load_state_dict(ckpt["model"])

    files = sorted(in_dir.glob("*.json"))
    for jp in tqdm(files, desc="sessions"):
        out_npz = out_dir / (jp.stem + ".npz")
        if out_npz.exists():
            continue
        with open(jp) as f:
            d = json.load(f)
        poses, frames, ids = collect_targets3d(d)
        if poses.size == 0 or len(ids) == 0:
            np.savez_compressed(out_npz, embeddings=np.zeros((0, 0, cfg.d_embed), dtype=np.float32),
                                valid_mask=np.zeros((0, 0), dtype=bool),
                                frames=np.array([]), ids=np.array([]))
            continue
        embeddings, valid_mask = embed_session(
            poses, model, device=args.device,
            window_len=args.window_len, min_valid_frac=args.min_valid_frac,
            batch_size=args.batch_size,
        )
        np.savez_compressed(
            out_npz,
            embeddings=embeddings,
            valid_mask=valid_mask,
            frames=np.array(frames, dtype=np.int64),
            ids=np.array(ids, dtype=np.int64),
        )


if __name__ == "__main__":
    main()
