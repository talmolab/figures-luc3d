"""Phase-0 separation diagnostic: does the frozen OSNet color embedding tell mice apart?

For each (session, camera) we build two cosine-similarity distributions:
  * DIFFERENT animal  — the two detections present in the SAME frame (guaranteed distinct).
  * SAME animal       — two crops sharing the same Liezl identity a few frames apart.
If the SAME distribution sits clearly to the right of the DIFFERENT one, an off-the-shelf
embedding already carries usable identity signal (decision gate for the unsupervised graph).

Run (from repo root):
  PYTHONPATH=. reid_env/bin/python scripts/reid/diagnostic_separation.py \
      --sessions 10072022131531 10072022133702 --cams top back
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from reid.crop_detections import iter_crops, resolve_video_path
from reid.reid_encoder import ReidEncoder

REPO = Path(__file__).resolve().parents[2]
MASTER = REPO / "outputs/detections_only_master_sheet.tsv"
H5_DIR = REPO / "outputs/detections_only_h5s"
LIEZL_DIR = REPO / "outputs/PAF_3d_kalman/detok_liezl_results"
DIAG_DIR = REPO / "outputs/PAF_3d_kalman/_diag"


def liezl_id_timeline(session: str, cam: str) -> dict[int, dict[int, int]]:
    """frame -> {slot: identity} for one camera from Liezl's result JSON."""
    p = LIEZL_DIR / f"{session}.json"
    out: dict[int, dict[int, int]] = {}
    if not p.exists():
        return out
    data = json.loads(p.read_text())
    for entry in data["frames"]:
        fi = entry["frame"]
        for key, ident in entry["assignments"]:
            cn, slot = key.split(":")
            if cn == cam:
                out.setdefault(fi, {})[int(slot)] = int(ident)
    return out


def frames_with_two_dets(h5_path: Path, session_idx: int, cap: int = 800,
                         window: int = 9000) -> list[int]:
    """Frame indices (within [0, window)) where exactly two detections are present.

    Bounded to a window so the sequential video decode stays short for the diagnostic.
    """
    with h5py.File(h5_path, "r") as f:
        ds = f["det_scores"][session_idx]            # (n_frames, n_dets)
    ndet = np.sum(np.isfinite(ds), axis=1)
    idx = np.where(ndet == 2)[0]
    idx = idx[idx < window]
    if len(idx) > cap:                                # even temporal sample
        idx = idx[np.linspace(0, len(idx) - 1, cap).astype(int)]
    return idx.tolist()


def run_session_cam(enc: ReidEncoder, df: pd.DataFrame, session: str, cam: str, delta: int = 8):
    sidx = int(df.index[df["session"].astype(str) == session][0])
    h5_path = H5_DIR / f"{cam}_predictions.h5"
    video = resolve_video_path(str(df.loc[sidx, f"{cam}_video"]))

    frames = frames_with_two_dets(h5_path, sidx)
    if not frames:
        return None
    want = set(frames) | {f + delta for f in frames}

    # Embed every needed crop, keyed (frame, slot).
    crops, keys = [], []
    for fi, slot, rgb in iter_crops(sidx, cam, h5_path, video, frame_filter=want):
        crops.append(rgb); keys.append((fi, slot))
    if not crops:
        return None
    feats = enc.embed(crops)
    emb = {k: feats[i] for i, k in enumerate(keys)}

    # DIFFERENT-animal pairs: both slots in the same 2-det frame.
    diff = []
    for fi in frames:
        a, b = emb.get((fi, 0)), emb.get((fi, 1))
        if a is not None and b is not None:
            diff.append(float(a @ b))

    # SAME-animal pairs: same Liezl identity at fi and fi+delta.
    tl = liezl_id_timeline(session, cam)
    same = []
    for fi in frames:
        cur, nxt = tl.get(fi, {}), tl.get(fi + delta, {})
        for slot, ident in cur.items():
            a = emb.get((fi, slot))
            if a is None:
                continue
            for slot2, ident2 in nxt.items():
                if ident2 == ident:
                    b = emb.get((fi + delta, slot2))
                    if b is not None:
                        same.append(float(a @ b))
    return np.array(same), np.array(diff)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sessions", nargs="+", required=True)
    ap.add_argument("--cams", nargs="+", default=["top", "back"])
    args = ap.parse_args()

    DIAG_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(MASTER, sep="\t", index_col=0)
    enc = ReidEncoder()
    print(f"encoder on {enc.device} (half={enc.half})")

    for session in args.sessions:
        fig, axes = plt.subplots(1, len(args.cams), figsize=(6 * len(args.cams), 4.5), squeeze=False)
        for j, cam in enumerate(args.cams):
            res = run_session_cam(enc, df, session, cam)
            ax = axes[0][j]
            if res is None:
                ax.set_title(f"{cam}: no 2-det frames"); continue
            same, diff = res
            for arr, lab, col in [(same, "same animal", "#2e7d32"), (diff, "different animal", "#c2185b")]:
                if len(arr):
                    ax.hist(arr, bins=50, range=(-0.2, 1.0), alpha=0.6, label=f"{lab} (n={len(arr)})", color=col)
            sm = float(np.median(same)) if len(same) else float("nan")
            dm = float(np.median(diff)) if len(diff) else float("nan")
            ax.axvline(sm, color="#2e7d32", ls="--"); ax.axvline(dm, color="#c2185b", ls="--")
            ax.set_title(f"{cam}  Δmedian={sm - dm:+.3f}\nsame={sm:.3f} diff={dm:.3f}")
            ax.set_xlabel("cosine similarity"); ax.legend(fontsize=8)
            print(f"{session} {cam}: same_med={sm:.3f} diff_med={dm:.3f} sep={sm - dm:+.3f} "
                  f"(n_same={len(same)} n_diff={len(diff)})")
        fig.suptitle(f"OSNet frozen embedding separability — session {session}")
        fig.tight_layout()
        out = DIAG_DIR / f"reid_sim_hist_{session}.png"
        fig.savefig(out, dpi=130); plt.close(fig)
        print(f"  saved {out}")


if __name__ == "__main__":
    main()
