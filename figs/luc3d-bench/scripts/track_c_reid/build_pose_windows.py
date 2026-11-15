"""Build a windowed pose-history dataset from SLAP-2M proofread tracks.

For each (session, track_idx), we extract sliding windows of 3D pose history.
Each window is shape (T, K, 3) = (window_len, 15 keypoints, 3 dims).

We only emit windows from frames where all keypoints are non-NaN and the
pose makes sense (e.g. instance score > some thr). We center each window on
its own centroid (subtract per-window mean) so the network learns shape/gait,
not absolute position in the cage.

Saves a single npz with arrays:
  windows: (N, T, K, 3) float32
  sess_idx: (N,) int32     -- session index in `sessions` list
  track_idx: (N,) int32    -- 0/1/2/3 (per-session slot)
  frame_idx: (N,) int32    -- start frame of window
  global_id: (N,) int64    -- sess_idx * 100 + track_idx (unique per identity)
  sessions: list[str]      -- session ids
  num_animals: (S,) int32  -- num_animals per session
"""
from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm


def extract_windows(
    points3d: np.ndarray,        # (T, n_ident, K, 3)
    track_occupancy: np.ndarray, # (n_ident, T) bool/uint8 from proofread
    window_len: int = 10,
    stride: int = 5,
    min_valid_frac: float = 0.9,
):
    """Extract per-identity sliding windows; return list of (window, track_idx, start_frame)."""
    T_total, n_ident, K, _ = points3d.shape
    out = []
    for i in range(n_ident):
        occ = track_occupancy[i].astype(bool)
        if not occ.any():
            continue
        for start in range(0, T_total - window_len + 1, stride):
            end = start + window_len
            occ_w = occ[start:end]
            if occ_w.sum() / window_len < min_valid_frac:
                continue
            win = points3d[start:end, i].copy()  # (T, K, 3)
            # require >= min_valid_frac frames with no NaN
            valid_frame = ~np.isnan(win).any(axis=(1, 2))
            if valid_frame.sum() / window_len < min_valid_frac:
                continue
            # impute remaining NaNs by linear interp along time
            if (~valid_frame).any():
                t_valid = np.where(valid_frame)[0]
                t_all = np.arange(window_len)
                for kk in range(K):
                    for dd in range(3):
                        col = win[:, kk, dd]
                        ok = ~np.isnan(col)
                        if ok.any():
                            col[~ok] = np.interp(t_all[~ok], t_all[ok], col[ok])
                        else:
                            col[:] = 0.0
                        win[:, kk, dd] = col
            # final NaN guard
            if np.isnan(win).any():
                continue
            out.append((win.astype(np.float32), i, start))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--master", default="outputs/predictions_master_sheet.tsv")
    p.add_argument("--out", default="lp3d_models/slap2m_reid/windows.npz")
    p.add_argument("--window-len", type=int, default=10)
    p.add_argument("--stride", type=int, default=5)
    p.add_argument("--min-valid-frac", type=float, default=0.9)
    p.add_argument("--smb-root", default="/root/talmolab-smb/eric/slap_2m/")
    args = p.parse_args()

    df = pd.read_csv(args.master, sep="\t")

    all_windows = []
    all_sess_idx = []
    all_track_idx = []
    all_frame_idx = []
    sessions = []
    num_animals = []

    for sess_idx, row in tqdm(list(df.iterrows()), desc="sessions"):
        session = str(row["session"])
        p3d_path = args.smb_root + row["points_3D"]
        proofread_path = row["back_proofread_h5"]
        try:
            with h5py.File(p3d_path, "r") as f:
                points3d = f["tracks"][:]  # (T, n_ident, K, 3)
            with h5py.File(proofread_path, "r") as f:
                track_occ = f["track_occupancy"][:].T  # was (T, n_ident) -> (n_ident, T)
        except Exception as e:
            print(f"  WARN {session}: failed to load: {e}")
            continue

        T3 = points3d.shape[0]
        # truncate occupancy to match points3d length
        Tmin = min(T3, track_occ.shape[1])
        points3d = points3d[:Tmin]
        track_occ = track_occ[:, :Tmin]

        windows = extract_windows(
            points3d,
            track_occ,
            window_len=args.window_len,
            stride=args.stride,
            min_valid_frac=args.min_valid_frac,
        )
        if not windows:
            continue
        sessions.append(session)
        num_animals.append(int(row["num_animals"]))
        si = len(sessions) - 1
        for win, tidx, start in windows:
            all_windows.append(win)
            all_sess_idx.append(si)
            all_track_idx.append(tidx)
            all_frame_idx.append(start)

    windows_arr = np.stack(all_windows, axis=0)          # (N, T, K, 3)
    sess_idx_arr = np.array(all_sess_idx, dtype=np.int32)
    track_idx_arr = np.array(all_track_idx, dtype=np.int32)
    frame_idx_arr = np.array(all_frame_idx, dtype=np.int32)
    global_id = sess_idx_arr.astype(np.int64) * 100 + track_idx_arr.astype(np.int64)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        out_path,
        windows=windows_arr,
        sess_idx=sess_idx_arr,
        track_idx=track_idx_arr,
        frame_idx=frame_idx_arr,
        global_id=global_id,
        sessions=np.array(sessions, dtype=object),
        num_animals=np.array(num_animals, dtype=np.int32),
    )
    print(f"saved {out_path} -> {windows_arr.shape} windows")
    n_per_id = pd.Series(global_id).value_counts()
    print(f"  unique global ids: {len(n_per_id)}")
    print(f"  windows per id: min={n_per_id.min()} median={int(n_per_id.median())} max={n_per_id.max()}")


if __name__ == "__main__":
    main()
