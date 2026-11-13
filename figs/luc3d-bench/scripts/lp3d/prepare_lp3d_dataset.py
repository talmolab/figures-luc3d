"""Prepare an LP3D-format multi-view dataset from SLAP-2M proofread labels.

For each chosen session and each bench camera (back, backL, mid, midL, top, topL):
  - Sample frames where the single proofread instance has all 15 keypoints visible
  - Decode those frames from the CRF12 video and save as PNGs
  - Aggregate keypoints into one CSV per camera in LP3D's multi-row header format

Outputs under <out_dir>:
  labeled-data/<session>_<camera>/imgXXXXXX.png
  CollectedData_<camera>.csv      (one per view, same image_path order across views)
  calibrations/<session>.toml     (6-camera subset of the source TOMLs)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import sleap_io as sio
from tqdm import tqdm

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))
from lib.slap2m_paths import session_dir, proofread_analysis_h5, calibration_toml  # noqa

BENCH_CAMERAS = ("back", "backL", "mid", "midL", "top", "topL")
KEYPOINT_NAMES = ["Nose", "Ear_R", "Ear_L", "TTI", "TailTip", "Head", "Trunk",
                  "Tail_0", "Tail_1", "Tail_2", "Shoulder_left", "Shoulder_right",
                  "Haunch_left", "Haunch_right", "Neck"]


def find_video(session_path: str, cam: str) -> Path | None:
    """Find the CRF12 denoised .mp4 for this (session, cam). Returns None if missing."""
    cam_dir = session_dir(session_path) / cam
    if not cam_dir.exists():
        return None
    # Pattern: <cam>-<timestamp>-0000_h265_CRF12_denoised.mp4
    matches = sorted(cam_dir.glob(f"{cam}-*_h265_CRF12_denoised.mp4"))
    if not matches:
        # Some sessions use h265_10bit
        matches = sorted(cam_dir.glob(f"{cam}-*_h265_10bit_CRF12_denoised.mp4"))
    return matches[0] if matches else None


def find_proofread_slp(session_path: str, cam: str) -> Path | None:
    cam_dir = session_dir(session_path) / cam
    if not cam_dir.exists():
        return None
    matches = sorted(cam_dir.glob(f"{cam}-*.predictions.proofread.slp"))
    matches = [m for m in matches if ".analysis." not in m.name and ".static.externals." not in m.name]
    return matches[0] if matches else None


def load_labels_kpts(slp_path: Path) -> np.ndarray:
    """Load proofread labels as (n_frames, 15, 2) float64, NaN where invisible.
    Assumes single track. If multiple, takes track 0.
    """
    labels = sio.load_file(str(slp_path))
    arr = labels.numpy()  # (frames, tracks, nodes, 2)
    if arr.shape[1] >= 1:
        return arr[:, 0].astype(np.float64)
    return np.full((arr.shape[0], 15, 2), np.nan)


def select_fully_visible_frames(kpts_per_cam: dict[str, np.ndarray], n_frames_total: int,
                                 n_to_sample: int, rng: np.random.Generator) -> list[int]:
    """Find frame indices where all 15 keypoints are present in all cameras."""
    visible = np.ones(n_frames_total, dtype=bool)
    for cam, arr in kpts_per_cam.items():
        nf = min(arr.shape[0], n_frames_total)
        vis_cam = np.all(np.isfinite(arr[:nf, :, 0]) & np.isfinite(arr[:nf, :, 1]), axis=1)
        visible[:nf] &= vis_cam
        if nf < n_frames_total:
            visible[nf:] = False
    valid_idxs = np.where(visible)[0]
    if len(valid_idxs) == 0:
        return []
    if len(valid_idxs) <= n_to_sample:
        return valid_idxs.tolist()
    # Even sample across the video to capture diverse poses
    spaced = np.linspace(0, len(valid_idxs) - 1, n_to_sample).round().astype(int)
    return [int(valid_idxs[i]) for i in spaced]


def write_calibration_toml_subset(src_toml: Path, dst_toml: Path, cameras: list[str]) -> None:
    """Read source TOML, write a subset containing only requested cameras, renumbered cam_0..."""
    try:
        import tomllib
    except ImportError:
        import tomli as tomllib
    with src_toml.open("rb") as f:
        data = tomllib.load(f)
    # Collect requested entries in the order of `cameras`
    subset = {}
    metadata_lines = {}
    for k, v in data.items():
        if not k.startswith("cam_"):
            metadata_lines[k] = v
    name_to_section = {}
    for k, v in data.items():
        if k.startswith("cam_") and "name" in v:
            name_to_section[v["name"]] = v
    out_lines = []
    for i, cam_name in enumerate(cameras):
        sec = name_to_section.get(cam_name)
        if sec is None:
            raise ValueError(f"Camera '{cam_name}' missing in {src_toml}")
        out_lines.append(f"[cam_{i}]")
        out_lines.append(f'name = "{cam_name}"')
        out_lines.append(f"size = {list(sec['size'])}")
        m = sec["matrix"]
        out_lines.append("matrix = [" + ", ".join("[" + ", ".join(f"{x}" for x in row) + "]" for row in m) + "]")
        out_lines.append(f"distortions = {list(sec['distortions'])}")
        out_lines.append(f"rotation = {list(sec['rotation'])}")
        out_lines.append(f"translation = {list(sec['translation'])}")
        out_lines.append("")
    # Add a placeholder metadata section (anipose requires this)
    out_lines.append("[metadata]")
    out_lines.append('adjusted = false')
    out_lines.append('error = 0.0')
    out_lines.append("")
    dst_toml.parent.mkdir(parents=True, exist_ok=True)
    dst_toml.write_text("\n".join(out_lines))


def extract_frames(video_path: Path, frame_indices: list[int], out_dir: Path, prefix: str) -> list[str]:
    """Extract frames from video and save as PNG. Returns list of relative paths."""
    out_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    paths = []
    for fi in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(fi))
        ok, frame = cap.read()
        if not ok:
            paths.append(None)
            continue
        out_name = f"img{fi:06d}.png"
        out_file = out_dir / out_name
        cv2.imwrite(str(out_file), frame)
        rel = f"labeled-data/{prefix}/{out_name}"
        paths.append(rel)
    cap.release()
    return paths


def build_collected_csv(image_paths: list[str], kpts: np.ndarray, keypoint_names: list[str],
                       scorer: str = "slap2m") -> pd.DataFrame:
    """Construct LP3D's CSV format (3 header rows: scorer / bodyparts / coords)."""
    cols = []
    for kp in keypoint_names:
        cols.append((scorer, kp, "x"))
        cols.append((scorer, kp, "y"))
    mi = pd.MultiIndex.from_tuples(cols, names=["scorer", "bodyparts", "coords"])
    n = len(image_paths)
    data = np.full((n, len(cols)), np.nan)
    for i in range(n):
        for k in range(len(keypoint_names)):
            data[i, 2 * k] = kpts[i, k, 0]
            data[i, 2 * k + 1] = kpts[i, k, 1]
    df = pd.DataFrame(data, index=image_paths, columns=mi)
    df.index.name = None
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--master", type=Path, default=REPO / "outputs" / "predictions_master_sheet.tsv")
    ap.add_argument("--sessions", nargs="+", required=True,
                    help="Session IDs (e.g. 10072022120554) to include")
    ap.add_argument("--n-per-session", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    df = pd.read_csv(args.master, sep="\t", index_col=0)
    df = df[df["session"].astype(str).isin([str(s) for s in args.sessions])]
    if df.empty:
        print(f"No matching sessions found in master sheet for: {args.sessions}", file=sys.stderr)
        sys.exit(1)

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "labeled-data").mkdir(exist_ok=True)
    (out_dir / "calibrations").mkdir(exist_ok=True)

    rng = np.random.default_rng(args.seed)
    cam_to_rows: dict[str, list[tuple[str, np.ndarray]]] = {c: [] for c in BENCH_CAMERAS}

    for _, row in tqdm(df.iterrows(), total=len(df), desc="sessions"):
        session_id = str(row["session"])
        session_path = row["session_path"]

        # Verify all required cameras have video + slp
        kpts_per_cam = {}
        videos = {}
        slps = {}
        ok = True
        for cam in BENCH_CAMERAS:
            slp = find_proofread_slp(session_path, cam)
            vid = find_video(session_path, cam)
            if slp is None or vid is None:
                print(f"  skip {session_id}: missing {cam} slp={slp} vid={vid}")
                ok = False
                break
            videos[cam] = vid
            slps[cam] = slp
            kpts_per_cam[cam] = load_labels_kpts(slp)
        if not ok:
            continue

        n_frames = min(arr.shape[0] for arr in kpts_per_cam.values())
        frame_idxs = select_fully_visible_frames(kpts_per_cam, n_frames, args.n_per_session, rng)
        if not frame_idxs:
            print(f"  skip {session_id}: no frames fully visible in all views")
            continue
        print(f"  {session_id}: {len(frame_idxs)} frames")

        # Write per-session calibration
        src_cal = calibration_toml(session_path)
        dst_cal = out_dir / "calibrations" / f"{session_id}.toml"
        write_calibration_toml_subset(src_cal, dst_cal, list(BENCH_CAMERAS))

        # Extract frames for each view in parallel (SMB I/O bound)
        def _extract(cam):
            cam_dir = out_dir / "labeled-data" / f"{session_id}_{cam}"
            return cam, extract_frames(videos[cam], frame_idxs, cam_dir, f"{session_id}_{cam}")
        cam_rel_paths = {}
        with ThreadPoolExecutor(max_workers=6) as ex:
            futs = [ex.submit(_extract, c) for c in BENCH_CAMERAS]
            for fut in as_completed(futs):
                cam, rels = fut.result()
                cam_rel_paths[cam] = rels
        for cam in BENCH_CAMERAS:
            kpts_this_cam = kpts_per_cam[cam]
            for rp, fi in zip(cam_rel_paths[cam], frame_idxs):
                if rp is None:
                    continue
                cam_to_rows[cam].append((rp, kpts_this_cam[fi]))

    # Write CSVs per view
    for cam in BENCH_CAMERAS:
        rows = cam_to_rows[cam]
        if not rows:
            print(f"WARNING: no rows for camera {cam}")
            continue
        img_paths = [r[0] for r in rows]
        kpts = np.stack([r[1] for r in rows], axis=0)
        csv_df = build_collected_csv(img_paths, kpts, KEYPOINT_NAMES)
        csv_path = out_dir / f"CollectedData_{cam}.csv"
        csv_df.to_csv(csv_path)
        print(f"wrote {csv_path} with {len(img_paths)} frames")

    # Final summary
    print(f"\nDataset prepared at {out_dir}")
    print(f"View names: {list(BENCH_CAMERAS)}")
    counts = {c: len(rs) for c, rs in cam_to_rows.items()}
    print(f"Counts per camera: {counts}")


if __name__ == "__main__":
    main()
