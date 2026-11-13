"""Build sio.RecordingSession + sio.Labels for the CrossViewTracker.

For each SLAP-2M session this loads:
  - 6 bench cameras (back, backL, mid, midL, top, topL) from `calibration.toml`
  - 6 per-camera `.predictions.slp` files (raw, non-proofread)

It produces a `sio.RecordingSession` linking each camera to its `sio.Video`,
plus a `sio.Labels` whose `labeled_frames` are the per-camera predictions.

The raw track names (e.g. "track_0", "track_1") are preserved on every
PredictedInstance — they index into `predictions_h5s/{cam}_predictions.h5` as
`tracks[session_idx, frame, raw_track_idx, ...]`. We will rely on these to
emit the `[{cam}:{raw_track_idx}, identity_id]` assignments in the LUC3D
JSON output schema after tracking.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import sleap_io as sio
import toml

# Shared canonical per-frame detection ordering (same function the H5 aggregator
# uses) so Liezl's emitted "{cam}:{det_pos}" indexes the identical detection slot.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "lib"))
from canonical_order import sort_instances  # noqa: E402

# Cameras fed to the CrossViewTracker. Default = the 6 GT-scored bench cameras.
# Set LIEZL_INPUT_CAMERAS (comma-separated) to add extra views (e.g. side,sideL)
# purely as triangulation/association constraints — those views are NOT scored
# (evaluate.py only scores BENCH_CAMERAS, which have proofread GT).
BENCH_CAMERAS = tuple(
    c.strip() for c in os.environ.get(
        "LIEZL_INPUT_CAMERAS", "back,backL,mid,midL,top,topL"
    ).split(",") if c.strip()
)


def _camera_from_toml_section(d: dict) -> sio.Camera:
    return sio.Camera(
        matrix=np.asarray(d["matrix"], dtype=np.float64),
        rvec=np.asarray(d["rotation"], dtype=np.float64),
        tvec=np.asarray(d["translation"], dtype=np.float64),
        dist=np.asarray(d["distortions"], dtype=np.float64),
        size=tuple(d["size"]),
        name=d["name"],
    )


def load_bench_camera_group(calibration_toml: Path) -> tuple[sio.CameraGroup, dict[str, sio.Camera]]:
    """Load `CameraGroup` filtered to BENCH_CAMERAS.

    Returns the camera group (cameras ordered as in BENCH_CAMERAS) and a
    name → camera mapping.
    """
    master = toml.load(str(calibration_toml))
    by_name: dict[str, sio.Camera] = {}
    for k, v in master.items():
        if k == "metadata":
            continue
        cam = _camera_from_toml_section(v)
        by_name[cam.name] = cam
    cams = []
    for name in BENCH_CAMERAS:
        if name not in by_name:
            raise RuntimeError(f"calibration {calibration_toml} missing camera {name}")
        cams.append(by_name[name])
    return sio.CameraGroup(cameras=cams), {c.name: c for c in cams}


def build_session_labels(
    calibration_toml: Path,
    raw_slp_by_cam: dict[str, Path],
    skeleton: sio.Skeleton | None = None,
) -> tuple[sio.RecordingSession, sio.Labels, dict[str, list[str]]]:
    """Build `(RecordingSession, Labels, raw_track_names_by_cam)`.

    `raw_track_names_by_cam[cam]` is the ordered list of `Track.name`s from the
    per-camera predictions SLP. The index of a track in this list IS the
    `raw_track_idx` that will be emitted as `"{cam}:{idx}"` in the LUC3D JSON.

    All cameras share one Skeleton (the first one we encounter, validated against the others).
    """
    camera_group, cam_by_name = load_bench_camera_group(calibration_toml)
    session = sio.RecordingSession(camera_group=camera_group)

    all_labeled_frames: list[sio.LabeledFrame] = []
    all_videos: list[sio.Video] = []
    raw_track_names_by_cam: dict[str, list[str]] = {}
    skel: sio.Skeleton | None = skeleton

    # Side-table to record (cam_name, raw_track_idx) for every original instance.
    # Keyed by id(Instance). This survives mutation of `instance.track` by the
    # CrossViewTracker because we capture it BEFORE tracking starts.
    raw_origin_by_inst_id: dict[int, tuple[str, int]] = {}

    for cam_name in BENCH_CAMERAS:
        slp = raw_slp_by_cam.get(cam_name)
        if slp is None or not slp.exists():
            # No predictions for this camera: still register the camera but no labeled frames.
            raw_track_names_by_cam[cam_name] = []
            continue
        labels_cam = sio.load_slp(str(slp), open_videos=False)
        if skel is None and labels_cam.skeletons:
            skel = labels_cam.skeletons[0]
        # Detections-only input has NO track identity. We index each detection by
        # its CANONICAL per-frame slot (det_pos) instead of a track index, so the
        # emitted "{cam}:{det_pos}" matches the widened-H5 slot exactly.
        track_names: list[str] = [t.name for t in labels_cam.tracks]
        raw_track_names_by_cam[cam_name] = track_names

        # Each cam has 1 video.
        for v in labels_cam.videos:
            all_videos.append(v)
            session.add_video(v, cam_by_name[cam_name])
        # Re-bind every instance to the shared skeleton so Labels has 1 skeleton.
        # ALSO: record the (cam, det_pos) origin for each instance BEFORE the
        # tracker mutates its `.track`. det_pos = the instance's position in the
        # canonically-sorted per-frame detection list (score desc, then centroid).
        for lf in labels_cam.labeled_frames:
            for det_pos, inst in enumerate(sort_instances(lf.instances)):
                if inst.skeleton is not skel:
                    inst.skeleton = skel
                raw_origin_by_inst_id[id(inst)] = (cam_name, det_pos)
        all_labeled_frames.extend(labels_cam.labeled_frames)

    skeletons = [skel] if skel is not None else []

    # Build a deduplicated tracks list from the union of per-camera tracks.
    track_names_set = []
    seen = set()
    for cam in BENCH_CAMERAS:
        for tn in raw_track_names_by_cam.get(cam, []):
            if tn not in seen:
                seen.add(tn)
                track_names_set.append(tn)
    tracks = [sio.Track(name=n) for n in track_names_set]
    # Re-assign per-instance .track from the (possibly new) Track objects in our shared list.
    track_by_name = {t.name: t for t in tracks}
    for lf in all_labeled_frames:
        for inst in lf.instances:
            if inst.track is not None and inst.track.name in track_by_name:
                inst.track = track_by_name[inst.track.name]

    labels = sio.Labels(
        labeled_frames=all_labeled_frames,
        videos=all_videos,
        skeletons=skeletons,
        tracks=tracks,
        sessions=[session],
    )
    return session, labels, raw_track_names_by_cam, raw_origin_by_inst_id


def load_master_sheet(tsv_path: Path):
    import pandas as pd
    return pd.read_csv(tsv_path, sep="\t", index_col=0)


def raw_pred_slp_paths(row) -> dict[str, Path]:
    """Return per-camera raw_pred_slp paths from a master-sheet row."""
    out = {}
    for cam in BENCH_CAMERAS:
        col = f"{cam}_raw_pred_slp"
        val = row.get(col, None)
        if isinstance(val, str) and val.strip():
            out[cam] = Path(val)
    return out


if __name__ == "__main__":
    import argparse, sys, json
    ap = argparse.ArgumentParser(description="Smoke-build a session for the CrossViewTracker.")
    ap.add_argument("--session", default="10072022120554", help="session id to build")
    ap.add_argument(
        "--master-sheet",
        default="/root/vast/eric/luc3d-bench/outputs/predictions_master_sheet.tsv",
    )
    args = ap.parse_args()

    df = load_master_sheet(Path(args.master_sheet))
    row = df[df["session"].astype(str) == args.session].iloc[0]
    calib = Path(row["calibration_toml"])
    raw_slps = raw_pred_slp_paths(row)
    print("calibration:", calib, file=sys.stderr)
    for k, v in raw_slps.items():
        print(f"  {k}: {v}", file=sys.stderr)

    session, labels, raw_tracks, raw_origin = build_session_labels(calib, raw_slps)
    print("RecordingSession cameras:", [c.name for c in session.cameras], file=sys.stderr)
    print("Labels:", labels, file=sys.stderr)
    print("raw_tracks per cam:", {k: v for k, v in raw_tracks.items()}, file=sys.stderr)
    print(json.dumps({"ok": True, "n_lfs": len(labels.labeled_frames),
                      "n_videos": len(labels.videos),
                      "n_tracks_global": len(labels.tracks)}))
