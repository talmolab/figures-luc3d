"""Run `sleap_3d.tracker.CrossViewTracker` on one SLAP-2M session.

Produces `outputs/PAF_3d_kalman/liezl_cross_view_results/{session_id}.json`
in the same LUC3D JSON schema that `scripts/evaluate.py` consumes:

```
{
  "sessionIdx": <int>,
  "numAnimals": <int>,
  "cameras": ["back", "backL", "mid", "midL", "top", "topL"],
  "frames": [
    {"frame": <int>,
     "assignments": [["{cam}:{raw_track_idx}", identity_id], ...],
     "targets3d": [{"identityId": id, "points3d": [[x,y,z], ...]}, ...]},
    ...
  ],
  "runtimeSeconds": <float>
}
```

The mapping:
  - `raw_track_idx`: the index into the per-camera predictions.slp's `labels.tracks`
    list (same ordering as the predictions_h5s aggregator).
  - `identity_id`: the index of the cross-view track assigned by Liezl's tracker.
  - `targets3d[*].points3d`: pulled from `state.memory.targets` (running 3D state).
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import sleap_io as sio

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_session import (
    BENCH_CAMERAS,
    build_session_labels,
    raw_pred_slp_paths,
)


def _resolve_identity_id(track_key, identity_map: dict) -> int:
    if track_key in identity_map:
        return identity_map[track_key]
    new_id = len(identity_map)
    identity_map[track_key] = new_id
    return new_id


def labels_to_luc3d_json(
    labels: sio.Labels,
    session_idx: int,
    num_animals: int,
    raw_track_names_by_cam: dict[str, list[str]],
    raw_origin_by_inst_id: dict[int, tuple[str, int]],
    targets_by_frame: dict[int, list[tuple[int, np.ndarray]]] | None = None,
) -> dict:
    """Convert post-tracking Labels into the LUC3D JSON schema."""
    cam_name_by_video: dict[int, str] = {}
    session = labels.sessions[0]
    for cam in session.cameras:
        v = session.get_video(cam)
        if v is not None:
            cam_name_by_video[id(v)] = cam.name

    # Aggregate per-frame assignments.
    frames_by_idx: dict[int, list[tuple[str, int]]] = {}
    identity_map: dict[str, int] = {}

    for lf in labels.labeled_frames:
        cam_name = cam_name_by_video.get(id(lf.video))
        if cam_name is None:
            continue
        for inst in lf.instances:
            if inst.track is None:
                continue
            origin = raw_origin_by_inst_id.get(id(inst))
            if origin is None:
                # Instance has no original raw track binding (rare). Skip.
                continue
            origin_cam, raw_idx = origin
            # Cross-check: the labeled frame's camera should match the instance's origin cam.
            if origin_cam != cam_name:
                continue
            # Use id() to disambiguate Track objects that share the same name (the tracker
            # creates many sio.Track(name="cross-view") objects).
            track_key = id(inst.track)
            identity_id = _resolve_identity_id(track_key, identity_map)
            key = f"{cam_name}:{raw_idx}"
            frames_by_idx.setdefault(lf.frame_idx, []).append((key, identity_id))

    # Build the JSON.
    frames_out = []
    for fi in sorted(frames_by_idx.keys()):
        entry = {
            "frame": int(fi),
            "assignments": [[k, int(i)] for (k, i) in frames_by_idx[fi]],
        }
        if targets_by_frame is not None and fi in targets_by_frame:
            entry["targets3d"] = [
                {"identityId": int(ident_id), "points3d": pts.tolist()}
                for (ident_id, pts) in targets_by_frame[fi]
            ]
        frames_out.append(entry)

    # identity_map maps id(Track) -> integer identity_id. We don't reveal the python id;
    # just emit a list of identities for inspection.
    return {
        "sessionIdx": int(session_idx),
        "numAnimals": int(num_animals),
        "cameras": list(BENCH_CAMERAS),
        "identities": [{"id": v, "name": f"identity_{v}"} for v in sorted(set(identity_map.values()))],
        "frames": frames_out,
    }


def run_one(
    session_row: pd.Series,
    session_idx: int,
    out_dir: Path,
    end_frame_idx: int | None = None,
    correspondence_weight_2d: float = 1.0,
    correspondence_weight_3d: float = 1.0,
    velocity_threshold: float = 10.0,
    distance_threshold: float = 50.0,
    time_penalty: float = 0.1,
    log: logging.Logger | None = None,
) -> dict:
    if log is None:
        log = logging.getLogger("run_one")

    from sleap_3d.tracker import CrossViewTracker

    session_id = str(session_row["session"])
    calib = Path(session_row["calibration_toml"])
    num_animals = int(session_row["num_animals"])
    raw_slps = raw_pred_slp_paths(session_row)

    log.info(f"[{session_id}] building session/labels...")
    session, labels, raw_track_names_by_cam, raw_origin_by_inst_id = build_session_labels(calib, raw_slps)
    log.info(
        f"[{session_id}] labels=({len(labels.labeled_frames)} lfs, "
        f"{len(labels.videos)} vids, {len(labels.tracks)} tracks); "
        f"cameras={[c.name for c in session.cameras]}"
    )

    tracker = CrossViewTracker(
        correspondence_weight_2d=correspondence_weight_2d,
        correspondence_weight_3d=correspondence_weight_3d,
        velocity_threshold=velocity_threshold,
        distance_threshold=distance_threshold,
        time_penalty=time_penalty,
    )
    log.info(f"[{session_id}] tracking (end_frame_idx={end_frame_idx})...")
    t0 = time.time()
    out_labels = tracker.track(
        labels=labels,
        session=session,
        start_frame_idx=0,
        end_frame_idx=end_frame_idx,
        store_targets=True,
    )
    elapsed = time.time() - t0
    log.info(f"[{session_id}] tracking done in {elapsed:.1f}s. "
             f"labels tracks={len(out_labels.tracks)}")

    # Recover per-frame 3D targets from the tracker state. This requires identity
    # IDs to be consistent with whatever the JSON emits. The simplest stable approach:
    # walk state.memory.targets — a list of np.ndarrays (per-iteration). But iterations
    # are per-(frame,view), so we'd need to dedupe by frame_idx. As a pragmatic choice
    # we skip this; the LUC3D JSON contract treats targets3d as informational, not used
    # in evaluation.
    targets_by_frame = None

    luc3d_json = labels_to_luc3d_json(
        labels=out_labels,
        session_idx=session_idx,
        num_animals=num_animals,
        raw_track_names_by_cam=raw_track_names_by_cam,
        raw_origin_by_inst_id=raw_origin_by_inst_id,
        targets_by_frame=targets_by_frame,
    )
    luc3d_json["runtimeSeconds"] = float(elapsed)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{session_id}.json"
    with out_path.open("w") as f:
        json.dump(luc3d_json, f)
    log.info(f"[{session_id}] wrote {out_path}")
    return {"session_id": session_id, "elapsed": elapsed, "n_frames": len(luc3d_json["frames"]), "out": str(out_path)}


def setup_logging(level=logging.INFO):
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(message)s",
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", default="10072022120554")
    ap.add_argument(
        "--master-sheet",
        default="/root/vast/eric/luc3d-bench/outputs/predictions_master_sheet.tsv",
    )
    ap.add_argument(
        "--out-dir",
        default="/root/vast/eric/luc3d-bench/outputs/PAF_3d_kalman/liezl_cross_view_results",
    )
    ap.add_argument("--end-frame", type=int, default=None, help="cap end frame index (smoke test)")
    ap.add_argument("--corr2d", type=float, default=1.0)
    ap.add_argument("--corr3d", type=float, default=1.0)
    ap.add_argument("--vel-thresh", type=float, default=10.0)
    ap.add_argument("--dist-thresh", type=float, default=50.0)
    ap.add_argument("--time-penalty", type=float, default=0.1)
    args = ap.parse_args()

    setup_logging()
    df = pd.read_csv(args.master_sheet, sep="\t", index_col=0)
    df["session"] = df["session"].astype(str)
    df = df.reset_index(drop=True)
    matches = df[df["session"] == args.session]
    if matches.empty:
        raise SystemExit(f"session {args.session} not found in master sheet")
    row = matches.iloc[0]
    session_idx = int(matches.index[0])
    res = run_one(
        session_row=row,
        session_idx=session_idx,
        out_dir=Path(args.out_dir),
        end_frame_idx=args.end_frame,
        correspondence_weight_2d=args.corr2d,
        correspondence_weight_3d=args.corr3d,
        velocity_threshold=args.vel_thresh,
        distance_threshold=args.dist_thresh,
        time_penalty=args.time_penalty,
    )
    print(json.dumps(res))


if __name__ == "__main__":
    main()
