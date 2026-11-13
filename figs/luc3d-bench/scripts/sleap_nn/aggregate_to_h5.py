"""Aggregate per-session .slp files into one H5 per camera, matching the
schema of outputs/predictions_h5s/<cam>_predictions.h5.

Existing schema (from H5_SCHEMA.txt):
  /tracks   (n_sessions, n_frames, n_animals, n_nodes, 2)  float64
  attrs: camera=<str>, source=<str>

No 'sessions' or 'scores' datasets in the baseline — we match exactly.
"""
from __future__ import annotations
import argparse
import h5py
import numpy as np
import sleap_io as sio
from pathlib import Path
import pandas as pd

REPO = Path(__file__).resolve().parents[2]


def aggregate_cam(retrack_dir, cam, out_h5, session_ids,
                  n_animals, n_nodes, n_frames):
    """Aggregate per-session .slp files for one camera into a single H5.

    Args:
        retrack_dir: Root directory containing <session>/<cam>.slp files.
        cam: Camera name (e.g. 'back').
        out_h5: Output H5 file path.
        session_ids: Ordered list of session identifiers (strings).
        n_animals: Max animals per frame (track slots).
        n_nodes: Number of skeleton nodes.
        n_frames: Max frames per session (shorter sessions are NaN-padded).
    """
    retrack_dir = Path(retrack_dir)
    n_sess = len(session_ids)
    tracks = np.full((n_sess, n_frames, n_animals, n_nodes, 2), np.nan,
                     dtype=np.float64)

    for si, sess in enumerate(session_ids):
        slp_path = retrack_dir / sess / f"{cam}.slp"
        if not slp_path.exists():
            continue
        labels = sio.load_slp(slp_path)
        skeleton = labels.skeletons[0] if labels.skeletons else None

        # Slot assignment by TOP-N TRACK LENGTH.
        # sleap-nn emits stable per-camera track IDs (the whole point of
        # tracking is to preserve identity across frames). For fragmented
        # multi-animal sessions, the dominant mice produce the longest tracks;
        # short noise tracks are discarded.
        #
        # Algorithm:
        #   1. Count frames per track (by track NAME — None tracks ignored).
        #   2. Pick the top n_animals tracks by frame count.
        #   3. Map track name → slot 0..n_animals-1 in length-descending order.
        #   4. Per frame, copy instances whose track is in the top-N to their
        #      assigned slot. Instances from other tracks are dropped.
        #
        # This restores STABLE slot identity across frames (slot 0 = same
        # mouse throughout the session), which is what LUC3D's matchPairwise
        # uses as the per-camera continuity signal via the `cam:trackIdx` key.
        from collections import Counter
        track_counts: Counter = Counter()
        for lf in labels.labeled_frames:
            for inst in lf.instances:
                if inst.track and inst.track.name:
                    track_counts[inst.track.name] += 1
        # Pick top-N by frame count; ties broken by first appearance for stability.
        # Use insertion order from Counter.most_common().
        top_tracks = [name for name, _ in track_counts.most_common(n_animals)]
        track_to_slot = {name: idx for idx, name in enumerate(top_tracks)}

        for lf in labels.labeled_frames:
            fi = lf.frame_idx
            if fi >= n_frames:
                continue
            for inst in lf.instances:
                if not inst.track or inst.track.name not in track_to_slot:
                    continue
                slot_idx = track_to_slot[inst.track.name]
                pts = inst.points
                if hasattr(pts, "dtype") and "xy" in pts.dtype.names:
                    xy = pts["xy"]
                    visible = pts["visible"]
                    n = min(len(xy), n_nodes)
                    for ni in range(n):
                        if visible[ni]:
                            tracks[si, fi, slot_idx, ni, 0] = xy[ni, 0]
                            tracks[si, fi, slot_idx, ni, 1] = xy[ni, 1]
                else:
                    if skeleton is None:
                        continue
                    for ni, node in enumerate(skeleton.nodes[:n_nodes]):
                        p = pts.get(node.name) if hasattr(pts, "get") else None
                        if p is None:
                            continue
                        if getattr(p, "visible", True):
                            tracks[si, fi, slot_idx, ni, 0] = getattr(p, "x", np.nan)
                            tracks[si, fi, slot_idx, ni, 1] = getattr(p, "y", np.nan)

    out_h5 = Path(out_h5)
    out_h5.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(out_h5, "w") as f:
        f.create_dataset("tracks", data=tracks, compression="gzip",
                         compression_opts=4)
        f.attrs["camera"] = cam
        f.attrs["source"] = "sleap-nn retracked"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--retrack-dir",
                    default=str(REPO / "outputs" / "sleap_nn_retracked"))
    ap.add_argument("--out-dir",
                    default=str(REPO / "outputs" / "sleap_nn_predictions_h5s"))
    ap.add_argument("--master",
                    default=str(REPO / "outputs" / "predictions_master_sheet.tsv"))
    a = ap.parse_args()
    df = pd.read_csv(a.master, sep="\t")
    session_ids = [str(s) for s in df["session"]]
    # Pull dims from the existing predictions_h5s (baseline schema).
    template = REPO / "outputs" / "predictions_h5s" / "back_predictions.h5"
    with h5py.File(template) as f:
        _, n_frames, n_animals, n_nodes, _ = f["tracks"].shape
    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for cam in ["back", "backL", "mid", "midL", "top", "topL"]:
        out_path = out_dir / f"{cam}_predictions.h5"
        aggregate_cam(a.retrack_dir, cam, out_path,
                      session_ids, n_animals, n_nodes, n_frames)
        print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
