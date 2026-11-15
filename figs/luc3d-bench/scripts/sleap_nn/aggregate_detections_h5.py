"""Aggregate the filter-only .slp detections into one widened H5 per camera.

Replaces the top-4-by-track-length slot assignment of aggregate_to_h5.py with a
NO-DROP, canonically-ordered detection axis:

  /tracks      (n_sessions, n_frames, n_dets, n_nodes, 2)  float64, NaN-padded
  /det_scores  (n_sessions, n_frames, n_dets)              float64, NaN-padded
  attrs: camera, source, n_dets, ordering="score_desc_then_centroid"

`n_dets` = dataset-wide max detections/frame (read from the filter ledger, or
overridden via --n-dets). Within each frame the detections are written in
canonical_order.sort_instances order → slot k is the (k+1)-th canonical
detection. This is the SAME ordering every tracker and the evaluator use, so a
JSON assignment "{cam}:{slot}" indexes exactly the detection at H5[frame, slot].

Memory: the H5 is written one session at a time (a ~100 MB per-session slice),
never materializing the full multi-GB array in RAM.

If a frame somehow has more than n_dets detections (only possible when --n-dets
caps below the true max), the lowest-score detections are dropped and the exact
count is logged — never silently.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import sleap_io as sio

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "lib"))
from canonical_order import sort_instances  # noqa: E402

DEFAULT_CAMS = ["back", "backL", "mid", "midL", "top", "topL"]


def n_dets_from_ledger(ledger: Path) -> int:
    """Dataset-wide max detections/frame across all filter_only entries."""
    m = 0
    if ledger.exists():
        for line in ledger.read_text().splitlines():
            try:
                d = json.loads(line)
                m = max(m, int(d.get("max_dets_per_frame", 0)))
            except Exception:
                pass
    return m


def _instance_xy_scores(inst, n_nodes):
    """Return (xy (n_nodes,2) NaN-padded, instance_score float)."""
    xy = np.full((n_nodes, 2), np.nan, dtype=np.float64)
    pts = inst.points
    if hasattr(pts, "dtype") and pts.dtype.names and "xy" in pts.dtype.names:
        pxy = pts["xy"]
        vis = pts["visible"] if "visible" in pts.dtype.names else np.ones(len(pxy), bool)
        n = min(len(pxy), n_nodes)
        for ni in range(n):
            if vis[ni]:
                xy[ni, 0] = pxy[ni, 0]
                xy[ni, 1] = pxy[ni, 1]
    else:
        arr = np.asarray(inst.numpy(), dtype=np.float64)
        n = min(arr.shape[0], n_nodes)
        xy[:n] = arr[:n]
    score = float(inst.score) if getattr(inst, "score", None) is not None else np.nan
    return xy, score


def aggregate_cam(detections_dir, cam, out_h5, session_ids,
                  n_dets, n_nodes, n_frames, log_fh=None):
    detections_dir = Path(detections_dir)
    n_sess = len(session_ids)
    out_h5 = Path(out_h5)
    out_h5.parent.mkdir(parents=True, exist_ok=True)

    chunk_frames = min(4096, n_frames)
    total_dropped = 0
    with h5py.File(out_h5, "w") as f:
        dt = f.create_dataset(
            "tracks", shape=(n_sess, n_frames, n_dets, n_nodes, 2),
            dtype="float64", fillvalue=np.nan,
            chunks=(1, chunk_frames, n_dets, n_nodes, 2),
            compression="gzip", compression_opts=4,
        )
        ds = f.create_dataset(
            "det_scores", shape=(n_sess, n_frames, n_dets),
            dtype="float64", fillvalue=np.nan,
            chunks=(1, chunk_frames, n_dets),
            compression="gzip", compression_opts=4,
        )
        for si, sess in enumerate(session_ids):
            slp_path = detections_dir / sess / f"{cam}.slp"
            if not slp_path.exists():
                continue
            labels = sio.load_slp(str(slp_path))
            sess_tracks = np.full((n_frames, n_dets, n_nodes, 2), np.nan, dtype=np.float64)
            sess_scores = np.full((n_frames, n_dets), np.nan, dtype=np.float64)
            for lf in labels.labeled_frames:
                fi = lf.frame_idx
                if fi >= n_frames:
                    continue
                ordered = sort_instances(lf.instances)
                if len(ordered) > n_dets:
                    total_dropped += len(ordered) - n_dets
                    if log_fh is not None:
                        log_fh.write(
                            f"DROP {cam} {sess} frame {fi}: "
                            f"{len(ordered)} dets > n_dets={n_dets}, "
                            f"dropped {len(ordered) - n_dets} lowest-score\n")
                    ordered = ordered[:n_dets]
                for slot, inst in enumerate(ordered):
                    xy, score = _instance_xy_scores(inst, n_nodes)
                    sess_tracks[fi, slot] = xy
                    sess_scores[fi, slot] = score
            dt[si] = sess_tracks
            ds[si] = sess_scores
        f.attrs["camera"] = cam
        f.attrs["source"] = "filter-only detections (no tracking)"
        f.attrs["n_dets"] = int(n_dets)
        f.attrs["ordering"] = "score_desc_then_centroid"
    return total_dropped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--detections-dir", default=str(REPO / "outputs" / "detections_only"))
    ap.add_argument("--out-dir", default=str(REPO / "outputs" / "detections_only_h5s"))
    ap.add_argument("--master", default=str(REPO / "outputs" / "predictions_master_sheet.tsv"))
    ap.add_argument("--cams", nargs="+", default=DEFAULT_CAMS)
    ap.add_argument("--n-dets", type=int, default=None,
                    help="detection-axis width; default = dataset-wide max from ledger")
    ap.add_argument("--max-n-dets", type=int, default=24,
                    help="safety cap on n_dets; if ledger max exceeds this, cap "
                         "and log per-frame truncation (default 24)")
    ap.add_argument("--log", default=str(REPO / "logs" / "aggregate_detok.log"))
    a = ap.parse_args()

    df = pd.read_csv(a.master, sep="\t")
    session_ids = [str(s) for s in df["session"]]

    template = REPO / "outputs" / "predictions_h5s" / "back_predictions.h5"
    with h5py.File(template) as f:
        _, n_frames, _, n_nodes, _ = f["tracks"].shape

    ledger = Path(a.detections_dir) / "_filter_ledger.jsonl"
    ledger_max = n_dets_from_ledger(ledger)
    if a.n_dets is not None:
        n_dets = a.n_dets
    else:
        n_dets = max(1, ledger_max)
        if n_dets > a.max_n_dets:
            print(f"[aggregate_detok] ledger max dets/frame={ledger_max} exceeds "
                  f"--max-n-dets={a.max_n_dets}; capping (truncation will be logged)")
            n_dets = a.max_n_dets

    print(f"[aggregate_detok] ledger max dets/frame={ledger_max}; "
          f"using n_dets={n_dets}; n_frames={n_frames}, n_nodes={n_nodes}")

    Path(a.log).parent.mkdir(parents=True, exist_ok=True)
    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    grand_dropped = 0
    with open(a.log, "a") as log_fh:
        log_fh.write(f"=== aggregate_detok n_dets={n_dets} ledger_max={ledger_max} ===\n")
        for cam in a.cams:
            out_path = out_dir / f"{cam}_predictions.h5"
            dropped = aggregate_cam(a.detections_dir, cam, out_path,
                                    session_ids, n_dets, n_nodes, n_frames, log_fh)
            grand_dropped += dropped
            print(f"wrote {out_path}  (truncation-dropped dets: {dropped})")
        log_fh.write(f"TOTAL dropped across cams: {grand_dropped}\n")
    print(f"[aggregate_detok] DONE. total truncation-dropped detections: {grand_dropped} "
          f"(expect 0 unless n_dets was capped)")


if __name__ == "__main__":
    main()
