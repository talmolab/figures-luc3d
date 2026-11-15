#!/usr/bin/env python3
"""Score a UDMT camera run against proofread GT, using the SAME per-camera 2D MOT
convention as scripts/evaluate.py (so numbers are directly comparable to the bench).

UDMT is OFF-POOL: its per-animal boxes are the tracker's own output (raw
tracking-results/<vid>/<latest>/<ds>_<id>_new.txt = [x,y,w,h] in resized coords),
NOT the shared sleap-nn detection pool. We map boxes back to original pixels (/RESIZE),
convert to [x1,y1,x2,y2] (matching evaluate.bbox_from_kpts), and run motmetrics with the
identical iou_matrix / auto_id setup.

Writes one row per (session,cam) to an eval CSV with tracker=="luc3d" so the existing
append_metrics_row.py / plotting code can ingest it.
"""
from __future__ import annotations
import argparse, csv, sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))
import evaluate as ev  # noqa: E402  (also installs np.asfarray shim + imports motmetrics)
import motmetrics as mm  # noqa: E402
import h5py  # noqa: E402

sys.path.insert(0, str(REPO / "scripts" / "lib"))
from slap2m_paths import proofread_analysis_h5  # noqa: E402

RESIZE = 0.8
IOU_THRESH = 0.5
EVAL_COLS = ["session", "camera", "tracker", "difficulty", "bedding", "animals",
             "mota", "motp", "idf1", "idp", "idr", "idtp", "idfp", "idfn",
             "num_switches", "num_fragmentations", "precision", "recall",
             "num_false_positives", "num_misses", "num_detections", "num_objects",
             "num_predictions", "mostly_tracked", "partially_tracked", "mostly_lost",
             "num_unique_objects", "num_frames"]


def load_udmt_boxes(proj: Path, vid: str) -> np.ndarray:
    """Return (n_animals, n_frames, 4) boxes [x1,y1,x2,y2] in ORIGINAL coords."""
    res = proj / "tracking-results" / vid
    subs = [d for d in res.iterdir() if d.is_dir()]
    if not subs:
        raise RuntimeError(f"no raw tracking subfolder in {res}")
    latest = max(subs, key=lambda d: d.stat().st_mtime)
    files = sorted(latest.glob(f"{vid}_*_new.txt"), key=lambda p: int(p.stem.split("_")[-2]))
    if not files:
        raise RuntimeError(f"no *_new.txt in {latest}")
    arrs = []
    for f in files:
        a = np.loadtxt(f, delimiter=",", ndmin=2)  # [x,y,w,h] resized coords
        a = a / RESIZE
        boxes = np.column_stack([a[:, 0], a[:, 1], a[:, 0] + a[:, 2], a[:, 1] + a[:, 3]])
        arrs.append(boxes)
    n = min(len(a) for a in arrs)
    return np.stack([a[:n] for a in arrs], axis=0)  # (n_animals, n_frames, 4)


def score(session_path: str, cam: str, proj: Path, vid: str):
    gt_h5 = proofread_analysis_h5(session_path, cam)
    if gt_h5 is None:
        raise RuntimeError(f"no proofread GT for {session_path}/{cam}")
    gt_tracks, gt_occ = ev.load_gt(gt_h5)
    udmt = load_udmt_boxes(proj, vid)            # (A, F, 4)
    n_frames = min(gt_tracks.shape[0], udmt.shape[1])
    n_animals = udmt.shape[0]

    acc = mm.MOTAccumulator(auto_id=True)
    for fi in range(n_frames):
        gt_boxes, gt_ids = [], []
        for t in range(gt_tracks.shape[1]):
            if not gt_occ[fi, t]:
                continue
            b = ev.bbox_from_kpts(gt_tracks[fi, t])
            if b is None:
                continue
            gt_boxes.append(b); gt_ids.append(int(t))
        pr_boxes, pr_ids = [], []
        for a in range(n_animals):
            box = udmt[a, fi]
            if np.all(np.isfinite(box)) and box[2] > box[0] and box[3] > box[1]:
                pr_boxes.append(box); pr_ids.append(a)
        gt_np = np.array(gt_boxes) if gt_boxes else np.empty((0, 4))
        pr_np = np.array(pr_boxes) if pr_boxes else np.empty((0, 4))
        dist = mm.distances.iou_matrix(gt_np, pr_np, max_iou=1.0 - IOU_THRESH)
        acc.update(gt_ids, pr_ids, dist)

    mh = mm.metrics.create()
    names = ["mota", "motp", "idf1", "idp", "idr", "idtp", "idfp", "idfn",
             "num_switches", "num_fragmentations", "precision", "recall",
             "num_false_positives", "num_misses", "num_detections", "num_objects",
             "num_predictions", "mostly_tracked", "partially_tracked", "mostly_lost",
             "num_unique_objects", "num_frames"]
    summ = mh.compute(acc, metrics=names, name="udmt")
    row = {k: summ[k]["udmt"] for k in names}
    return row, n_frames, n_animals


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", required=True)
    ap.add_argument("--session-path", required=True, help="e.g. 2022-10-07/10072022145420")
    ap.add_argument("--cam", required=True)
    ap.add_argument("--proj", required=True)
    ap.add_argument("--vid", required=True)
    ap.add_argument("--difficulty", default="4")
    ap.add_argument("--bedding", default="black")
    ap.add_argument("--animals", default="4")
    ap.add_argument("--out-csv", required=True)
    args = ap.parse_args()

    row, nf, na = score(args.session_path, args.cam, Path(args.proj), args.vid)
    full = {"session": args.session, "camera": args.cam, "tracker": "luc3d",
            "difficulty": args.difficulty, "bedding": args.bedding, "animals": args.animals}
    full.update(row)
    out = Path(args.out_csv)
    exists = out.exists()
    with open(out, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=EVAL_COLS)
        if not exists:
            w.writeheader()
        w.writerow({k: full.get(k, "") for k in EVAL_COLS})
    print(f"{args.cam}: IDF1={row['idf1']:.3f} switches={int(row['num_switches'])} "
          f"frames={int(row['num_frames'])} -> {out}")


if __name__ == "__main__":
    main()
