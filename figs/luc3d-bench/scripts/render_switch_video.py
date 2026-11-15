#!/usr/bin/env python3
"""Render a camera clip with tracker identities color-coded, so ID switches are visible.

Each predicted detection is drawn in the color of its Liezl identity id (stable color
per identity). Each is tagged with the GT animal it matches (by IoU) as "gtA"/"gtB".
A switch = the SAME physical mouse (same gt tag) suddenly changing color. Frames where
motmetrics records a SWITCH are flagged with a red banner + the inter-animal 3D distance
(from points3d GT) so you can see whether it's a close pass or a re-entry.

Usage:
  PYTHONPATH=. .venv/bin/python scripts/render_switch_video.py \
      --session 10072022145420 --cam mid \
      --results-dir outputs/PAF_3d_kalman/_liezlsweep/G_keeptrack_3d6 \
      --pred-h5-dir outputs/keeptrack_h5s \
      --out outputs/PAF_3d_kalman/_diag/videos/10072022145420_mid.mp4 \
      [--window 90]   # auto-pick densest switch cluster, +/- window frames
      [--start S --end E]  # or an explicit frame range
"""
import argparse
import json
import tomllib
from pathlib import Path

import cv2
import h5py
import numpy as np
import pandas as pd
import motmetrics as mm

from scripts.evaluate import load_gt, bbox_from_kpts, luc3d_assignments_for_cam

REPO = Path(__file__).resolve().parents[1]
MASTER = REPO / "outputs" / "detections_only_master_sheet.tsv"
SLAP2M = Path("/root/talmolab-smb/eric/slap_2m")
CAMS = ["back", "backL", "mid", "midL", "top", "topL"]

# BGR identity palette (high-contrast)
PAL = [(60, 60, 230), (230, 160, 40), (60, 200, 60), (200, 60, 200),
       (40, 220, 220), (230, 230, 230), (120, 80, 200), (80, 160, 230)]
GT_TAGS = "ABCDEFGH"


def calib_Rt(path, cam):
    calib = tomllib.load(open(path, "rb"))
    for k, c in calib.items():
        if k.startswith("cam_") and c.get("name") == cam:
            R, _ = cv2.Rodrigues(np.array(c["rotation"], float))
            t = np.array(c["translation"], float).reshape(3, 1)
            return R, t
    return None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", required=True)
    ap.add_argument("--cam", required=True)
    ap.add_argument("--results-dir", required=True)
    ap.add_argument("--pred-h5-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--window", type=int, default=120)
    ap.add_argument("--start", type=int, default=None)
    ap.add_argument("--end", type=int, default=None)
    ap.add_argument("--fps", type=int, default=30)
    a = ap.parse_args()

    ms = pd.read_csv(MASTER, sep="\t")
    ms["session"] = ms["session"].astype(str)
    row = ms[ms["session"] == a.session].iloc[0]
    sidx = int(ms.index[ms["session"] == a.session][0])
    # session index inside the per-cam H5 = row order in master sheet
    sidx = int(np.where(ms["session"].values == a.session)[0][0])

    with h5py.File(f"{a.pred_h5_dir}/{a.cam}_predictions.h5", "r") as f:
        raw = f["tracks"][sidx]      # (n_frames, n_slots, 15, 2)
    gt, occ = load_gt(row[f"{a.cam}_proofread_h5"])
    n = min(raw.shape[0], gt.shape[0]); na = raw.shape[1]
    ids = luc3d_assignments_for_cam(json.load(open(f"{a.results_dir}/{a.session}.json")), a.cam, n, na)

    p3d = None
    if isinstance(row["points_3D"], str):
        p = SLAP2M / row["points_3D"]
        if p.exists():
            with h5py.File(p, "r") as f:
                p3d = f["tracks"][:]
    R, t = calib_Rt(row["calibration_toml"], a.cam)

    # --- recompute matching to find switch frames + GT tag per predicted slot ---
    acc = mm.MOTAccumulator(auto_id=True)
    # also remember, per frame, mapping pred-id -> matched gt track (for tagging)
    for fi in range(n):
        gb, gi = [], []
        for tr in range(gt.shape[1]):
            if occ[fi, tr]:
                b = bbox_from_kpts(gt[fi, tr])
                if b is not None:
                    gb.append(b); gi.append(tr)
        pb, pi = [], []
        for s in range(na):
            b = bbox_from_kpts(raw[fi, s])
            if b is None:
                continue
            iv = int(ids[fi, s])
            if iv >= 0:
                pb.append(b); pi.append(iv)
        d = mm.distances.iou_matrix(np.array(gb) if gb else np.empty((0, 4)),
                                    np.array(pb) if pb else np.empty((0, 4)), max_iou=0.5)
        acc.update(gi, pi, d)
    ev = acc.mot_events.reset_index()
    sw = ev[ev.Type == "SWITCH"]
    sw_frames = sorted(sw.FrameId.unique().tolist())
    print(f"{a.session}/{a.cam}: {len(sw_frames)} switch frames")

    # pick window
    if a.start is not None and a.end is not None:
        start, end = a.start, a.end
    elif sw_frames:
        # densest cluster: frame with most switches within +-window
        arr = np.array(sw_frames)
        best = max(arr, key=lambda f0: int(np.sum(np.abs(arr - f0) <= a.window)))
        start, end = max(0, best - a.window), min(n, best + a.window)
        print(f"  densest cluster around frame {best} -> [{start},{end}]")
    else:
        start, end = 0, min(n, 2 * a.window)

    # GT tag per predicted-id within the window: for each frame, match pred bbox->gt by IoU
    vid = SLAP2M / row[f"{a.cam}_video"]
    cap = cv2.VideoCapture(str(vid))
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    vw = cv2.VideoWriter(a.out, cv2.VideoWriter_fourcc(*"mp4v"), a.fps, (W, H))

    sw_set = set(sw_frames)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    for fi in range(start, end):
        okf, frame = cap.read()
        if not okf:
            break
        # GT bbox + tag (stable identity = GT track index)
        gboxes = {}
        for tr in range(gt.shape[1]):
            if occ[fi, tr]:
                b = bbox_from_kpts(gt[fi, tr])
                if b is not None:
                    gboxes[tr] = b
                    cv2.rectangle(frame, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])),
                                  (180, 180, 180), 1)
                    cv2.putText(frame, f"gt{GT_TAGS[tr]}", (int(b[0]), int(b[1]) - 4),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1, cv2.LINE_AA)
        # predicted detections colored by identity
        for s in range(na):
            kp = raw[fi, s]
            b = bbox_from_kpts(kp)
            if b is None:
                continue
            iv = int(ids[fi, s])
            if iv < 0:
                continue
            col = PAL[iv % len(PAL)]
            pts = kp[np.isfinite(kp[:, 0])]
            for (x, y) in pts:
                cv2.circle(frame, (int(x), int(y)), 3, col, -1)
            cx, cy = np.nanmean(kp[:, 0]), np.nanmean(kp[:, 1])
            # which gt does this pred match? (max IoU)
            tag = "?"
            best_iou = 0.3
            for tr, gbx in gboxes.items():
                ix0, iy0 = max(b[0], gbx[0]), max(b[1], gbx[1])
                ix1, iy1 = min(b[2], gbx[2]), min(b[3], gbx[3])
                iw, ih = max(0, ix1 - ix0), max(0, iy1 - iy0)
                inter = iw * ih
                ua = (b[2]-b[0])*(b[3]-b[1]) + (gbx[2]-gbx[0])*(gbx[3]-gbx[1]) - inter
                iou = inter / ua if ua > 0 else 0
                if iou > best_iou:
                    best_iou = iou; tag = GT_TAGS[tr]
            cv2.putText(frame, f"id{iv}|gt{tag}", (int(cx) - 20, int(cy)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2, cv2.LINE_AA)

        # header
        is_sw = fi in sw_set
        inter_mm = np.nan
        if p3d is not None and fi < p3d.shape[0]:
            cents = [np.nanmean(p3d[fi, tr], axis=0) for tr in gboxes]
            cents = [c for c in cents if np.isfinite(c).all()]
            if len(cents) >= 2:
                inter_mm = min(np.linalg.norm(cents[i]-cents[j])
                               for i in range(len(cents)) for j in range(i+1, len(cents)))
        bar = (0, 0, 200) if is_sw else (40, 40, 40)
        cv2.rectangle(frame, (0, 0), (W, 26), bar, -1)
        msg = f"{a.session} {a.cam}  f{fi}"
        if np.isfinite(inter_mm):
            msg += f"  inter-animal={inter_mm:.0f}mm"
        if is_sw:
            msg += "   *** ID SWITCH ***"
        cv2.putText(frame, msg, (6, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        vw.write(frame)
    cap.release(); vw.release()
    # re-encode to h264 for broad playback
    print("wrote", a.out)


if __name__ == "__main__":
    main()
