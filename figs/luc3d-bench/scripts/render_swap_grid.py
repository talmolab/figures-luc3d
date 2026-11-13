#!/usr/bin/env python3
"""Multi-camera GRID video of an ID swap, with the swap made unmistakable.

All 6 cameras shown in a 2x3 grid, time-synced. Identity colors are GLOBAL: id0 is
the same color in every camera, so a cross-view swap shows as the colors flipping in
sync. Each GT animal has a fixed CORRECT identity (session-wide modal). Boxes are
color-coded by correctness:
  GREEN box  -> track matched to its GT animal AND wearing the CORRECT id
  RED box    -> track matched to a GT animal but wearing the WRONG id (a swap)
  thin id-color box -> unmatched / unknown
Each instance is drawn as a skeleton: keypoints are big per-node colored dots and the
skeleton edges connect them (edges drawn in the identity color so the swap is obvious).

To see the swap *happen*, give a lead-in: set --start well before --swap-start so the
clip opens with correct (green) tracking, then transitions to the red swap.

Usage:
  PYTHONPATH=. .venv/bin/python scripts/render_swap_grid.py \
      --session 10072022163346 --start 9000 --end 11000 \
      --results-dir outputs/PAF_3d_kalman/_liezlsweep/G_keeptrack_3d6 \
      --pred-h5-dir outputs/keeptrack_h5s \
      --out outputs/PAF_3d_kalman/_diag/grids/ep1.mp4 [--focal-track 0]
"""
import argparse
import json
from pathlib import Path

import cv2
import h5py
import numpy as np
import pandas as pd

from scripts.evaluate import load_gt, bbox_from_kpts, luc3d_assignments_for_cam

REPO = Path(__file__).resolve().parents[1]
MASTER = REPO / "outputs" / "detections_only_master_sheet.tsv"
SLAP2M = Path("/root/talmolab-smb/eric/slap_2m")
CAMS = ["back", "backL", "mid", "midL", "top", "topL"]
PAL = [(60, 60, 230), (230, 160, 40), (60, 200, 60), (200, 60, 200),
       (40, 220, 220), (230, 230, 230), (120, 80, 200)]
PAL_NAME = ["RED", "ORANGE", "GREEN", "MAGENTA", "CYAN", "WHITE", "PURPLE"]
GT_TAGS = "ABCDEFGH"
CELL_W = 470

# Mouse skeleton (15 nodes). Index order matches the prediction/GT h5 `tracks` arrays:
# 0 Nose 1 Ear_R 2 Ear_L 3 TTI 4 TailTip 5 Head 6 Trunk 7 Tail_0 8 Tail_1
# 9 Tail_2 10 Shoulder_left 11 Shoulder_right 12 Haunch_left 13 Haunch_right 14 Neck
EDGES = [(3, 5), (3, 7), (3, 8), (3, 9), (3, 12), (3, 13), (3, 6),
         (5, 0), (5, 14), (5, 10), (5, 11), (5, 1), (5, 2), (3, 4)]
# distinct vivid colors per node so the pose reads clearly (BGR)
NODE_PAL = [
    (0, 0, 255), (0, 128, 255), (0, 200, 255), (0, 255, 255), (0, 255, 128),
    (0, 255, 0), (128, 255, 0), (255, 255, 0), (255, 200, 0), (255, 128, 0),
    (255, 0, 0), (255, 0, 128), (255, 0, 255), (180, 0, 255), (255, 255, 255),
]
GREEN = (40, 200, 40)
RED = (0, 0, 255)


def iou(a, b):
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1]); ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0, ix1 - ix0), max(0, iy1 - iy0); inter = iw * ih
    ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def draw_instance(frame, kp, idcol):
    """Draw one instance as a skeleton: edges in identity color, big per-node dots.

    Node radius scales with the animal's pixel extent so the keypoints stay prominent at
    any zoom. Each node gets a distinct color so the pose reads clearly.
    """
    fin = np.isfinite(kp[:, 0]) & np.isfinite(kp[:, 1])
    if fin.sum() < 2:
        return
    pts = kp[fin]
    span = max(np.ptp(pts[:, 0]), np.ptp(pts[:, 1]))
    node_r = int(np.clip(span * 0.06, 4, 14))
    lw = max(2, node_r // 3)
    for a, b in EDGES:
        if a < len(kp) and b < len(kp) and fin[a] and fin[b]:
            pa = (int(kp[a, 0]), int(kp[a, 1])); pb = (int(kp[b, 0]), int(kp[b, 1]))
            cv2.line(frame, pa, pb, idcol, lw, cv2.LINE_AA)
    for ni in range(len(kp)):
        if fin[ni]:
            p = (int(kp[ni, 0]), int(kp[ni, 1]))
            cv2.circle(frame, p, node_r + 1, (20, 20, 20), -1, cv2.LINE_AA)   # dark halo
            cv2.circle(frame, p, node_r, NODE_PAL[ni % len(NODE_PAL)], -1, cv2.LINE_AA)


def correct_ids(per_cam, n_min, stride=20):
    """Session-wide modal id per GT track, sampled over the whole session."""
    votes = {}
    for cam, (raw, gt, occ, ids, n, na, nt) in per_cam.items():
        for fi in range(0, n, stride):
            gb = {t: bbox_from_kpts(gt[fi, t]) for t in range(nt) if occ[fi, t]}
            gb = {t: b for t, b in gb.items() if b is not None}
            pb = []
            for s in range(na):
                b = bbox_from_kpts(raw[fi, s])
                if b is None:
                    continue
                iv = int(ids[fi, s])
                if iv >= 0:
                    pb.append((b, iv))
            pairs = sorted(((iou(gb[t], pbx), t, iv) for t in gb for pbx, iv in pb), reverse=True)
            ut = set()
            for io, t, iv in pairs:
                if io < 0.5 or t in ut:
                    continue
                votes.setdefault(t, {}).setdefault(iv, 0); votes[t][iv] += 1; ut.add(t)
    return {t: max(v, key=v.get) for t, v in votes.items()}


def build_session(session, results_dir, pred_h5_dir, show_truth=False):
    """Load all per-camera predictions, GT, video captures, and the modal correct ids."""
    ms = pd.read_csv(MASTER, sep="\t"); ms["session"] = ms["session"].astype(str)
    row = ms[ms["session"] == session].iloc[0]
    h5idx = int(np.where(ms["session"].values == session)[0][0])
    jd = json.load(open(f"{results_dir}/{session}.json"))

    per_cam = {}
    caps = {}
    for cam in CAMS:
        gtp = row[f"{cam}_proofread_h5"]
        if not isinstance(gtp, str) or not gtp:
            continue
        raw = h5py.File(f"{pred_h5_dir}/{cam}_predictions.h5", "r")["tracks"][h5idx]
        gt, occ = load_gt(gtp)
        n = min(raw.shape[0], gt.shape[0]); na = raw.shape[1]; nt = gt.shape[1]
        ids = luc3d_assignments_for_cam(jd, cam, n, na)
        per_cam[cam] = (raw, gt, occ, ids, n, na, nt)
        caps[cam] = cv2.VideoCapture(str(SLAP2M / row[f"{cam}_video"]))

    reproj = {}
    if show_truth:
        for cam in per_cam:
            rp = row[f"{cam}_reproj_h5"]
            rp = rp if str(rp).startswith("/") else str(SLAP2M / str(rp))
            if Path(rp).exists():
                with h5py.File(rp, "r") as f:
                    reproj[cam] = np.transpose(f["tracks"][:], (3, 0, 2, 1))  # (frames,tracks,nodes,2)

    correct = correct_ids(per_cam, 0)
    cam0 = next(iter(per_cam))
    cw = int(caps[cam0].get(cv2.CAP_PROP_FRAME_WIDTH)); ch = int(caps[cam0].get(cv2.CAP_PROP_FRAME_HEIGHT))
    cell_h = int(CELL_W * ch / cw)
    return dict(session=session, per_cam=per_cam, caps=caps, reproj=reproj,
                correct=correct, cw=cw, ch=ch, cell_h=cell_h)


def canvas_dims(cell_h):
    cols, rows = 3, 2
    HEAD = 70
    return cols * CELL_W, HEAD + rows * (cell_h + 22), cols, rows, HEAD


def open_writer(out, fps, cell_h):
    canvas_w, canvas_h, _, _, _ = canvas_dims(cell_h)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    return cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*"mp4v"), fps, (canvas_w, canvas_h))


def title_card(vw, cell_h, lines, n_frames=45):
    """Write a solid title card (used to separate concatenated examples)."""
    canvas_w, canvas_h, _, _, _ = canvas_dims(cell_h)
    card = np.zeros((canvas_h, canvas_w, 3), np.uint8)
    card[:] = (25, 25, 25)
    y = canvas_h // 2 - 18 * len(lines)
    for i, (txt, scale, col) in enumerate(lines):
        (tw, _), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)
        cv2.putText(card, txt, ((canvas_w - tw) // 2, y), cv2.FONT_HERSHEY_SIMPLEX,
                    scale, col, 2, cv2.LINE_AA)
        y += int(40 * scale) + 18
    for _ in range(n_frames):
        vw.write(card)


def compute_rois(sess, start, end):
    """Per-camera ROI = union of GT boxes over [start,end], padded — to zoom in."""
    rois = {}
    for cam, (raw, gt, occ, ids, n, na, nt) in sess["per_cam"].items():
        xs0 = ys0 = 1e9; xs1 = ys1 = -1e9
        for fi in range(start, min(end, n)):
            for t in range(nt):
                if occ[fi, t]:
                    b = bbox_from_kpts(gt[fi, t])
                    if b is not None:
                        xs0 = min(xs0, b[0]); ys0 = min(ys0, b[1]); xs1 = max(xs1, b[2]); ys1 = max(ys1, b[3])
        cw0 = int(sess["caps"][cam].get(cv2.CAP_PROP_FRAME_WIDTH))
        ch0 = int(sess["caps"][cam].get(cv2.CAP_PROP_FRAME_HEIGHT))
        if xs1 < 0:
            rois[cam] = (0, 0, cw0, ch0); continue
        padx = max(60, (xs1 - xs0) * 0.5); pady = max(60, (ys1 - ys0) * 0.5)
        rois[cam] = (max(0, int(xs0 - padx)), max(0, int(ys0 - pady)),
                     min(cw0, int(xs1 + padx)), min(ch0, int(ys1 + pady)))
    return rois


def render_clip(vw, sess, start, end, focal=None, swap_start=None, swap_end=None,
                label="", show_truth=False):
    """Append one clip [start,end) of session `sess` to writer `vw`."""
    per_cam = sess["per_cam"]; caps = sess["caps"]; correct = sess["correct"]
    reproj = sess["reproj"]; cell_h = sess["cell_h"]
    canvas_w, canvas_h, cols, rows, HEAD = canvas_dims(cell_h)
    rois = compute_rois(sess, start, end)

    for cam in per_cam:
        caps[cam].set(cv2.CAP_PROP_POS_FRAMES, start)

    foc = focal
    wrong_streak = 0
    for fi in range(start, end):
        canvas = np.zeros((canvas_h, canvas_w, 3), np.uint8)
        any_swap_tags = []
        foc_wrong_cams = 0; foc_det_cams = 0  # cross-view consensus on focal track
        for ci, cam in enumerate(CAMS):
            if cam not in per_cam:
                continue
            raw, gt, occ, ids, n, na, nt = per_cam[cam]
            ok, frame = caps[cam].read()
            if not ok or fi >= n:
                continue
            # brighten the dark IR footage so the mice are visible (CLAHE on luma)
            yuv = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
            yuv[:, :, 0] = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(yuv[:, :, 0])
            frame = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
            # GT boxes
            gb = {}
            for t in range(nt):
                if occ[fi, t]:
                    b = bbox_from_kpts(gt[fi, t])
                    if b is not None:
                        gb[t] = b
            # predicted
            preds = []
            for s in range(na):
                b = bbox_from_kpts(raw[fi, s])
                if b is None:
                    continue
                iv = int(ids[fi, s])
                if iv >= 0:
                    preds.append((b, iv, raw[fi, s]))
            # match pred->gt
            pairs = sorted(((iou(gb[t], b), t, k) for t in gb for k, (b, iv, kp) in enumerate(preds)),
                           reverse=True)
            assigned = {}  # pred k -> gt t
            ut = set()
            for io, t, k in pairs:
                if io < 0.5 or t in ut or k in assigned:
                    continue
                assigned[k] = t; ut.add(t)
            # draw GT boxes (thin grey)
            for t, b in gb.items():
                cv2.rectangle(frame, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (150, 150, 150), 1)
            # draw preds: skeleton + correctness-colored box
            for k, (b, iv, kp) in enumerate(preds):
                col = PAL[iv % len(PAL)]
                draw_instance(frame, kp, col)
                t = assigned.get(k)
                matched_correct = (t is not None and t in correct and iv == correct[t])
                wrong = (t is not None and t in correct and iv != correct[t])
                if foc is not None:
                    # spotlight mode: only flag the focal track as wrong (clean story)
                    if t == foc and t in correct:
                        foc_det_cams += 1
                        if iv != correct[t]:
                            foc_wrong_cams += 1
                    wrong = wrong and (t == foc)
                if wrong:
                    bc, thick = RED, 5
                elif matched_correct:
                    bc, thick = GREEN, 3
                else:
                    bc, thick = col, 1
                cv2.rectangle(frame, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), bc, thick)
                if wrong:
                    cx, cy = int((b[0]+b[2])/2), int((b[1]+b[3])/2)
                    rad = int(max(b[2]-b[0], b[3]-b[1]) * 0.9)
                    cv2.circle(frame, (cx, cy), rad, RED, 3)
                    cv2.arrowedLine(frame, (cx, max(0, int(b[1]) - rad - 30)), (cx, int(b[1]) - 3),
                                    RED, 3, tipLength=0.4)
                lab = f"id{iv}"
                if t is not None:
                    lab += f"|gt{GT_TAGS[t]}"
                if wrong:
                    lab = f"WRONG {lab}->{correct[t]}"
                    any_swap_tags.append(f"{cam}:gt{GT_TAGS[t]}")
                cv2.putText(frame, lab, (int(b[0]), max(12, int(b[1]) - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, bc, 2, cv2.LINE_AA)
            # ground-truth reprojection of the focal animal (present even during dropout)
            if show_truth and foc is not None and cam in reproj and fi < reproj[cam].shape[0]:
                c = np.nanmean(reproj[cam][fi, foc], axis=0)
                if np.isfinite(c).all():
                    tx, ty = int(c[0]), int(c[1])
                    cv2.drawMarker(frame, (tx, ty), (255, 255, 255), cv2.MARKER_TILTED_CROSS, 26, 2)
                    cv2.putText(frame, f"TRUE gt{GT_TAGS[foc]}", (tx + 10, ty),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            rx0, ry0, rx1, ry1 = rois[cam]
            crop = frame[ry0:ry1, rx0:rx1]
            # letterbox crop into the fixed cell, preserving aspect
            ch_, cw_ = crop.shape[:2]
            sc = min(CELL_W / cw_, cell_h / ch_)
            rw, rh = max(1, int(cw_ * sc)), max(1, int(ch_ * sc))
            cell = np.zeros((cell_h, CELL_W, 3), np.uint8)
            res = cv2.resize(crop, (rw, rh))
            oy, ox = (cell_h - rh) // 2, (CELL_W - rw) // 2
            cell[oy:oy + rh, ox:ox + rw] = res
            r_, c_ = ci // cols, ci % cols
            y0 = HEAD + r_ * (cell_h + 22); x0 = c_ * CELL_W
            # cam label strip
            strip = (0, 0, 160) if any(tag.startswith(cam + ":") for tag in any_swap_tags) else (45, 45, 45)
            cv2.rectangle(canvas, (x0, y0), (x0 + CELL_W, y0 + 20), strip, -1)
            cv2.putText(canvas, cam, (x0 + 6, y0 + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
            canvas[y0 + 22:y0 + 22 + cell_h, x0:x0 + CELL_W] = cell

        # header
        focal_wrong_this_frame = foc_det_cams > 0 and foc_wrong_cams > foc_det_cams / 2.0
        if foc is not None:
            wrong_streak = wrong_streak + 1 if focal_wrong_this_frame else 0
            hdr_col = (0, 0, 200) if focal_wrong_this_frame else (30, 100, 30)
        else:
            hdr_col = (0, 0, 200) if any_swap_tags else (30, 30, 30)
        legend = "  ".join(f"gt{GT_TAGS[t]}=id{i}({PAL_NAME[i%len(PAL_NAME)]})" for t, i in correct.items())
        cv2.rectangle(canvas, (0, 0), (canvas_w, HEAD), hdr_col, -1)
        phase = ""
        if swap_start is not None:
            if fi < swap_start:
                phase = "  [BEFORE swap onset]"
            elif swap_end is not None and fi >= swap_end:
                phase = "  [AFTER - recovered]"
            else:
                phase = "  [DURING SWAP]"
        cap_txt = f"{label}" if label else ""
        cv2.putText(canvas, f"{sess['session']}  f{fi}{phase}   {cap_txt}", (8, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.66, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(canvas, f"correct: {legend}", (8, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 255, 200), 1, cv2.LINE_AA)
        if foc is not None:
            cid = correct.get(foc)
            if focal_wrong_this_frame:
                msg = (f"*** gt{GT_TAGS[foc]} MISLABELED (should be id{cid}/{PAL_NAME[cid%len(PAL_NAME)]}) "
                       f"-- {foc_wrong_cams}/{foc_det_cams} cams wrong, {wrong_streak} consecutive frames ***")
            else:
                msg = (f"gt{GT_TAGS[foc]} correct = id{cid}/{PAL_NAME[cid%len(PAL_NAME)]}"
                       f"  ({foc_wrong_cams}/{foc_det_cams} cams disagree)")
        else:
            msg = "*** ID SWAP: " + ", ".join(sorted(set(any_swap_tags))) + " wearing WRONG id ***" if any_swap_tags else "ids correct"
        cv2.putText(canvas, msg, (8, 64), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        vw.write(canvas)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", required=True)
    ap.add_argument("--start", type=int, required=True)
    ap.add_argument("--end", type=int, required=True)
    ap.add_argument("--results-dir", required=True)
    ap.add_argument("--pred-h5-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--focal-track", type=int, default=None,
                    help="GT track to spotlight; only its mislabels are ringed (clean story)")
    ap.add_argument("--swap-start", type=int, default=None, help="frame the focal mislabel begins (phase labels)")
    ap.add_argument("--swap-end", type=int, default=None, help="frame the focal mislabel ends")
    ap.add_argument("--show-truth", action="store_true",
                    help="overlay focal animal's GT-reprojected position (always present, even in dropout)")
    ap.add_argument("--label", default="", help="error-class caption shown in the header")
    a = ap.parse_args()

    sess = build_session(a.session, a.results_dir, a.pred_h5_dir, a.show_truth)
    print("correct id per GT track:",
          {GT_TAGS[t]: f"{i}({PAL_NAME[i%len(PAL_NAME)]})" for t, i in sess["correct"].items()})
    vw = open_writer(a.out, a.fps, sess["cell_h"])
    render_clip(vw, sess, a.start, a.end, focal=a.focal_track,
                swap_start=a.swap_start, swap_end=a.swap_end, label=a.label, show_truth=a.show_truth)
    for c in sess["caps"].values():
        c.release()
    vw.release()
    print("wrote", a.out)


if __name__ == "__main__":
    main()
