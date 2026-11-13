#!/usr/bin/env python
"""Score the Bartul UDMT runs vs the points3d-reprojection GT and append rows to
bartul_eval_per_camera.csv (tracker label = 'UDMT (off-pool, own det.)').

Reuses score_udmt.load_udmt_boxes + evaluate.load_gt + the same motmetrics setup;
only the GT path differs (Bartul per-serial GT instead of slap2m proofread)."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

REPO = Path("/root/vast/eric/luc3d-bench")
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "scripts" / "udmt"))
import evaluate as ev
import motmetrics as mm
from score_udmt import load_udmt_boxes, IOU_THRESH

OUT = REPO / "outputs/bartul_smoke"
SESSION = "20250827_141755"
SERIALS = ["21241563","21369048","21372315","21372316","22085397"]
PROJ_ROOT = OUT / "UDMT_runs_ep20"
GT_DIR = OUT / "gt"
WINDOW = 9000


def score_cam(cam):
    projs = sorted(PROJ_ROOT.glob(f"{SESSION}_{cam}-*"))
    if not projs:
        return None
    proj = projs[0]
    vids = list((proj / "tracking-results").glob("*"))
    if not vids:
        return None
    vid = vids[0].name
    try:
        udmt = load_udmt_boxes(proj, vid)   # (A, F, 4) original coords
    except Exception as e:
        print(f"  {cam}: no UDMT output ({e})")
        return None
    gt_tracks, gt_occ = ev.load_gt(GT_DIR / cam / "proofread.analysis.h5")
    n_frames = min(gt_tracks.shape[0], udmt.shape[1], WINDOW)
    acc = mm.MOTAccumulator(auto_id=True)
    # UDMT is a POINT tracker (fixed ~80px box); box-IoU@0.5 vs tall keypoint GT boxes is
    # structurally unfair. Score fairly by CENTROID: a UDMT track matches a GT animal if its
    # box-center falls inside the GT keypoint bbox (dist = center-to-GT-center; invalid outside).
    for fi in range(n_frames):
        gtc, gtbox, gti = [], [], []
        for t in range(gt_tracks.shape[1]):
            if not gt_occ[fi, t]:
                continue
            b = ev.bbox_from_kpts(gt_tracks[fi, t])
            if b is None:
                continue
            gtc.append(np.array([(b[0]+b[2])/2, (b[1]+b[3])/2])); gtbox.append(b); gti.append(int(t))
        prc, pri = [], []
        for a in range(udmt.shape[0]):
            box = udmt[a, fi]
            if np.all(np.isfinite(box)) and box[2] > box[0] and box[3] > box[1]:
                prc.append(np.array([(box[0]+box[2])/2, (box[1]+box[3])/2])); pri.append(a)
        if gtc and prc:
            D = np.full((len(gtc), len(prc)), np.nan)
            for i, (gc, gb) in enumerate(zip(gtc, gtbox)):
                for j, pc in enumerate(prc):
                    if gb[0] <= pc[0] <= gb[2] and gb[1] <= pc[1] <= gb[3]:
                        D[i, j] = np.linalg.norm(gc - pc)
        else:
            D = np.empty((len(gtc), len(prc)))
        acc.update(gti, pri, D)
    mh = mm.metrics.create()
    s = mh.compute(acc, metrics=["idf1","idp","idr","mota","num_switches","recall","num_frames"], name="u")
    return {"tracker": "UDMT (off-pool, centroid-scored)", "camera": cam,
            "idf1": s["idf1"]["u"], "idp": s["idp"]["u"], "idr": s["idr"]["u"],
            "mota": s["mota"]["u"], "num_switches": s["num_switches"]["u"],
            "recall": s["recall"]["u"], "num_frames": s["num_frames"]["u"]}


def main():
    rows = [r for r in (score_cam(c) for c in SERIALS) if r]
    if not rows:
        print("no UDMT results yet"); return
    for r in rows:
        print(f"  {r['camera']}: IDF1={r['idf1']:.3f} sw={int(r['num_switches'])}")
    csv = OUT / "bartul_eval_per_camera.csv"
    df = pd.read_csv(csv, dtype={"camera": str})
    df = df[df["tracker"] != "UDMT (off-pool, centroid-scored)"]   # drop stale UDMT rows
    df = pd.concat([df, pd.DataFrame(rows)], ignore_index=True)
    df.to_csv(csv, index=False)
    print(f"appended {len(rows)} UDMT rows; IDF1(mean)={np.mean([r['idf1'] for r in rows]):.3f} "
          f"sw(sum)={int(sum(r['num_switches'] for r in rows))}")


if __name__ == "__main__":
    main()
