#!/usr/bin/env python3
"""Mouse-Dyad-10M: what Fig 3c's five-camera value measures (read-only QC).

Fig 3c reports the distance from LUC3D's k-camera solve to the dataset's reference 3D
(`<ts>_points3d_translated_rotated_metric.h5`). At k = 5 it is a median 1.21 mm. This asks
whether that gap reflects hand corrections to keypoint positions or the way the reference
was triangulated.

Per session it compares the reference with an independent per-frame DLT of the same
per-camera 2D tracks and calibration, made outside LUC3D (`BMimica/scratch/`, see
DATA-LOCATIONS.md). The DLT is mapped onto the reference by one similarity transform per
session (Umeyama). It reports:

  gap_med_mm / gap_p95_mm   |DLT - reference| over every keypoint both have
  pts_gt10mm                share of keypoints more than 10 mm apart
  rough_*_mm                median |second difference over frames| (mm/frame^2), i.e. jitter
  ref_nan / dlt_nan         share of keypoints missing (DLT: seen in < 2 cameras)
  luc3d_k5_p50_mm           LUC3D's own five-camera value for the session (out/fig4_by_views_q.json)

Result (2026-11-16, 46 sessions compared: 48 scratch files after excluding 20250904_183921, 2 unreadable): in the 39
sessions where the scratch DLT agrees with LUC3D (gap < 3 mm), the DLT sits a median 1.25 mm
from the reference against LUC3D's 1.21 mm. The reference is a median 23x smoother frame to
frame (10x to 42x) and has no missing keypoints. The gap is spread over all frames (about 1.5%
of keypoints are > 10 mm apart). So the reference is a temporally smoothed, gap-filled
triangulation of the same detections, and the five-camera value measures single-frame versus
smoothed triangulation, not hand corrections. In 7 sessions the scratch DLT is 8 to 331 mm off
the reference while LUC3D's value is normal (1.08 to 1.21 mm); that is a fault of the scratch
pass (not a track-order swap, checked) and is not diagnosed further.

Writes data/qc/reference_vs_dlt.csv.

    cd figs && $PY qc/reference_vs_dlt.py        # BMIMICA=/mnt/vast/eric/BMimica to override
"""
import csv, glob, json, os, sys
from multiprocessing import Pool
from pathlib import Path

import h5py
import numpy as np

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from src.qc_exclude import excluded  # noqa: E402

BMIMICA = os.environ.get("BMIMICA", "/root/vast/eric/BMimica")
OUT = HERE / "data" / "qc" / "reference_vs_dlt.csv"
AGREE_MM = 3.0


def sim_align(src, dst):
    m = ~np.isnan(src).any(-1) & ~np.isnan(dst).any(-1)
    a, b = src[m], dst[m]
    ma, mb = a.mean(0), b.mean(0)
    ac, bc = a - ma, b - mb
    U, S, Vt = np.linalg.svd(ac.T @ bc)
    D = np.eye(3)
    D[2, 2] = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ D @ U.T
    s = (S * np.diag(D)).sum() / (ac ** 2).sum()
    return lambda x: s * (x - ma) @ R.T + mb


def rough(x):
    return float(np.nanmedian(np.linalg.norm(np.diff(x, 2, axis=0), axis=-1)))


def run(path):
    sess = os.path.basename(path).replace("_triangulated.h5", "")
    try:
        ref = h5py.File(glob.glob(f"{BMIMICA}/{sess}/*_points3d_translated_rotated_metric.h5")[0], "r")["tracks"][:]
        dlt = h5py.File(path, "r")["pts_3d"][:]
    except Exception as ex:  # one scratch file is unreadable
        return dict(session=sess, error=str(ex))
    n = min(len(ref), len(dlt))
    ref, dlt = ref[:n], dlt[:n]
    A = sim_align(dlt.reshape(-1, 3), ref.reshape(-1, 3))(dlt.reshape(-1, 3)).reshape(dlt.shape)
    e = np.linalg.norm(A - ref, axis=-1) * 1000
    e = e[~np.isnan(e)]
    return dict(session=sess, n_frames=n,
                ref_nan=float(np.isnan(ref).any(-1).mean()), dlt_nan=float(np.isnan(dlt).any(-1).mean()),
                gap_med_mm=float(np.median(e)), gap_p95_mm=float(np.percentile(e, 95)),
                pts_gt10mm=float((e > 10).mean()),
                rough_ref_mm=rough(ref) * 1000, rough_dlt_mm=rough(A) * 1000)


def main():
    k5 = {r["session"]: r["by_k"]["5"]["dlt"]["p50"]
          for r in json.load(open(HERE / "out" / "fig4_by_views_q.json"))["per_session"]}
    paths = sorted(glob.glob(f"{BMIMICA}/scratch/triangulation_results/*_triangulated.h5"))
    paths = [p for p in paths if not excluded(os.path.basename(p).replace("_triangulated.h5", ""))]
    with Pool(8) as pool:
        rows = pool.map(run, paths)
    for r in rows:
        r["luc3d_k5_p50_mm"] = k5.get(r["session"])
        r["agrees_with_luc3d"] = ("gap_med_mm" in r) and r["gap_med_mm"] < AGREE_MM
    keys = ["session", "n_frames", "ref_nan", "dlt_nan", "gap_med_mm", "gap_p95_mm", "pts_gt10mm",
            "rough_ref_mm", "rough_dlt_mm", "luc3d_k5_p50_mm", "agrees_with_luc3d", "error"]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, keys)
        w.writeheader()
        w.writerows(rows)
    ok = [r for r in rows if r["agrees_with_luc3d"]]
    med = lambda k: float(np.median([r[k] for r in ok]))
    ratio = [r["rough_dlt_mm"] / r["rough_ref_mm"] for r in ok]
    print(f"{len(rows)} sessions, {len(ok)} agree with LUC3D (gap < {AGREE_MM} mm)")
    print(f"gap median {med('gap_med_mm'):.3f} mm, LUC3D k=5 median {med('luc3d_k5_p50_mm'):.3f} mm")
    print(f"reference smoother by a median {np.median(ratio):.1f}x ({min(ratio):.1f}-{max(ratio):.1f}x)")
    print(f"reference complete in every session: {all(r.get('ref_nan') == 0 for r in rows if 'error' not in r)}")


if __name__ == "__main__":
    main()
