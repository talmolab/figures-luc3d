#!/usr/bin/env python3
"""J3 (LUC3D-JS plan, 2026-10-20): does the Fig 2 reference alignment change when its
triangulation is LUC3D v1.0.5's JavaScript instead of fig2_measure.py's NumPy DLT?

fig2_solvers_export.py aligns each session's proofread 3D (P-frame) into the
calibration frame: undistort a frame sample of the raw 2D (cv2), triangulate it
(NumPy `triangulate_batch`), RANSAC-fit a similarity to the proofread 3D. The scale
gives `mm_per_unit`, which turns 2C / E / F errors into millimetres. This script runs
that recipe twice per session on identical inputs -- once as exported, once with the
undistort + triangulate steps done by LUC3D (figs/lucid_geom.mjs: Camera.undistortPoint
+ triangulatePointDLT) -- and reports how far the triangulated points, the similarity
(s, R, t) and mm_per_unit move. It also checks the NumPy arm reproduces the mm_per_unit
the current export stored, so the comparison is against what the figures used.

    POSE_DIR=/root/vast/eric/lucid-main-md56/pose \
      /root/vast/eric/luc3d-bench/lp3d_env/bin/python figs/j3_alignment_check.py

Writes figs/out/j3_alignment_check.json.
"""
import glob
import json
import os
import sys

import h5py
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.environ.get("LUC3D_BENCH_SCRIPTS", "/root/vast/eric/luc3d-bench/scripts/bartul"))
from build_gt_reproj import load_calibration, load_sleap_2d, undistort, ransac_align, SERIALS  # noqa: E402
from fig2_measure import triangulate_batch, ROOT  # noqa: E402  (the arm being replaced)
import lucid_geom  # noqa: E402

EXPORT = os.path.join(HERE, "out", "fig4_input_md56.json")


def cameras_of(cams_all):
    # Exactly fig2_solvers_export.py's camera block (the arguments of LUC3D's Camera).
    return [dict(name=c, matrix=[[float(v) for v in row] for row in cams_all[c]["K"]],
                 distortions=[float(v) for v in np.asarray(cams_all[c]["dist"]).ravel()[:5]],
                 rvec=[float(v) for v in np.asarray(cams_all[c]["rvec"]).ravel()],
                 tvec=[float(v) for v in np.asarray(cams_all[c]["t"]).ravel()],
                 size=[1280, 1024]) for c in SERIALS]


def main():
    exp = json.load(open(EXPORT))
    stored = {b["session"]: b["mm_per_unit"] for b in exp["blocks"]}
    rows, stamp = [], None
    # J3_LIMIT=<n>: first n sessions only (smoke test); the deposit says how many ran.
    for sid in exp["sessions"][:int(os.environ.get("J3_LIMIT", 0)) or None]:
        sd = f"{ROOT}/{sid}"
        cams_all = load_calibration(glob.glob(f"{sd}/calibration/*_calibration.toml")[0])
        with h5py.File(glob.glob(f"{sd}/*points3d*.h5")[0]) as f:
            Xp = f["tracks"][:]
        raw = {c: load_sleap_2d(glob.glob(f"{sd}/{c}/*.analysis.h5")[0])[:, :2] for c in SERIALS}
        F = min(Xp.shape[0], min(v.shape[0] for v in raw.values()))
        A, N, C = Xp.shape[1], Xp.shape[2], len(SERIALS)
        Ps = np.stack([cams_all[c]["P"] for c in SERIALS], 0)
        samp = np.arange(0, F, max(1, F // 4000))          # the export's sample
        S = len(samp)

        # NumPy arm, verbatim from fig2_solvers_export.py L157-166.
        und_s = np.stack([undistort(raw[c][samp], cams_all[c]["K"], cams_all[c]["dist"])
                          for c in SERIALS], 0)
        flat = und_s.reshape(C, -1, 2)
        Xn = triangulate_batch(flat, Ps, ~np.isnan(flat).any(-1)).reshape(S, A, N, 3)
        # LUC3D arm: the same RAW 2D, undistorted and triangulated by v1.0.5.
        raw_s = np.stack([raw[c][samp] for c in SERIALS], 0).reshape(C, -1, 2)
        Xj, stamp = lucid_geom.triangulate(cameras_of(cams_all), raw_s)
        Xj = Xj.reshape(S, A, N, 3)

        both_n = (~np.isnan(Xn).any(-1)) & (~np.isnan(Xp[samp]).any(-1))
        both_j = (~np.isnan(Xj).any(-1)) & (~np.isnan(Xp[samp]).any(-1))
        (sn, Rn, tn), _, _ = ransac_align(Xp[samp][both_n], Xn[both_n], thresh=8.0)
        (sj, Rj, tj), _, _ = ransac_align(Xp[samp][both_j], Xj[both_j], thresh=8.0)
        common = both_n & both_j
        mm_n, mm_j = 1000.0 / sn, 1000.0 / sj
        r = dict(session=sid, points=int(common.sum()),
                 same_mask=bool((both_n == both_j).all()),
                 max_abs_point_diff=float(np.abs(Xn[common] - Xj[common]).max()),
                 p50_point_dist=float(np.median(np.linalg.norm(Xn[common] - Xj[common], axis=-1))),
                 p99_point_dist=float(np.percentile(np.linalg.norm(Xn[common] - Xj[common], axis=-1), 99)),
                 p99_point_dist_mm=float(np.percentile(np.linalg.norm(Xn[common] - Xj[common], axis=-1), 99) * 1000.0 / sn),
                 rel_scale_diff=float(abs(sj - sn) / sn),
                 max_abs_R_diff=float(np.abs(Rj - Rn).max()),
                 max_abs_t_diff=float(np.abs(tj - tn).max()),
                 mm_per_unit_numpy=mm_n, mm_per_unit_js=mm_j,
                 mm_per_unit_stored=stored[sid],
                 numpy_reproduces_stored=bool(mm_n == stored[sid]))
        rows.append(r)
        print(f"{sid}: pts {r['points']} same_mask {r['same_mask']} "
              f"dX max {r['max_abs_point_diff']:.3g} p50 {r['p50_point_dist']:.3g} p99 {r['p99_point_dist']:.3g} ({r['p99_point_dist_mm']:.3g} mm)  ds/s {r['rel_scale_diff']:.3g}  "
              f"dR {r['max_abs_R_diff']:.3g}  dt {r['max_abs_t_diff']:.3g}  "
              f"stored==numpy {r['numpy_reproduces_stored']}", flush=True)

    summary = {k: max(r[k] for r in rows) for k in
               ("max_abs_point_diff", "p50_point_dist", "p99_point_dist", "p99_point_dist_mm", "rel_scale_diff", "max_abs_R_diff", "max_abs_t_diff")}
    summary["all_same_mask"] = all(r["same_mask"] for r in rows)
    summary["numpy_reproduces_stored_all"] = all(r["numpy_reproduces_stored"] for r in rows)
    out = dict(generated_by="figs/j3_alignment_check.py", **stamp, export=os.path.basename(EXPORT),
               sessions=len(rows), summary=summary, per_session=rows)
    with open(os.path.join(HERE, "out", "j3_alignment_check.json"), "w") as f:
        json.dump(out, f, indent=1)
    print("SUMMARY", json.dumps(summary))


if __name__ == "__main__":
    main()
