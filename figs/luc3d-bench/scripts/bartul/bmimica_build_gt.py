#!/usr/bin/env python
"""Stage 2: points3d-reprojection GT for all 50 runnable BMimica sessions.

Per session: recover the rigid+scale transform (P-frame -> calibration frame)
via RANSAC-Procrustes vs a re-triangulation of the raw SLEAP 2D, reproject the
proofread 3D into each of the 5 cameras, write per-serial GT analysis.h5.

Output: outputs/bmimica/gt/{session}/{serial}/proofread.analysis.h5
        (parent dir == serial, so evaluate.py reads the camera name from it)
Reuses the validated helpers in build_gt_reproj.py. Resumable + parallel.
"""
import os, glob, sys
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np, h5py, cv2

sys.path.insert(0, "/root/vast/eric/luc3d-bench/scripts/bartul")
from build_gt_reproj import (load_calibration, load_sleap_2d, undistort,
                             triangulate_all, procrustes_scale, ransac_align, SERIALS)

ROOT = "/root/vast/eric/BMimica"
OUT = "/root/vast/eric/luc3d-bench/outputs/bmimica/gt"


def runnable_sessions():
    out = []
    for s in sorted(glob.glob(ROOT + "/*")):
        sid = os.path.basename(s)
        if os.path.isdir(s) and sid[0].isdigit() and \
           glob.glob(s+"/calibration/*_calibration.toml") and glob.glob(s+"/*points3d*.h5") and \
           sum(1 for c in SERIALS if glob.glob(f"{s}/{c}/*.slp")) == 5:
            out.append(sid)
    return out


def build_one(sid):
    sd = f"{ROOT}/{sid}"
    outdir = f"{OUT}/{sid}"
    done = [f"{outdir}/{c}/proofread.analysis.h5" for c in SERIALS]
    if all(os.path.exists(p) and os.path.getsize(p) > 1000 for p in done):
        return (sid, "skip", None)
    try:
        cams = load_calibration(glob.glob(f"{sd}/calibration/*_calibration.toml")[0])
        if any(c not in cams for c in SERIALS):
            return (sid, "FAIL", "calib missing a serial")
        with h5py.File(glob.glob(f"{sd}/*points3d*.h5")[0]) as f:
            Xp = f["tracks"][:]  # (F,2,15,3)
        F3 = Xp.shape[0]
        # cap to first 2 tracks: some cams over-fragment (e.g. 12 tracks) which breaks
        # the cross-camera stack; only 2 animals matter and RANSAC handles any noise.
        sleap = {c: load_sleap_2d(glob.glob(f"{sd}/{c}/*.analysis.h5")[0])[:, :2] for c in SERIALS}
        F = min(F3, min(v.shape[0] for v in sleap.values()))
        # transform recovery on sampled frames
        samp = np.arange(0, F, max(1, F // 4000))
        und = np.stack([undistort(sleap[c][samp], cams[c]["K"], cams[c]["dist"]) for c in SERIALS], 0)
        Ps = np.stack([cams[c]["P"] for c in SERIALS], 0)
        Xc = triangulate_all(und, Ps)
        Xp_s = Xp[samp]
        both = (~np.isnan(Xc).any(-1)) & (~np.isnan(Xp_s).any(-1))
        (s, R, t), err, inl = ransac_align(Xp_s[both], Xc[both], thresh=8.0)
        # apply + reproject
        flat = Xp[:F].reshape(-1, 3); good = ~np.isnan(flat).any(1)
        Xcal = np.full_like(flat, np.nan); Xcal[good] = (s * (R @ flat[good].T).T + t)
        Xcal = Xcal.reshape(F, 2, 15, 3)
        residmed = float(np.median(err[inl])) if inl.any() else float("nan")
        for c in SERIALS:
            cam = cams[c]; f3 = Xcal.reshape(-1, 3); g = ~np.isnan(f3).any(1)
            proj = np.full((f3.shape[0], 2), np.nan)
            if g.any():
                pp, _ = cv2.projectPoints(f3[g], cam["rvec"], cam["t"], cam["K"], cam["dist"])
                proj[g] = pp.reshape(-1, 2)
            gt2d = proj.reshape(F, 2, 15, 2)
            tracks = np.transpose(gt2d, (1, 3, 2, 0)).astype(np.float64)  # (2,2,15,F)
            occ = (~np.isnan(gt2d).any(-1)).any(-1).astype(np.uint8)      # (F,2)
            cdir = f"{outdir}/{c}"; os.makedirs(cdir, exist_ok=True)
            with h5py.File(f"{cdir}/proofread.analysis.h5", "w") as h:
                h.create_dataset("tracks", data=tracks, compression="gzip", compression_opts=1)
                h.create_dataset("track_occupancy", data=occ)
        return (sid, "ok", f"scale={s:.1f} inl={inl.mean():.2f} resid_mm={residmed:.2f} F={F}")
    except Exception as e:
        return (sid, "ERR", str(e)[:200])


def main():
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument("--workers", type=int, default=6); a = ap.parse_args()
    sids = runnable_sessions()
    print(f"GT build: {len(sids)} sessions, {a.workers} workers", flush=True)
    ok = 0
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for f in as_completed([ex.submit(build_one, s) for s in sids]):
            sid, st, info = f.result()
            if st in ("ok", "skip"): ok += 1
            print(f"  {sid}: {st}  {info or ''}", flush=True)
    print(f"DONE GT {ok}/{len(sids)}", flush=True)


if __name__ == "__main__":
    main()
