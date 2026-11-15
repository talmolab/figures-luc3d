#!/usr/bin/env python
"""Build per-camera 2D ground-truth for the Bartul session by reprojecting the
PROOFREAD 3D points (points3d_translated_rotated_metric.h5) into each camera.

The proofread 3D is in a *translated/rotated/metric* frame, NOT the raw
calibration frame, so we first recover the rigid+scale transform (P-frame ->
calibration-frame) via robust (RANSAC) Procrustes against a re-triangulation of
the raw SLEAP 2D. Identity swaps in the raw SLEAP tracks become RANSAC outliers.

Output: one {serial}.gt.analysis.h5 per camera with
    /tracks           (n_tracks=2, 2_xy, 15_nodes, n_frames)   distorted pixels
    /track_occupancy  (n_frames, 2_tracks)  uint8
matching the layout scripts/evaluate.py:load_gt expects.
"""
import os, glob, sys
import numpy as np
import h5py, cv2, toml
from scipy.spatial.transform import Rotation

SESSION_DIR = "/root/vast/eric/bartul-data/20250827_141755"
OUT_DIR = "/root/vast/eric/luc3d-bench/outputs/bartul_smoke/gt"
SERIALS = ["21241563", "21369048", "21372315", "21372316", "22085397"]


def load_calibration(toml_path):
    cfg = toml.load(toml_path)
    cams = {}
    for sec, v in cfg.items():
        if not sec.startswith("cam_"):
            continue
        serial = v["name"]
        K = np.array(v["matrix"], float)
        dist = np.array(v["distortions"], float)
        rvec = np.array(v["rotation"], float)
        tvec = np.array(v["translation"], float)
        R = Rotation.from_rotvec(rvec).as_matrix()
        P = K @ np.hstack([R, tvec.reshape(3, 1)])
        cams[serial] = dict(K=K, dist=dist, R=R, t=tvec, rvec=rvec, P=P)
    return cams


def load_sleap_2d(analysis_h5):
    with h5py.File(analysis_h5) as f:
        raw = f["tracks"][:]  # (n_tracks, 2, n_nodes, n_frames)
    return raw.transpose(3, 0, 2, 1)  # (F, T, N, 2)  distorted px


def undistort(pts, K, dist):
    shp = pts.shape
    flat = pts.reshape(-1, 2).copy()
    valid = ~np.isnan(flat).any(1)
    if valid.any():
        u = cv2.undistortPoints(flat[valid].reshape(-1, 1, 2), K, dist, P=K).reshape(-1, 2)
        flat[valid] = u
    return flat.reshape(shp)


def triangulate_all(pts_und, Ps, min_cams=2):
    """pts_und: (C,F,T,N,2) undistorted px; Ps: (C,3,4). -> (F,T,N,3)."""
    C, F, T, N, _ = pts_und.shape
    out = np.full((F, T, N, 3), np.nan)
    valid = ~np.isnan(pts_und).any(-1)  # (C,F,T,N)
    for f in range(F):
        for t in range(T):
            for n in range(N):
                cm = valid[:, f, t, n]
                if cm.sum() < min_cams:
                    continue
                rows = []
                for c in np.where(cm)[0]:
                    u, v = pts_und[c, f, t, n]
                    rows.append(u * Ps[c, 2] - Ps[c, 0])
                    rows.append(v * Ps[c, 2] - Ps[c, 1])
                A = np.array(rows)
                _, _, Vt = np.linalg.svd(A)
                X = Vt[-1]
                out[f, t, n] = X[:3] / X[3]
    return out


def procrustes_scale(src, dst):
    """Best rigid+scale mapping src->dst (Umeyama). Returns (s,R,t)."""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    sc, dc = src - mu_s, dst - mu_d
    H = sc.T @ dc / len(src)
    U, S, Vt = np.linalg.svd(H)
    D = np.eye(3)
    if np.linalg.det(U @ Vt) < 0:
        D[2, 2] = -1
    R = (Vt.T @ D @ U.T)
    var_s = (sc ** 2).sum() / len(src)
    s = np.trace(np.diag(S) @ D) / var_s
    t = mu_d - s * R @ mu_s
    return s, R, t


def ransac_align(src, dst, iters=300, thresh=5.0, seed=0):
    """Robust src->dst rigid+scale. thresh in mm (dst units)."""
    rng = np.random.default_rng(seed)
    n = len(src)
    best_in, best = None, None
    for _ in range(iters):
        idx = rng.choice(n, 4, replace=False)
        try:
            s, R, t = procrustes_scale(src[idx], dst[idx])
        except np.linalg.LinAlgError:
            continue
        pred = (s * (R @ src.T).T + t)
        err = np.linalg.norm(pred - dst, axis=1)
        inl = err < thresh
        if best_in is None or inl.sum() > best_in.sum():
            best_in, best = inl, (s, R, t)
    # refit on inliers
    s, R, t = procrustes_scale(src[best_in], dst[best_in])
    pred = (s * (R @ src.T).T + t)
    err = np.linalg.norm(pred - dst, axis=1)
    return (s, R, t), err, best_in


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    calib_toml = glob.glob(os.path.join(SESSION_DIR, "calibration", "*_calibration.toml"))[0]
    cams = load_calibration(calib_toml)
    print("calib cams:", list(cams))

    p3_file = glob.glob(os.path.join(SESSION_DIR, "*points3d*.h5"))[0]
    with h5py.File(p3_file) as f:
        Xp = f["tracks"][:]  # (F, 2, 15, 3)  proofread identity, P-frame
    F3 = Xp.shape[0]
    print("points3d:", Xp.shape, "nan-frac", np.isnan(Xp).mean().round(4))

    # raw SLEAP 2D per cam
    sleap = {}
    for s in SERIALS:
        a = glob.glob(os.path.join(SESSION_DIR, s, "*.analysis.h5"))[0]
        sleap[s] = load_sleap_2d(a)
    F2 = min(v.shape[0] for v in sleap.values())
    F = min(F2, F3)
    print("frames: 2d", F2, "3d", F3, "-> use", F)

    # ---- recover transform on a sample of frames ----
    # use every Kth frame across the recording for a global fit
    samp = np.arange(0, F, max(1, F // 4000))
    C = len(SERIALS)
    und = np.stack([undistort(sleap[s][samp], cams[s]["K"], cams[s]["dist"]) for s in SERIALS], 0)
    Ps = np.stack([cams[s]["P"] for s in SERIALS], 0)
    Xc = triangulate_all(und, Ps)                    # (len(samp),2,15,3) calib frame, SLEAP identity
    Xp_s = Xp[samp]

    both = (~np.isnan(Xc).any(-1)) & (~np.isnan(Xp_s).any(-1))
    src = Xp_s[both]      # P-frame
    dst = Xc[both]        # calib-frame
    print("correspondences for transform fit:", len(src))
    (s, R, t), err, inl = ransac_align(src, dst, thresh=8.0)
    print(f"transform: scale={s:.5f}  inliers={inl.mean():.3f}  "
          f"inlier-resid mm: med={np.median(err[inl]):.2f} p90={np.percentile(err[inl],90):.2f}")

    # ---- apply transform to ALL proofread 3D, reproject to each cam ----
    Xp_use = Xp[:F].reshape(-1, 3)
    good = ~np.isnan(Xp_use).any(1)
    Xcal = np.full_like(Xp_use, np.nan)
    Xcal[good] = (s * (R @ Xp_use[good].T).T + t)
    Xcal = Xcal.reshape(F, 2, 15, 3)

    val_report = {}
    for si, serial in enumerate(SERIALS):
        cam = cams[serial]
        flat3 = Xcal.reshape(-1, 3)
        g = ~np.isnan(flat3).any(1)
        proj = np.full((flat3.shape[0], 2), np.nan)
        if g.any():
            pp, _ = cv2.projectPoints(flat3[g], cam["rvec"], cam["t"], cam["K"], cam["dist"])
            proj[g] = pp.reshape(-1, 2)
        gt2d = proj.reshape(F, 2, 15, 2)   # (F,T,N,2) distorted px, proofread identity

        # validate vs raw SLEAP 2D (min over track assignment, since SLEAP id may swap)
        sl = sleap[serial][:F]
        m = (~np.isnan(gt2d).any(-1)) & (~np.isnan(sl).any(-1))
        # direct (same-track) residual
        d_same = np.linalg.norm(gt2d - sl, axis=-1)[m]
        val_report[serial] = float(np.nanmedian(d_same)) if m.any() else float("nan")

        # write GT analysis.h5:  tracks (T,2,N,F)
        tracks = np.transpose(gt2d, (1, 3, 2, 0)).astype(np.float64)  # (T,2,N,F)
        occ = (~np.isnan(gt2d).any(-1)).any(-1).astype(np.uint8)      # (F,T) any-node-visible
        out = os.path.join(OUT_DIR, f"{serial}.gt.analysis.h5")
        with h5py.File(out, "w") as h:
            h.create_dataset("tracks", data=tracks, compression="gzip", compression_opts=1)
            h.create_dataset("track_occupancy", data=occ)
            h.create_dataset("node_names", data=np.array([b"n%d" % i for i in range(15)]))
        print(f"  wrote {out}  tracks{tracks.shape} occ{occ.shape}")

    print("\nreproj-GT vs raw SLEAP median px (same-track; high where SLEAP swapped):")
    for s_, v in val_report.items():
        print(f"  {s_}: {v:.2f} px")
    print("\nNOTE: low median (~few px) = calibration+transform correct. Elevated tail expected where raw SLEAP identity swapped (GT is the corrected one).")


if __name__ == "__main__":
    main()
