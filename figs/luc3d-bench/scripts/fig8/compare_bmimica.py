"""Fig 8b: LUC3D's triangulators against Lightning Pose 3D's, on the 50 BMimica
sessions (5 cameras, 2 mice, 15 nodes), scored by HELD-OUT-VIEW reprojection error.

Same protocol as Fig 4d and as scripts/fig8/compare_triangulators.py: solve from
k of the C views, project into the view held out, score against the detection
there. Reference-free.

Methods
  luc3d_dlt        pose/triangulation.js:triangulatePointDLT   (node bridge, real app code)
  luc3d_refined    pose/triangulation.js:triangulatePointBA    (node bridge, real app code)
  aniposelib       aniposelib CameraGroup.triangulate          -- this IS the triangulation
                   stage inside `eks multicam --calibration`, i.e. Lightning Pose 3D's
                   shipped 3D path
  pairwise_median  OpenCV undistortPoints + triangulatePoints per camera PAIR, median
                   over the C(C,2) pairs -- the triangulation the LP3D paper describes

WHY THE FULL EKS SMOOTHER IS NOT ONE OF THE METHODS HERE. EKS is an ensemble
method: its observation noise comes from the variance across several independently
seeded networks, and BMimica has ONE detection set. Running it on a single seed
would either divide by a zero ensemble variance or silently make up an uncertainty
it was never given. So on this corpus we compare the part of Lightning Pose 3D's
3D path that a single-seed dataset can legitimately exercise -- its triangulator --
and the full smoother is exercised on their own 3-seed fly data by
compare_triangulators.py. That split is a limitation of OUR corpus, not of EKS,
and the caption has to say so in those words.

Ground truth 2D is the proofread per-camera analysis h5 (identity consistent across
cameras), so every method sees identical, correctly associated input and the
comparison isolates the solve.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import cv2
import h5py
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from compare_triangulators import load_calibration, reproject  # noqa: E402

BRIDGE = Path(__file__).with_name("luc3d_triangulate_bridge.mjs")
REPO = Path("/root/vast/eric/luc3d-bench")


def load_gt(path: Path) -> np.ndarray:
    """proofread.analysis.h5 -> (T, A, K, 2)."""
    with h5py.File(path, "r") as h:
        t = np.asarray(h["tracks"])          # (A, 2, K, T)
    return np.transpose(t, (3, 0, 2, 1))     # (T, A, K, 2)


# --------------------------------------------------------------------------- #
def luc3d_batch(obs: np.ndarray, cams: list[dict], refine: bool, chunk: int = 40000,
                lucid_dir: str | None = None, names: list[str] | None = None):
    """obs (C, N, 2) DISTORTED px -> (N,3) dlt, (N,3) refined.

    lucid_dir selects WHICH triangulation.js to run. main ships plain
    Levenberg-Marquardt on squared error in ideal-pinhole pixels; the soft-L1 +
    L1-polish native-space solver that Fig 4 describes and times lives only on the
    lucid-bundle-adj worktree. When lucid_dir points there, the raw (distorted)
    observations and Camera definitions are sent so the solver runs in its native
    mode, exactly as figs/fig4_measure.mjs invokes it."""
    C, N, _ = obs.shape
    und = np.full_like(obs, np.nan)
    for ci, cam in enumerate(cams):
        ok = np.all(np.isfinite(obs[ci]), axis=1)
        if ok.any():
            pts = obs[ci][ok].reshape(-1, 1, 2).astype(np.float64)
            und[ci][ok] = cv2.undistortPoints(pts, cam["K"], cam["dist"],
                                              P=cam["K"]).reshape(-1, 2)
    Ps = [[list(map(float, r)) for r in cam["P"]] for cam in cams]

    dlt = np.full((N, 3), np.nan)
    ref = np.full((N, 3), np.nan)
    for s in range(0, N, chunk):
        e = min(s + chunk, N)
        cases = []
        for n in range(s, e):
            o, p = [], []
            for ci in range(C):
                v = und[ci, n]
                if np.all(np.isfinite(v)):
                    o.append([float(v[0]), float(v[1])])
                    p.append(Ps[ci])
                else:
                    o.append(None)
                    p.append(None)
            cases.append({"obs": o, "P": p})
        payload = {"cases": cases, "refine": refine}
        env = None
        if lucid_dir:
            import os
            env = dict(os.environ, LUCID_DIR=lucid_dir)
            payload["camDefs"] = [
                {"name": (names[ci] if names else f"c{ci}"),
                 "matrix": [list(map(float, r)) for r in cams[ci]["K"]],
                 "dist": list(map(float, cams[ci]["dist"])),
                 "rvec": list(map(float, cams[ci]["rvec"])),
                 "tvec": list(map(float, cams[ci]["tvec"])),
                 "size": [1280, 1024]}
                for ci in range(C)]
            for j, cse in enumerate(payload["cases"]):
                gi = s + j
                cse["raw"] = [
                    ([float(obs[ci, gi, 0]), float(obs[ci, gi, 1])]
                     if np.all(np.isfinite(obs[ci, gi])) else None)
                    for ci in range(C)]
        proc = subprocess.run(["node", str(BRIDGE)], env=env,
                              input=json.dumps(payload),
                              capture_output=True, text=True)
        if proc.returncode != 0:
            raise SystemExit(f"bridge failed:\n{proc.stderr[-3000:]}")
        r = json.loads(proc.stdout)
        for i, (a, b) in enumerate(zip(r["dlt"], r["ba"])):
            if a is not None and len(a) == 3:
                dlt[s + i] = a
            if b is not None and len(b) == 3:
                ref[s + i] = b
    return dlt, ref


def aniposelib_triangulate(obs: np.ndarray, cgroup) -> np.ndarray:
    """obs (C, N, 2) -> (N, 3). The triangulator inside `eks multicam`."""
    return np.asarray(cgroup.triangulate(obs.astype(np.float64), progress=False))


def pairwise_median_triangulate(obs: np.ndarray, cams: list[dict]) -> np.ndarray:
    """Per-pair DLT in normalised coords, median over pairs (the LP3D paper's)."""
    C, N, _ = obs.shape
    norm, Prt = [], []
    for ci, cam in enumerate(cams):
        pts = obs[ci].reshape(-1, 1, 2).astype(np.float64)
        n = np.full((N, 2), np.nan)
        ok = np.all(np.isfinite(obs[ci]), axis=1)
        if ok.any():
            n[ok] = cv2.undistortPoints(pts[ok], cam["K"], cam["dist"]).reshape(-1, 2)
        norm.append(n)
        Prt.append(np.hstack([cam["R"], cam["tvec"].reshape(3, 1)]))

    ests = []
    for i in range(C):
        for j in range(i + 1, C):
            ok = np.all(np.isfinite(norm[i]), axis=1) & np.all(np.isfinite(norm[j]), axis=1)
            out = np.full((N, 3), np.nan)
            if ok.any():
                X = cv2.triangulatePoints(Prt[i], Prt[j],
                                          norm[i][ok].T.copy(), norm[j][ok].T.copy())
                with np.errstate(invalid="ignore", divide="ignore"):
                    out[ok] = (X[:3] / X[3]).T
            ests.append(out)
    E = np.stack(ests, axis=0)
    with np.errstate(invalid="ignore"):
        return np.nanmedian(E, axis=0)


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--master", type=Path,
                    default=REPO / "outputs/bmimica/bmimica_master.tsv")
    ap.add_argument("--stride", type=int, default=360,
                    help="frame stride; 360 gives ~500 frames per session")
    ap.add_argument("--sessions", type=int, default=0, help="0 = all")
    ap.add_argument("--out", type=Path, default=REPO / "outputs/fig8/bmimica_triangulators.csv")
    ap.add_argument("--no-refine", action="store_true")
    ap.add_argument("--branch-dir", type=str, default="",
                    help="also evaluate the soft-L1+polish solver from this checkout")
    ap.add_argument("--raw-root", type=str, default="/root/vast/eric/BMimica",
                    help="root holding <session>/<camera>/<camera>-*.analysis.h5")
    args = ap.parse_args()

    from aniposelib.cameras import CameraGroup

    master = pd.read_csv(args.master, sep="\t")
    if args.sessions:
        master = master.head(args.sessions)

    rows = []
    for si, row in master.iterrows():
        sess = str(row["session"])
        cal_path = Path(row["calibration_toml"])
        if not cal_path.exists():
            print(f"[skip] {sess}: no calibration"); continue
        cams = load_calibration(cal_path)
        names = list(cams)

        # The REAL per-camera detections, not outputs/bmimica/gt/*/proofread.analysis.h5
        # -- that file holds REPROJECTIONS of a common 3D solve, so triangulating it
        # from C-1 views reproduces the held-out view to machine precision and every
        # method scores exactly 0.000 px. Verified: the first pilot did precisely that.
        gt = {}
        missing = False
        for cam in names:
            cand = sorted((Path(args.raw_root) / sess / cam).glob(f"{cam}-*.analysis.h5"))
            if not cand:
                missing = True; break
            gt[cam] = load_gt(cand[0])
        if missing:
            print(f"[skip] {sess}: missing per-camera analysis h5"); continue

        # Cameras in a session can disagree on frame count, animal-slot count and
        # node count (a per-camera SLEAP file carries its own track table). Crop
        # all three to the common minimum before flattening -- otherwise the
        # per-camera arrays have different N and np.stack raises "all input arrays
        # must have the same shape" partway through the corpus.
        T = min(v.shape[0] for v in gt.values())
        A = min(v.shape[1] for v in gt.values())
        Kn = min(v.shape[2] for v in gt.values())
        idx = np.arange(0, T, args.stride)
        # (C, N, 2) flattened over frames x animals x nodes
        flat = {c: gt[c][idx][:, :A, :Kn].reshape(-1, 2) for c in names}
        N = next(iter(flat.values())).shape[0]

        print(f"\n=== {sess}  ({si+1}/{len(master)})  {len(names)} cams, "
              f"{len(idx)} frames, {N} points ===", flush=True)

        for held in names:
            keep = [c for c in names if c != held]
            kc = [cams[c] for c in keep]
            obs = np.stack([flat[c] for c in keep], axis=0)
            truth = flat[held]

            # Only score points the held-out camera actually saw AND that at
            # least two kept cameras saw -- otherwise the comparison is scoring
            # missing data, not a solve.
            seen = np.all(np.isfinite(truth), axis=1)
            nvis = np.sum(np.all(np.isfinite(obs), axis=2), axis=0)
            use = seen & (nvis >= 2)
            if use.sum() == 0:
                continue
            obs_u = obs[:, use, :]

            cg = CameraGroup.load(str(cal_path))
            cg = cg.subset_cameras_names(keep) if hasattr(cg, "subset_cameras_names") else cg

            results = {}
            d, r = luc3d_batch(obs_u, kc, refine=not args.no_refine)
            results["luc3d_dlt"] = d
            if not args.no_refine:
                results["luc3d_refined"] = r
            if args.branch_dir:
                _, rb = luc3d_batch(obs_u, kc, refine=True,
                                    lucid_dir=args.branch_dir, names=keep)
                results["luc3d_refined_branch"] = rb
            try:
                results["aniposelib"] = aniposelib_triangulate(obs_u, cg)
            except Exception as e:
                print(f"   aniposelib failed: {e}")
            results["pairwise_median"] = pairwise_median_triangulate(obs_u, kc)

            for method, p3 in results.items():
                err = np.linalg.norm(reproject(p3, cams[held]) - truth[use], axis=1)
                err = err[np.isfinite(err)]
                if err.size == 0:
                    continue
                rows.append({
                    "session": sess, "held_out": held, "method": method,
                    "n": int(err.size), "median_px": float(np.median(err)),
                    "p75": float(np.percentile(err, 75)),
                    "p95": float(np.percentile(err, 95)),
                })
            line = "  ".join(f"{m}={np.median(np.linalg.norm(reproject(p, cams[held]) - truth[use], axis=1)[np.isfinite(np.linalg.norm(reproject(p, cams[held]) - truth[use], axis=1))]):.3f}"
                             for m, p in results.items())
            print(f"  held={held}  n={int(use.sum())}  {line}", flush=True)

        # checkpoint after every session
        args.out.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(args.out, index=False)

    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)
    print(f"\nwrote {args.out}  ({len(df)} rows)")
    if len(df):
        # Eric's ask: average across all held-out cameras.
        per_sess = df.groupby(["session", "method"])["median_px"].mean().reset_index()
        print("\nPer-method, mean of the 5 held-out-camera medians, pooled over sessions:")
        print(per_sess.groupby("method")["median_px"].agg(["mean", "median", "count"]).round(4))


if __name__ == "__main__":
    sys.exit(main())
