"""Fig 8b/8c: LUC3D's triangulators against Lightning Pose 3D's, on identical 2D
and identical calibration, scored by HELD-OUT-VIEW reprojection error.

Protocol is Fig 4d's, unchanged: solve the 3D point from k of the C views,
project it into the view that was held out, and score against the raw detection
there. Reference-free -- no method is scored on the objective it minimises, which
is the whole reason Fig 4 refuses to rank solvers against the proofread 3D.

Methods
  luc3d_dlt      pose/triangulation.js:triangulatePointDLT   (via the node bridge)
  luc3d_refined  pose/triangulation.js:triangulatePointBA    (via the node bridge)
  eks            `eks multicam --calibration ...`            (Lightning Pose 3D's
                 own documented 3D path: aniposelib DLT + nonlinear EKS with
                 variance inflation). Run per held-out camera on a reduced
                 calibration so it sees exactly the views LUC3D sees.

NOTE on fairness, to be carried into the caption: EKS is an ENSEMBLE method that
also smooths TEMPORALLY. Here it gets its ensemble (all seeds) and its smoothing,
while LUC3D gets the per-frame ensemble mean and no temporal information at all.
That favours EKS on this metric, not LUC3D. Stated rather than corrected.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

try:
    import tomllib
except ModuleNotFoundError:  # py<3.11
    import tomli as tomllib

BRIDGE = Path(__file__).with_name("luc3d_triangulate_bridge.mjs")


# --------------------------------------------------------------------------- #
# calibration
# --------------------------------------------------------------------------- #
def load_calibration(path: Path) -> dict:
    """anipose .toml -> {name: {K, dist, R, t, P}}; P = K[R|t] for ideal pinhole."""
    raw = tomllib.loads(path.read_text())
    cams = {}
    for key, c in raw.items():
        if not key.startswith("cam"):
            continue
        K = np.array(c["matrix"], dtype=np.float64)
        dist = np.array(c["distortions"], dtype=np.float64).ravel()
        rvec = np.array(c["rotation"], dtype=np.float64).ravel()
        tvec = np.array(c["translation"], dtype=np.float64).ravel()
        R, _ = cv2.Rodrigues(rvec)
        Rt = np.hstack([R, tvec.reshape(3, 1)])
        cams[c["name"]] = {
            "K": K, "dist": dist, "rvec": rvec, "tvec": tvec,
            "R": R, "P": K @ Rt, "toml_key": key, "raw": c,
        }
    return cams


def write_subset_calibration(cal_path: Path, keep: list[str], out: Path) -> None:
    """Re-emit a calibration containing only `keep`, renumbered cam_0..cam_n."""
    raw = tomllib.loads(cal_path.read_text())
    by_name = {c["name"]: c for k, c in raw.items() if k.startswith("cam")}
    lines = []
    for i, name in enumerate(keep):
        c = by_name[name]
        lines.append(f"[cam_{i}]")
        lines.append(f'name = "{c["name"]}"')
        lines.append(f"size = {list(c['size'])}")
        lines.append(f"matrix = {[list(map(float, r)) for r in c['matrix']]}")
        lines.append(f"distortions = {list(map(float, c['distortions']))}")
        lines.append(f"rotation = {list(map(float, c['rotation']))}")
        lines.append(f"translation = {list(map(float, c['translation']))}")
        lines.append("")
    out.write_text("\n".join(lines))


# --------------------------------------------------------------------------- #
# 2D input
# --------------------------------------------------------------------------- #
def load_seed_csvs(input_dir: Path, cam: str) -> list[pd.DataFrame]:
    files = sorted(p for p in input_dir.glob("*.csv") if cam in p.name)
    if not files:
        raise SystemExit(f"no CSVs for camera {cam} in {input_dir}")
    return [pd.read_csv(p, header=[0, 1, 2], index_col=0) for p in files], files


def ensemble_mean_xy(dfs: list[pd.DataFrame], bodyparts: list[str]) -> np.ndarray:
    """-> (T, K, 2) mean over seeds."""
    stacks = []
    for df in dfs:
        cols = df.columns
        scorer = cols[0][0]
        xy = np.stack(
            [np.stack([df[(scorer, bp, "x")].to_numpy(),
                       df[(scorer, bp, "y")].to_numpy()], axis=-1)
             for bp in bodyparts], axis=1)
        stacks.append(xy)
    return np.mean(np.stack(stacks, axis=0), axis=0)


def bodypart_list(df: pd.DataFrame) -> list[str]:
    seen, out = set(), []
    for _, bp, _ in df.columns:
        if bp not in seen:
            seen.add(bp)
            out.append(bp)
    return out


# --------------------------------------------------------------------------- #
# LUC3D via the node bridge
# --------------------------------------------------------------------------- #
def luc3d_triangulate(obs_by_cam: np.ndarray, cams: list[dict], refine: bool):
    """obs_by_cam (C, N, 2) in DISTORTED px -> (N,3) dlt, (N,3) refined.

    LUC3D's DLT minimises an algebraic error in IDEAL-PINHOLE coordinates, so the
    observations are undistorted into pixel space first and paired with P=K[R|t].
    """
    C, N, _ = obs_by_cam.shape
    und = np.empty_like(obs_by_cam)
    for ci, cam in enumerate(cams):
        pts = obs_by_cam[ci].reshape(-1, 1, 2).astype(np.float64)
        und[ci] = cv2.undistortPoints(pts, cam["K"], cam["dist"], P=cam["K"]).reshape(-1, 2)

    cases = []
    for n in range(N):
        obs, Ps = [], []
        for ci, cam in enumerate(cams):
            p = und[ci, n]
            if not np.all(np.isfinite(p)):
                obs.append(None)
                Ps.append(None)
            else:
                obs.append([float(p[0]), float(p[1])])
                Ps.append([list(map(float, row)) for row in cam["P"]])
        cases.append({"obs": obs, "P": Ps})

    proc = subprocess.run(
        ["node", str(BRIDGE)],
        input=json.dumps({"cases": cases, "refine": refine}),
        capture_output=True, text=True,
    )
    if proc.returncode != 0:
        raise SystemExit(f"bridge failed:\n{proc.stderr}")
    res = json.loads(proc.stdout)

    def to_arr(lst):
        out = np.full((N, 3), np.nan)
        for i, p in enumerate(lst):
            if p is not None and len(p) == 3:
                out[i] = p
        return out

    return to_arr(res["dlt"]), to_arr(res["ba"])


# --------------------------------------------------------------------------- #
# EKS
# --------------------------------------------------------------------------- #
def run_eks(input_dir: Path, files_by_cam: dict, keep: list[str], cal_path: Path,
            bodyparts: list[str], eks_bin: Path, workdir: Path) -> np.ndarray:
    """-> (T, K, 3) in the calibration's world frame."""
    sub = workdir / ("eks_in_" + "_".join(keep))
    sub.mkdir(parents=True, exist_ok=True)
    for cam in keep:
        for f in files_by_cam[cam]:
            tgt = sub / f.name
            if not tgt.exists():
                shutil.copy(f, tgt)
    cal = workdir / ("cal_" + "_".join(keep) + ".toml")
    write_subset_calibration(cal_path, keep, cal)
    out = workdir / ("eks_out_" + "_".join(keep))

    cmd = [str(eks_bin), "multicam", "--input-dir", str(sub),
           "--calibration", str(cal), "--save-dir", str(out),
           "--bodypart-list", *bodyparts]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    res = out / "multicam_3d_results.csv"
    if not res.exists():
        raise SystemExit(f"EKS produced no 3D for {keep}\nSTDOUT{proc.stdout[-2000:]}\n"
                         f"STDERR{proc.stderr[-2000:]}")
    df = pd.read_csv(res, header=[0, 1, 2], index_col=0)
    scorer = df.columns[0][0]
    return np.stack([np.stack([df[(scorer, bp, "x")].to_numpy(),
                               df[(scorer, bp, "y")].to_numpy(),
                               df[(scorer, bp, "z")].to_numpy()], axis=-1)
                     for bp in bodyparts], axis=1)


# --------------------------------------------------------------------------- #
def reproject(pts3d: np.ndarray, cam: dict) -> np.ndarray:
    """(N,3) -> (N,2) in DISTORTED px, i.e. the space the detections live in."""
    ok = np.all(np.isfinite(pts3d), axis=1)
    out = np.full((len(pts3d), 2), np.nan)
    if ok.sum():
        proj, _ = cv2.projectPoints(pts3d[ok].reshape(-1, 1, 3),
                                    cam["rvec"], cam["tvec"], cam["K"], cam["dist"])
        out[ok] = proj.reshape(-1, 2)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", type=Path, required=True)
    ap.add_argument("--calibration", type=Path, required=True)
    ap.add_argument("--eks-bin", type=Path,
                    default=Path("/root/vast/eric/luc3d-bench/eks_env/bin/eks"))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--max-frames", type=int, default=0)
    args = ap.parse_args()

    cams = load_calibration(args.calibration)
    names = list(cams)
    print(f"cameras: {names}")

    dfs_by_cam, files_by_cam = {}, {}
    for cam in names:
        dfs, files = load_seed_csvs(args.input_dir, cam)
        dfs_by_cam[cam], files_by_cam[cam] = dfs, files
    bps = bodypart_list(dfs_by_cam[names[0]][0])
    print(f"{len(bps)} bodyparts, {len(dfs_by_cam[names[0]])} seeds/camera")

    xy = {cam: ensemble_mean_xy(dfs_by_cam[cam], bps) for cam in names}
    T = min(v.shape[0] for v in xy.values())
    if args.max_frames:
        T = min(T, args.max_frames)
    print(f"{T} frames")

    workdir = Path(tempfile.mkdtemp(prefix="fig8_"))
    rows = []

    for held in names:
        keep = [c for c in names if c != held]
        print(f"\n=== held out: {held}  (solve from {keep}) ===")

        obs = np.stack([xy[c][:T].reshape(-1, 2) for c in keep], axis=0)
        dlt, ref = luc3d_triangulate(obs, [cams[c] for c in keep], refine=True)

        eks3d = run_eks(args.input_dir, files_by_cam, keep, args.calibration,
                        bps, args.eks_bin, workdir)[:T].reshape(-1, 3)

        truth = xy[held][:T].reshape(-1, 2)
        for method, p3 in (("luc3d_dlt", dlt), ("luc3d_refined", ref), ("eks", eks3d)):
            err = np.linalg.norm(reproject(p3, cams[held]) - truth, axis=1)
            err = err[np.isfinite(err)]
            rows.append({
                "held_out": held, "method": method, "n": int(err.size),
                "median_px": float(np.median(err)),
                "p25": float(np.percentile(err, 25)),
                "p75": float(np.percentile(err, 75)),
                "p95": float(np.percentile(err, 95)),
                "mean_px": float(np.mean(err)),
            })
            print(f"  {method:16s} n={err.size:6d}  median={np.median(err):8.3f} px"
                  f"  p95={np.percentile(err,95):8.3f}")

    df = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"\nwrote {args.out}")
    print(df.pivot(index="held_out", columns="method", values="median_px").round(3))


if __name__ == "__main__":
    sys.exit(main())
