"""Evaluate the candidate triangulators of triangulate_variants.py against
aniposelib and LUC3D's shipped pair, on BMimica, held-out-view protocol.

Same protocol as compare_bmimica.py so the numbers are directly comparable:
solve from C-1 views, project into the held-out view, score against the real
detection there. The held-out view is never used by any method, so a method
cannot win by overfitting the metric.

All variants are run in UNDISTORTED PIXEL space with P = K[R|t], so "reprojection
error" means the same thing for every method and for the score.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import h5py
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from compare_triangulators import load_calibration, reproject          # noqa: E402
from triangulate_variants import VARIANTS                              # noqa: E402

REPO = Path("/root/vast/eric/luc3d-bench")


def load_gt(path: Path):
    with h5py.File(path, "r") as h:
        t = np.asarray(h["tracks"])                    # (A, 2, K, T)
        sc = np.asarray(h["point_scores"]) if "point_scores" in h else None
    xy = np.transpose(t, (3, 0, 2, 1))                 # (T, A, K, 2)
    conf = np.transpose(sc, (2, 0, 1)) if sc is not None else None   # (T, A, K)
    return xy, conf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--master", type=Path, default=REPO / "outputs/bmimica/bmimica_master.tsv")
    ap.add_argument("--raw-root", type=str, default="/root/vast/eric/BMimica")
    ap.add_argument("--stride", type=int, default=360)
    ap.add_argument("--sessions", type=int, default=0)
    ap.add_argument("--out", type=Path, default=REPO / "outputs/fig8/bmimica_variants.csv")
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
            continue
        cams = load_calibration(cal_path)
        names = list(cams)

        gt, conf = {}, {}
        miss = False
        for cam in names:
            cand = sorted((Path(args.raw_root) / sess / cam).glob(f"{cam}-*.analysis.h5"))
            if not cand:
                miss = True
                break
            gt[cam], conf[cam] = load_gt(cand[0])
        if miss:
            continue

        T = min(v.shape[0] for v in gt.values())
        A = min(v.shape[1] for v in gt.values())
        Kn = min(v.shape[2] for v in gt.values())
        idx = np.arange(0, T, args.stride)
        flat = {c: gt[c][idx][:, :A, :Kn].reshape(-1, 2) for c in names}
        flatc = {c: (conf[c][idx][:, :A, :Kn].reshape(-1) if conf[c] is not None
                     else np.ones(len(flat[c]))) for c in names}

        print(f"\n=== {sess} ({si+1}/{len(master)}) {len(idx)} frames ===", flush=True)
        for held in names:
            keep = [c for c in names if c != held]
            kc = [cams[c] for c in keep]
            obs = np.stack([flat[c] for c in keep], axis=0)
            cf = np.stack([flatc[c] for c in keep], axis=0)
            truth = flat[held]

            seen = np.all(np.isfinite(truth), axis=1)
            nvis = np.sum(np.all(np.isfinite(obs), axis=2), axis=0)
            use = seen & (nvis >= 2)
            if not use.any():
                continue
            obs_u = obs[:, use, :]
            cf_u = np.nan_to_num(cf[:, use], nan=0.0)

            # Undistort into ideal-pinhole pixels once; every variant shares it.
            und = np.full_like(obs_u, np.nan)
            for ci, cam in enumerate(kc):
                ok = np.all(np.isfinite(obs_u[ci]), axis=1)
                if ok.any():
                    und[ci][ok] = cv2.undistortPoints(
                        obs_u[ci][ok].reshape(-1, 1, 2), cam["K"], cam["dist"],
                        P=cam["K"]).reshape(-1, 2)
            P = np.stack([cam["P"] for cam in kc], axis=0)

            results = {}
            for name, fn in VARIANTS.items():
                try:
                    results[name] = fn(und, P, cf_u)
                except Exception as e:
                    print(f"   {name} failed: {e}")

            cg = CameraGroup.load(str(cal_path))
            if hasattr(cg, "subset_cameras_names"):
                cg = cg.subset_cameras_names(keep)
            try:
                results["aniposelib"] = np.asarray(
                    cg.triangulate(obs_u.astype(np.float64), progress=False))
            except Exception as e:
                print(f"   aniposelib failed: {e}")

            line = []
            for method, p3 in results.items():
                err = np.linalg.norm(reproject(p3, cams[held]) - truth[use], axis=1)
                err = err[np.isfinite(err)]
                if err.size == 0:
                    continue
                med = float(np.median(err))
                rows.append({"session": sess, "held_out": held, "method": method,
                             "n": int(err.size), "median_px": med,
                             "p95": float(np.percentile(err, 95))})
                line.append(f"{method}={med:.3f}")
            print("  " + held + "  " + "  ".join(line), flush=True)

        args.out.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(args.out, index=False)

    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)
    per = df.groupby(["session", "method"]).median_px.mean().reset_index()
    print("\nMean over held-out cameras, pooled over sessions:")
    print(per.groupby("method").median_px.agg(["mean", "median", "count"]).round(4).sort_values("mean").to_string())
    w = per.pivot(index="session", columns="method", values="median_px")
    if "aniposelib" in w:
        print("\nvs aniposelib (negative = better):")
        for m in sorted(w.columns):
            if m == "aniposelib":
                continue
            d = w[m] - w["aniposelib"]
            print(f"  {m:14s} mean {d.mean():+.4f} px  wins {int((d < 0).sum())}/{len(d)}")


if __name__ == "__main__":
    main()
