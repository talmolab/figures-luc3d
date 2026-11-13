"""Fig 4f extension: solve time per keypoint for LUC3D's DLT, LUC3D's refinement, and
aniposelib (Lightning Pose 3D's shipped triangulator), on identical real data.

READ THIS BEFORE PUTTING THE THREE BARS SIDE BY SIDE. The comparison is not
apples-to-apples and the caption has to say so in one sentence, because the
difference in the numbers is mostly a difference in *how the work is batched*, not in
how much arithmetic each method does:

  * LUC3D solves ONE keypoint per call, in JavaScript, because that is what the app
    does -- a user triangulates a frame at a time and the solver is called per point.
    The per-call overhead is part of the app's real cost and belongs in the number.
  * aniposelib solves a WHOLE ARRAY of keypoints per call, in numpy, so its per-point
    cost is amortised over the batch and most of the loop runs in C. Calling it once
    per point instead would be far slower and would not be how anyone uses it.

So the honest framing for the panel is "cost per keypoint as each library is actually
used", not "which algorithm is faster". Both batched and per-call aniposelib timings
are reported here so the caption can quote the right one and state the other.

Timing method: LUC3D's is taken inside the node process around the solve calls only
(the bridge's JSON transport is excluded); aniposelib's with perf_counter around the
call. Same machine, same points, single-threaded, best of `--repeats` passes.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import cv2
import h5py
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from compare_triangulators import load_calibration                    # noqa: E402

BRIDGE = Path(__file__).with_name("luc3d_triangulate_bridge.mjs")
REPO = Path("/root/vast/eric/luc3d-bench")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--master", type=Path, default=REPO / "outputs/bmimica/bmimica_master.tsv")
    ap.add_argument("--raw-root", type=str, default="/root/vast/eric/BMimica")
    ap.add_argument("--n-points", type=int, default=20000)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--out", type=Path, default=REPO / "outputs/fig8/triangulator_time.csv")
    args = ap.parse_args()

    from aniposelib.cameras import CameraGroup

    master = pd.read_csv(args.master, sep="\t")
    row = master.iloc[0]
    sess = str(row["session"])
    cal_path = Path(row["calibration_toml"])
    cams = load_calibration(cal_path)
    names = list(cams)

    gt = {}
    for cam in names:
        p = sorted((Path(args.raw_root) / sess / cam).glob(f"{cam}-*.analysis.h5"))[0]
        with h5py.File(p, "r") as h:
            gt[cam] = np.transpose(np.asarray(h["tracks"]), (3, 0, 2, 1))

    T = min(v.shape[0] for v in gt.values())
    A = min(v.shape[1] for v in gt.values())
    K = min(v.shape[2] for v in gt.values())
    flat = {c: gt[c][:T, :A, :K].reshape(-1, 2) for c in names}
    obs = np.stack([flat[c] for c in names], axis=0)
    ok = np.all(np.isfinite(obs), axis=2).all(axis=0)
    obs = obs[:, ok, :][:, : args.n_points, :]
    n = obs.shape[1]
    print(f"{sess}: {n} keypoints seen in all {len(names)} cameras")

    und = np.empty_like(obs)
    for ci, cam in enumerate(names):
        und[ci] = cv2.undistortPoints(obs[ci].reshape(-1, 1, 2), cams[cam]["K"],
                                      cams[cam]["dist"], P=cams[cam]["K"]).reshape(-1, 2)

    rows = []
    BRANCH = "/root/vast/eric/sleap-3d-gui/scratch/repos/lucid-bundle-adj"
    camDefs = [{"name": c, "matrix": [list(map(float, r)) for r in cams[c]["K"]],
                "dist": list(map(float, cams[c]["dist"])),
                "rvec": list(map(float, cams[c]["rvec"])),
                "tvec": list(map(float, cams[c]["tvec"])),
                "size": [1280, 1024]} for c in names]

    # ---- LUC3D, one keypoint per call (how the app calls it) --------------- #
    cases = [{"obs": [[float(und[ci, i, 0]), float(und[ci, i, 1])] for ci in range(len(names))],
              "P": [[list(map(float, r)) for r in cams[c]["P"]] for c in names]}
             for i in range(n)]
    proc = subprocess.run(["node", str(BRIDGE)],
                          input=json.dumps({"cases": cases, "refine": True,
                                            "bench": True, "repeats": args.repeats}),
                          capture_output=True, text=True)
    if proc.returncode != 0:
        raise SystemExit(proc.stderr[-3000:])
    t = json.loads(proc.stdout)["timing"]
    rows.append({"method": "luc3d_dlt", "mode": "per-call (JS, as the app calls it)",
                 "us_per_keypoint": t["dlt_us_per_keypoint"], "n": n})
    rows.append({"method": "luc3d_refined_total", "mode": "per-call (JS): DLT seed + refinement",
                 "us_per_keypoint": t["refined_total_us_per_keypoint"], "n": n})
    rows.append({"method": "luc3d_refine_only", "mode": "per-call (JS): refinement step alone",
                 "us_per_keypoint": t["refine_only_us_per_keypoint"], "n": n})
    rows.append({"method": "luc3d_dlt", "mode": "BATCHED (JS): triangulatePoints, as the app calls it",
                 "us_per_keypoint": t.get("dlt_batched_us_per_keypoint"), "n": n})
    rows.append({"method": "luc3d_refined_total", "mode": "BATCHED (JS): triangulatePoints + triangulatePointsBA",
                 "us_per_keypoint": t.get("refined_batched_total_us_per_keypoint"), "n": n})

    # ---- the lucid-bundle-adj refinement, which is what Fig 4 actually times #
    # main's triangulatePointBA is plain Levenberg-Marquardt on squared error in
    # ideal-pinhole pixels; the soft-L1 + L1-polish native-space solver Fig 4
    # describes lives only on that worktree. Time both and let the caption say which.
    for i, c in enumerate(cases):
        c["raw"] = [[float(obs[ci, i, 0]), float(obs[ci, i, 1])] for ci in range(len(names))]
    import os
    env = dict(os.environ, LUCID_DIR=BRANCH)
    proc2 = subprocess.run(["node", str(BRIDGE)], env=env,
                           input=json.dumps({"cases": cases, "refine": True, "bench": True,
                                             "repeats": args.repeats, "camDefs": camDefs}),
                           capture_output=True, text=True)
    if proc2.returncode == 0:
        t2 = json.loads(proc2.stdout)["timing"]
        rows.append({"method": "luc3d_refine_only_branch",
                     "mode": "per-call (JS): soft-L1+polish, native space (lucid-bundle-adj)",
                     "us_per_keypoint": t2["refine_only_us_per_keypoint"], "n": n})
        rows.append({"method": "luc3d_refined_total_branch",
                     "mode": "per-call (JS): DLT seed + soft-L1+polish (lucid-bundle-adj)",
                     "us_per_keypoint": t2["refined_total_us_per_keypoint"], "n": n})
    else:
        print("branch solver bench failed:\n" + proc2.stderr[-1500:])

    # ---- aniposelib, batched (how anyone actually uses it) ----------------- #
    cg = CameraGroup.load(str(cal_path))
    pts = obs.astype(np.float64)
    cg.triangulate(pts[:, :64], progress=False)          # warm up numpy/JIT paths
    best = np.inf
    for _ in range(args.repeats):
        t0 = time.perf_counter()
        cg.triangulate(pts, progress=False)
        best = min(best, time.perf_counter() - t0)
    rows.append({"method": "aniposelib", "mode": "batched (numpy, whole array per call)",
                 "us_per_keypoint": 1e6 * best / n, "n": n})

    # ---- aniposelib, one keypoint per call (the like-for-like control) ----- #
    m = min(n, 2000)
    t0 = time.perf_counter()
    for i in range(m):
        cg.triangulate(pts[:, i:i + 1], progress=False)
    per = time.perf_counter() - t0
    rows.append({"method": "aniposelib", "mode": "per-call (numpy, one point per call)",
                 "us_per_keypoint": 1e6 * per / m, "n": m})

    df = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print()
    print(df.to_string(index=False))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
