"""Fig 8d, the version that is actually a test: LUC3D's cross-view tracker on
social-DANNCE's REAL per-camera 2D detections.

TRIADS ships `COM/predict00/com3d.pickle` for all six sessions -- the full
`save_data` from their COM network, which `dannce/engine/utils/save.py` writes with
the comment "Save undistorted 2D COMs and their 3D triangulations". Each sample
holds, per camera, a (n_animals, 2) array of 2D detections plus a confidence.

Why this is a real test and the earlier BEDDING one was not: these are INDEPENDENT
per-camera network outputs. Triangulating them and reprojecting gives a median
cross-view residual of ~9.4 px (p95 ~33, max >3000, none below 0.01) against 0.00 px
for anything derived by projecting a single 3D estimate. For scale, our own detector
on BMimica sits at 2.58 px -- so this data is HARDER than ours, not easier, and it
has three animals rather than two.

What identity means here, stated precisely because it bounds the claim: the row index
of the COM array is the network's output CHANNEL, and channel order is what s-DANNCE
uses as identity -- a human fixes it once per cohort during labelling. So the
"ground truth" is itself a per-camera learned assignment that can be wrong. A
disagreement between LUC3D and the channel is therefore not automatically LUC3D's
error, which is why `triads_score.py` adjudicates disagreements by reprojection
residual rather than assuming the channel is right.

Distortion: the COMs are already undistorted (verified -- undistorting again moves
the cross-view residual by 0.04 px), so the emitted calibration carries ZERO
distortion coefficients. Handing LUCID the real coefficients would undistort twice.
"""
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np

from sdannce_prepare import load_dannce_cam


def sample_order(keys):
    """'<exp>_<frame>' -> sorted by (exp, frame)."""
    def parse(k):
        a, b = k.split("_")
        return (int(a), int(b))
    return sorted(keys, key=parse)


def write_toml(cams, names, size, out: Path) -> None:
    lines = []
    for i, (name, c) in enumerate(zip(names, cams)):
        lines += [
            f"[cam_{i}]",
            f'name = "{name}"',
            f"size = [{size[0]}, {size[1]}]",
            f"matrix = {[list(map(float, r)) for r in c['K']]}",
            # ZERO distortion on purpose: the COMs are already undistorted.
            "distortions = [0.0, 0.0, 0.0, 0.0, 0.0]",
            f"rotation = {list(map(float, c['rvec']))}",
            f"translation = {list(map(float, c['tvec']))}",
            "",
        ]
    out.write_text("\n".join(lines))


def triangulate_channel(coms, P, a):
    """DLT the channel-`a` detections across cameras -> (3,) or None."""
    A = []
    for xy, Pm in zip(coms, P):
        v = xy[a]
        if not np.all(np.isfinite(v)):
            continue
        A.append(v[0] * Pm[2] - Pm[0])
        A.append(v[1] * Pm[2] - Pm[1])
    if len(A) < 4:
        return None
    _, _, Vt = np.linalg.svd(np.asarray(A))
    X = Vt[-1]
    if abs(X[3]) < 1e-12:
        return None
    return X[:3] / X[3]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", type=str, default="")
    ap.add_argument("--size", type=int, nargs=2, default=[1920, 1200])
    args = ap.parse_args()

    # Absolute, so the emitted meta can be consumed from any working directory --
    # a relative blob path baked into the JSON breaks the node driver.
    args.out_dir = args.out_dir.resolve()
    args.session = args.session.resolve()
    sess = args.session
    cal = sorted((sess / "calibration").glob("hires_cam*_params.mat"),
                 key=lambda p: int("".join(c for c in p.stem if c.isdigit())))
    cams = [load_dannce_cam(p) for p in cal]
    names = [f"Camera{i+1}" for i in range(len(cams))]
    P = [c["K"] @ np.hstack([c["R"], c["tvec"].reshape(3, 1)]) for c in cams]

    d = pickle.load(open(sess / "COM" / "predict00" / "com3d.pickle", "rb"))
    keys = sample_order(d.keys())[:: args.stride]
    if args.max_frames:
        keys = keys[: args.max_frames]

    A = int(np.asarray(d[keys[0]][names[0]]["COM"]).shape[0])
    args.out_dir.mkdir(parents=True, exist_ok=True)
    toml_path = args.out_dir / f"{sess.name}.toml"
    write_toml(cams, names, args.size, toml_path)

    rng = np.random.default_rng(args.seed)
    coords, index, off = [], [], 0
    n_missing = 0
    for k in keys:
        s = d[k]
        coms = [np.asarray(s[n]["COM"], dtype=float) for n in names]

        # Minimum pairwise 3D separation, from the CHANNEL grouping -- used only to
        # stratify difficulty, never to inform the tracker.
        X = [triangulate_channel(coms, P, a) for a in range(A)]
        seps = [float(np.linalg.norm(X[i] - X[j]))
                for i in range(A) for j in range(i + 1, A)
                if X[i] is not None and X[j] is not None]
        sep = min(seps) if seps else None

        entry = {"i": int(k.split("_")[1]), "c": {}, "s": sep, "b": None}
        for ci, n in enumerate(names):
            insts = []
            for a in range(A):
                xy = coms[ci][a]
                if not np.all(np.isfinite(xy)):
                    n_missing += 1
                    continue
                insts.append((a, xy))
            # IDENTITY STRIPPING: the row index IS s-DANNCE's identity, so shuffle
            # it away before the tracker ever sees the frame.
            rng.shuffle(insts)
            lst = []
            for a, xy in insts:
                coords.extend([float(xy[0]), float(xy[1])])
                lst.append([int(a), off])
                off += 1
            entry["c"][n] = lst
        index.append(entry)

    arr = np.asarray(coords, dtype=np.float64)
    assert arr.size == off * 2, (arr.size, off)
    stem = args.out_dir / (f"{sess.name}_com" + (f"_{args.tag}" if args.tag else ""))
    arr.tofile(str(stem) + ".bin")
    meta = {
        "session": sess.name, "source": "triads_com", "cameras": names,
        "n_animals": A, "n_nodes": 1, "n_instances": off,
        "binary": str(stem) + ".bin", "calibration": str(toml_path),
        "tag": args.tag, "frames": index,
    }
    (stem.parent / (stem.name + ".json")).write_text(json.dumps(meta))
    print(f"{sess.name}: {len(index)} frames, {A} animals, {len(names)} cams, "
          f"{off} detections ({n_missing} missing), -> {stem.name}.json+.bin")


if __name__ == "__main__":
    main()
