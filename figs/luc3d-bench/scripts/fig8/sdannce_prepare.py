"""Fig 8d: turn a social-DANNCE BEDDING session into the inputs LUC3D's cross-view
tracker takes -- an anipose calibration and identity-stripped per-camera 2D.

Two 2D sources, deliberately kept apart because their ground truth is different:

  human      the *_RAT{1,2}_Label3D_dannce.mat hand labels. `data_2d` is (frames,
             23*2) PER CAMERA, NaN-free, and the animal is named by the filename,
             so identity ground truth is a human's. But RAT1 and RAT2 are labelled
             on largely DIFFERENT frames -- only the intersection is usable for a
             two-animal association test, and that is ~11 frames per session.
             Audit-grade n, gold-grade truth.

  projected  s-DANNCE's own `save_data_AVG0.mat` 3D (90,000 frames x 2 animals x
             23 keypoints) projected into all six cameras. Full session length, but
             the identity ground truth is s-DANNCE's channel assignment -- which is
             "identity by construction" and is exactly the thing that has never been
             audited. Panel-grade n, weaker truth. The `human` set audits it.

Projected 2D is geometrically consistent BY CONSTRUCTION and is therefore an
easier input than any real detector produces. That is not hidden -- it is why
`--noise-px`, `--dropout` and `--occlude-views` exist, and why panel e sweeps them.

Calibration: DANNCE `hires_camN_params.mat` is MATLAB-convention (K transposed,
row-vector points, t in mm) -> anipose .toml. The conversion is VERIFIED, not
assumed: reprojecting the humans' own `data_3d` must land on their own `data_2d`.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import scipy.io as sio


# --------------------------------------------------------------------------- #
def load_dannce_cam(path: Path) -> dict:
    m = sio.loadmat(path)
    K = np.asarray(m["K"], dtype=np.float64).T          # MATLAB stores K transposed
    R = np.asarray(m["r"], dtype=np.float64)
    t = np.asarray(m["t"], dtype=np.float64).ravel()
    rd = np.asarray(m["RDistort"], dtype=np.float64).ravel()
    td = np.asarray(m["TDistort"], dtype=np.float64).ravel()
    dist = np.array([rd[0], rd[1], td[0], td[1], 0.0], dtype=np.float64)
    # DANNCE projects row vectors: x_cam = X @ R + t. OpenCV uses column vectors
    # (x_cam = R_cv @ X + t), so R_cv = R.T.
    R_cv = R.T
    rvec, _ = cv2.Rodrigues(R_cv)
    return {"K": K, "R": R_cv, "rvec": rvec.ravel(), "tvec": t, "dist": dist}


def project(pts3d: np.ndarray, cam: dict) -> np.ndarray:
    """(N,3) mm -> (N,2) px, full distortion."""
    ok = np.all(np.isfinite(pts3d), axis=1)
    out = np.full((len(pts3d), 2), np.nan)
    if ok.any():
        p, _ = cv2.projectPoints(pts3d[ok].reshape(-1, 1, 3), cam["rvec"],
                                 cam["tvec"], cam["K"], cam["dist"])
        out[ok] = p.reshape(-1, 2)
    return out


def write_toml(cams: dict, size: tuple[int, int], out: Path) -> None:
    lines = []
    for i, (name, c) in enumerate(cams.items()):
        lines += [
            f"[cam_{i}]",
            f'name = "{name}"',
            f"size = [{size[0]}, {size[1]}]",
            f"matrix = {[list(map(float, r)) for r in c['K']]}",
            f"distortions = {list(map(float, c['dist']))}",
            f"rotation = {list(map(float, c['rvec']))}",
            f"translation = {list(map(float, c['tvec']))}",
            "",
        ]
    out.write_text("\n".join(lines))


# --------------------------------------------------------------------------- #
def reshape_labels(flat: np.ndarray, nkp: int, dim: int) -> np.ndarray:
    """(F, nkp*dim) -> (F, nkp, dim), trying both MATLAB orderings and letting the
    caller pick by reprojection error."""
    F = flat.shape[0]
    return {
        "F-order": flat.reshape(F, dim, nkp).transpose(0, 2, 1),   # [d][kp]
        "C-order": flat.reshape(F, nkp, dim),                      # [kp][d]
    }


def verify_calibration(session: Path, cams: dict, nkp: int = 23) -> dict:
    """Reproject the humans' own data_3d onto their own data_2d. Returns the best
    (order3d, order2d) and its median error in px."""
    lab = sorted((session / "labels").glob("*_RAT*_Label3D_dannce.mat"))
    if not lab:
        return {}
    m = sio.loadmat(lab[0])
    ld = m["labelData"]
    names = [str(c[0]) for c in m["camnames"][0]]

    best = None
    for o3 in ("F-order", "C-order"):
        for o2 in ("F-order", "C-order"):
            errs = []
            for ci, cname in enumerate(names):
                e = ld[ci, 0]
                d3 = reshape_labels(np.asarray(e["data_3d"][0, 0]), nkp, 3)[o3]
                d2 = reshape_labels(np.asarray(e["data_2d"][0, 0]), nkp, 2)[o2]
                proj = project(d3.reshape(-1, 3), cams[cname])
                errs.append(np.linalg.norm(proj - d2.reshape(-1, 2), axis=1))
            med = float(np.nanmedian(np.concatenate(errs)))
            if best is None or med < best["median_px"]:
                best = {"order3d": o3, "order2d": o2, "median_px": med}
    return best


# --------------------------------------------------------------------------- #
def build_projected(session: Path, cams: dict, order: dict, args) -> dict:
    m = sio.loadmat(session / "sdannce" / "save_data_AVG0.mat")
    pred = np.asarray(m["pred"])            # (T, A, 3, K)
    sid = np.asarray(m["sampleID"]).ravel()
    T, A, _, K = pred.shape
    idx = np.arange(0, T, args.stride)[: args.max_frames or None]

    # --bouts: restrict to contiguous CLOSE-INTERACTION episodes. Pooled over a
    # whole session the two rats are >200 mm apart most of the time, so cross-view
    # association is trivial and no amount of added noise makes it fail -- the
    # first sweep returned a flat 1.000 out to 40 px. The bouts are where the
    # problem actually lives, and they are 1-8.5% of frames depending on session.
    # Each bout is padded so the tracker enters it with identity already
    # established, which is the realistic condition.
    bout_of = None
    if args.bouts:
        cent_all = np.nanmean(pred, axis=3)                 # (T, A, 3)
        sep_all = np.linalg.norm(cent_all[:, 0] - cent_all[:, 1], axis=1)
        close = sep_all < args.bouts
        d = np.diff(close.astype(int))
        starts = np.flatnonzero(d == 1) + 1
        ends = np.flatnonzero(d == -1) + 1
        if close[0]:
            starts = np.r_[0, starts]
        if close[-1]:
            ends = np.r_[ends, len(close)]
        keep, bout_of, bid = [], {}, 0
        for a, b in zip(starts, ends):
            if b - a < args.bout_min_len:
                continue
            lo, hi = max(0, a - args.bout_pad), min(T, b + args.bout_pad)
            for f in range(lo, hi):
                if f not in bout_of:
                    bout_of[f] = bid
                    keep.append(f)
            bid += 1
        idx = np.array(sorted(keep), dtype=int)
        print(f"  bouts: {bid} episodes (<{args.bouts:.0f} mm, >={args.bout_min_len} f, "
              f"pad {args.bout_pad}) -> {len(idx)} frames")

    rng = np.random.default_rng(args.seed)
    frames = []
    for fi in idx:
        # Inter-animal 3D separation for this frame, from the UNDEGRADED truth.
        # Cross-view association is only hard when the animals are close; without
        # this axis a near-perfect pooled score just means the rats were mostly
        # far apart, which is a fact about the recording and not about the tracker.
        cent = [np.nanmean(pred[fi, a].T, axis=0) for a in range(A)]
        sep = float(np.linalg.norm(cent[0] - cent[1])) if A == 2 else float("nan")
        per_cam = {}
        for cname, cam in cams.items():
            insts = []
            for a in range(A):
                X = pred[fi, a].T                        # (K, 3)
                if not np.all(np.isfinite(X)):
                    continue
                xy = project(X, cam)
                if args.noise_px > 0:
                    xy = xy + rng.normal(0, args.noise_px, xy.shape)
                if args.dropout > 0:
                    drop = rng.random(len(xy)) < args.dropout
                    xy[drop] = np.nan
                if np.all(np.isnan(xy)):
                    continue
                insts.append({"id": int(a),
                              "pts": [[None, None] if not np.isfinite(p).all()
                                      else [float(p[0]), float(p[1])] for p in xy]})
            if args.occlude_views > 0 and rng.random() < args.occlude_views:
                insts = []
            # IDENTITY STRIPPING: shuffle so instance ORDER carries no identity.
            rng.shuffle(insts)
            per_cam[cname] = insts
        frames.append({"idx": int(sid[fi]) if fi < len(sid) else int(fi),
                       "cams": per_cam, "sep": sep,
                       "bout": (bout_of[int(fi)] if bout_of is not None else None)})
    return {"source": "projected", "cameras": list(cams), "n_animals": A,
            "n_nodes": K, "frames": frames,
            "noise_px": args.noise_px, "dropout": args.dropout,
            "occlude_views": args.occlude_views, "tag": args.tag}


def build_human(session: Path, cams: dict, order: dict, args, nkp: int = 23) -> dict:
    labs = sorted((session / "labels").glob("*_RAT*_Label3D_dannce.mat"))
    per_rat = {}
    cam_names = None
    for p in labs:
        rat = 0 if "RAT1" in p.name else 1
        m = sio.loadmat(p)
        ld = m["labelData"]
        cam_names = [str(c[0]) for c in m["camnames"][0]]
        sids = np.asarray(ld[0, 0]["data_sampleID"][0, 0]).ravel()
        d2 = {}
        for ci, cname in enumerate(cam_names):
            e = ld[ci, 0]
            d2[cname] = reshape_labels(np.asarray(e["data_2d"][0, 0]), nkp, 2)[order["order2d"]]
        per_rat[rat] = {"sids": sids, "d2": d2}

    if len(per_rat) < 2:
        return {"source": "human", "cameras": list(cams), "n_animals": 0,
                "n_nodes": nkp, "frames": []}

    shared = np.intersect1d(per_rat[0]["sids"], per_rat[1]["sids"])
    rng = np.random.default_rng(args.seed)
    frames = []
    for s in shared:
        per_cam = {}
        for cname in cam_names:
            insts = []
            for rat in (0, 1):
                k = int(np.where(per_rat[rat]["sids"] == s)[0][0])
                xy = per_rat[rat]["d2"][cname][k]
                if not np.isfinite(xy).any():
                    continue
                insts.append({"id": rat,
                              "pts": [[None, None] if not np.isfinite(p).all()
                                      else [float(p[0]), float(p[1])] for p in xy]})
            rng.shuffle(insts)
            per_cam[cname] = insts
        frames.append({"idx": int(s), "cams": per_cam})
    return {"source": "human", "cameras": list(cams), "n_animals": 2,
            "n_nodes": nkp, "frames": frames}


def write_binary(d: dict, out_stem: Path) -> None:
    """Split a prepared set into a float64 blob + a compact index.

    A full 90,000-frame session is ~25M keypoints. As JSON that is ~400 MB of text
    whose parse materialises ~25M boxed [x,y] arrays INSIDE V8's pointer-compressed
    heap -- the same scarce resource the save-path OOMs live on. The blob is one
    Float64Array the driver hands to Instance(), which adopts a flat Float64Array by
    reference, so no per-point object is ever allocated.
    """
    K = d["n_nodes"]
    rows = []            # (frameOrdinal, cam, trueId) in blob order
    coords = []
    index = []
    off = 0
    for fr in d["frames"]:
        entry = {"i": fr["idx"], "c": {}, "s": fr.get("sep"), "b": fr.get("bout")}
        for cam, insts in fr["cams"].items():
            lst = []
            for inst in insts:
                for p in inst["pts"]:
                    coords.append(np.nan if p[0] is None else p[0])
                    coords.append(np.nan if p[1] is None else p[1])
                lst.append([inst["id"], off])
                off += 1
            entry["c"][cam] = lst
        index.append(entry)
    arr = np.asarray(coords, dtype=np.float64)
    assert arr.size == off * K * 2, (arr.size, off, K)
    arr.tofile(str(out_stem) + ".bin")
    meta = {k: v for k, v in d.items() if k != "frames"}
    meta["n_instances"] = off
    meta["binary"] = str(out_stem) + ".bin"
    meta["frames"] = index
    (out_stem.parent / (out_stem.name + ".json")).write_text(json.dumps(meta))


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--stride", type=int, default=30)
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--noise-px", type=float, default=0.0)
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--occlude-views", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", type=str, default="")
    ap.add_argument("--bouts", type=float, default=0.0,
                    help="keep only close-interaction bouts, separation < this (mm)")
    ap.add_argument("--bout-pad", type=int, default=30)
    ap.add_argument("--bout-min-len", type=int, default=30)
    ap.add_argument("--binary", action="store_true",
                    help="emit a float64 blob + compact index instead of one big JSON")
    args = ap.parse_args()

    sess = args.session
    cal_files = sorted((sess / "calibration").glob("hires_cam*_params.mat"),
                       key=lambda p: int("".join(ch for ch in p.stem if ch.isdigit())))
    cams = {f"Camera{i+1}": load_dannce_cam(p) for i, p in enumerate(cal_files)}
    print(f"{sess.name}: {len(cams)} cameras")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    toml_path = args.out_dir / f"{sess.name}.toml"
    write_toml(cams, (1920, 1200), toml_path)

    order = verify_calibration(sess, cams)
    if order:
        print(f"  calibration check: reproject data_3d -> data_2d, "
              f"median {order['median_px']:.3f} px  ({order['order3d']}/{order['order2d']})")
        if order["median_px"] > 5.0:
            print("  *** WARNING: calibration conversion does not reproduce the "
                  "humans' own labels; do not trust anything downstream ***")
    else:
        order = {"order3d": "F-order", "order2d": "F-order"}
        print("  no human labels in this session; calibration UNVERIFIED")

    tag = f"_{args.tag}" if args.tag else ""
    for name, builder in (("human", build_human), ("projected", build_projected)):
        if name == "human" and not (sess / "labels").exists():
            continue
        d = builder(sess, cams, order, args)
        d["session"] = sess.name
        d["calibration"] = str(toml_path)
        if args.binary:
            stem = args.out_dir / f"{sess.name}_{name}{tag}"
            write_binary(d, stem)
            print(f"  wrote {stem.name}.json+.bin: {len(d['frames'])} frames, "
                  f"{d['n_animals']} animals")
        else:
            out = args.out_dir / f"{sess.name}_{name}{tag}.json"
            out.write_text(json.dumps(d))
            print(f"  wrote {out.name}: {len(d['frames'])} frames, {d['n_animals']} animals")


if __name__ == "__main__":
    main()
