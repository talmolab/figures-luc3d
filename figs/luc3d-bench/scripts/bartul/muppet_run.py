#!/usr/bin/env python
"""3D-MuPPET tracking method on OUR shared detection pool (single session).

Faithful port of 3D-MuPPET/Tracking/TrackingInference.py's tracking half:
  - per-camera SORT (Bewley 2016, max_age=10)  [abewley/sort, vendored]
  - first-frame cross-view "dynamic matching" (Huang et al. 2020): per camera-pair
    triangulate every detection pair -> mean reprojection error -> Hungarian; then
    greedy-cluster the matched 3D points (TTI keypoint) within DminThresh into
    global identities.
  - maintain global ID via per-camera SORT track id; drop a camera's detection when
    its tracklet is lost/unmatched (paper: "skip the detections of the given camera").
We SKIP their YOLO+ViTPose detector and feed OUR canonical detection pool, so this is
apples-to-apples with Liezl/LUC3D/ByteTrack (shared detections). Emits the LUC3D JSON
schema consumed by scripts/evaluate.py + cross_view_metric.py.
"""
import os, sys, json, argparse, itertools
import numpy as np, cv2
from scipy.spatial import distance_matrix
from scipy.optimize import linear_sum_assignment
import h5py

REPO = "/root/vast/eric/luc3d-bench"
sys.path.insert(0, f"{REPO}/scripts/bartul")
sys.path.insert(0, f"{REPO}/outputs/muppet/Repositories/sort")
from sort import Sort
from build_gt_reproj import load_calibration   # K, dist, R, t, rvec, P per serial

TTI = 3  # central keypoint used for 3D clustering (their pigeon bottom-keel idx == 3)


def bbox_from_kps(kp, pad=5.0, min_valid=3):
    v = np.isfinite(kp[:, 0]) & np.isfinite(kp[:, 1])
    if v.sum() < min_valid:
        return None
    x, y = kp[v, 0], kp[v, 1]
    return np.array([x.min()-pad, y.min()-pad, x.max()+pad, y.max()+pad])


def iou(a, b):
    ix, iy = max(a[0], b[0]), max(a[1], b[1]); ax, ay = min(a[2], b[2]), min(a[3], b[3])
    if ax <= ix or ay <= iy:
        return 0.0
    inter = (ax-ix)*(ay-iy)
    return inter / ((a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter + 1e-9)


def undistort(pts, K, dist):
    out = pts.copy(); v = ~np.isnan(pts).any(1)
    if v.any():
        u = cv2.undistortPoints(pts[v].reshape(-1, 1, 2), K, dist, P=K).reshape(-1, 2)
        out[v] = u
    return out


def triangulate_pair(kp1u, kp2u, P1, P2):
    """Two-view DLT for a (15,2) undistorted keypoint pair -> (15,3)."""
    N = kp1u.shape[0]; out = np.full((N, 3), np.nan)
    for k in range(N):
        if np.isnan(kp1u[k]).any() or np.isnan(kp2u[k]).any():
            continue
        u1, v1 = kp1u[k]; u2, v2 = kp2u[k]
        A = np.array([u1*P1[2]-P1[0], v1*P1[2]-P1[1], u2*P2[2]-P2[0], v2*P2[2]-P2[1]])
        _, _, Vt = np.linalg.svd(A); X = Vt[-1]
        out[k] = X[:3]/X[3]
    return out


def reproj_err(X3d, kp_dist, cam):
    """mean px error between reprojected (distorted) 3D and the original distorted kp."""
    g = ~np.isnan(X3d).any(1)
    if g.sum() < 3:
        return np.inf
    pp, _ = cv2.projectPoints(X3d[g], cam["rvec"], cam["t"], cam["K"], cam["dist"])
    pp = pp.reshape(-1, 2)
    d = np.linalg.norm(pp - kp_dist[g], axis=1)
    return float(np.nanmean(d))


def matching_algorithm(kps_by_cam, cams, Dmin, inf_penalty=None):
    """kps_by_cam: {cam: list of (15,2) distorted keypoint arrays}. Returns
    {cam: {global_id: det_index or None}}. Port of MatchingAlgorithm().

    `inf_penalty` is OFF by default, which keeps this function bit-identical to the
    upstream port -- do not turn it on for the faithful arm. It exists for
    muppet_run_reid.py, which calls this at ARBITRARY frames rather than only at a
    hand-picked init frame where every camera has good detections. `reproj_err`
    returns `inf` for a detection with fewer than 3 valid keypoints, and one all-inf
    row makes `linear_sum_assignment` raise "cost matrix is infeasible" (hit on
    session 20250903_141046, 2026-08-17). When set, non-finite costs become this
    finite penalty -- the same 1e6 sentinel `err` is already initialised to, i.e.
    "no information" -- so the assignment stays feasible, and any pair MATCHED at or
    above the penalty is then dropped rather than allowed to inject a meaningless 3D
    point into the clustering."""
    cam_names = list(kps_by_cam.keys())
    und = {c: [undistort(kp, cams[c]["K"], cams[c]["dist"]) for kp in kps_by_cam[c]] for c in cam_names}
    points_list, index_list = [], []
    for c1, c2 in itertools.combinations(cam_names, 2):
        k1, k2 = kps_by_cam[c1], kps_by_cam[c2]
        u1, u2 = und[c1], und[c2]
        if len(k1) == 0 or len(k2) == 0:
            continue
        err = np.full((len(k1), len(k2)), 1e6, dtype=np.float64)
        p3d = {}
        for i in range(len(k1)):
            for j in range(len(k2)):
                X = triangulate_pair(u1[i], u2[j], cams[c1]["P"], cams[c2]["P"])
                e = 0.5*(reproj_err(X, k1[i], cams[c1]) + reproj_err(X, k2[j], cams[c2]))
                err[i, j] = e; p3d[(i, j)] = X
        if inf_penalty is not None:
            err = np.where(np.isfinite(err), err, float(inf_penalty))
        ri, ci = linear_sum_assignment(err)          # Hungarian (min reproj error)
        for i, j in zip(ri, ci):
            if inf_penalty is not None and err[i, j] >= inf_penalty:
                continue      # matched only because every option was uninformative
            X = p3d[(i, j)]
            if np.isnan(X[TTI]).any():
                continue
            points_list.append(X[TTI].tolist())
            index_list.append([(c1, i), (c2, j)])
    if len(points_list) < 2:
        # trivial: assign per camera in det order
        return {c: {g: (g if g < len(kps_by_cam[c]) else None) for g in range(max(len(v) for v in kps_by_cam.values()))} for c in cam_names}
    D = distance_matrix(points_list, points_list); np.fill_diagonal(D, np.inf)
    clusters = []; dmin = 0.0
    while dmin < Dmin:
        mi = np.unravel_index(D.argmin(), D.shape); dmin = D[mi]
        if dmin >= Dmin:
            break
        pair = set(index_list[mi[0]] + index_list[mi[1]])
        exist = set(x for x, S in enumerate(clusters) for p in S if p in pair)
        if len(exist) == 0:
            clusters.append(pair)
        elif len(exist) == 1:
            S = clusters[list(exist)[0]]; present = [p[0] for p in pair]
            for p in list(S):
                if p[0] in present:
                    for sp in [q for q in pair if q[0] == p[0]]:
                        pair.discard(sp)
            clusters[list(exist)[0]].update(pair)
        D[mi[0], mi[1]] = np.inf; D[mi[1], mi[0]] = np.inf
    final = {c: {} for c in cam_names}
    for gid, S in enumerate(clusters):
        for c in cam_names:
            idx = [p[1] for p in S if p[0] == c]
            final[c][gid] = idx[0] if idx else None
    return final


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--det-dir", default=f"{REPO}/outputs/bartul_smoke/det_h5")
    ap.add_argument("--calib", required=True)
    ap.add_argument("--cameras", required=True, help="comma-separated serials")
    ap.add_argument("--out", required=True)
    ap.add_argument("--num-animals", type=int, default=2)
    ap.add_argument("--session-idx", type=int, default=0,
                    help="session axis index into a multi-session detection pool H5 (default 0)")
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--dmin", type=float, default=100.0, help="Huang 3D match threshold mm (pigeon paper=200; mice smaller)")
    ap.add_argument("--warmup", type=int, default=5)
    args = ap.parse_args()

    SER = args.cameras.split(",")
    cams = load_calibration(args.calib)
    det = {c: h5py.File(f"{args.det_dir}/{c}_predictions.h5")["tracks"][args.session_idx] for c in SER}
    F = min(v.shape[0] for v in det.values())
    if args.max_frames:
        F = min(F, args.max_frames)

    sorters = {c: Sort(max_age=10) for c in SER}
    # per frame per cam: slot->track_id map + list of (slot, kp)
    slot2track = {c: [dict() for _ in range(F)] for c in SER}
    valid_slots = {c: [list() for _ in range(F)] for c in SER}
    ntracks = np.zeros((F, len(SER)), int)
    for fi in range(F):
        for ci, c in enumerate(SER):
            dets = []; slots = []
            for a in range(det[c].shape[1]):
                b = bbox_from_kps(det[c][fi, a])
                if b is not None:
                    dets.append(b); slots.append(a)
            valid_slots[c][fi] = list(zip(slots, dets))
            if dets:
                out = sorters[c].update(np.array(dets))
                ntracks[fi, ci] = len(out)
                # map each input det slot -> track id by best IoU
                for si, (slot, b) in enumerate(zip(slots, dets)):
                    best, bt = 0.3, None
                    for row in out:
                        j = iou(b, row[:4])
                        if j > best:
                            best, bt = j, int(row[4])
                    if bt is not None:
                        slot2track[c][fi][slot] = bt
            else:
                sorters[c].update(np.empty((0, 5)))

    # init frame: first frame >= warmup where every cam has >= num_animals tracks
    init = None
    for fi in range(args.warmup, F):
        if all(ntracks[fi, ci] >= args.num_animals for ci in range(len(SER))):
            init = fi; break
    if init is None:
        init = args.warmup
    # build MatchingDict at init frame (keypoints per cam, det order = valid slots order)
    kps_by_cam = {}; slotorder = {}
    for c in SER:
        vs = valid_slots[c][init]
        kps_by_cam[c] = [det[c][init, slot] for slot, _ in vs]
        slotorder[c] = [slot for slot, _ in vs]
    final = matching_algorithm(kps_by_cam, cams, args.dmin)

    # link det_index (in init frame) -> global id -> sort track id
    global_by_track = {c: {} for c in SER}
    for c in SER:
        for gid, di in final[c].items():
            if di is None or di >= len(slotorder[c]):
                continue
            slot = slotorder[c][di]
            tid = slot2track[c][init].get(slot)
            if tid is not None:
                global_by_track[c][tid] = gid
    print(f"init_frame={init}  global ids found per cam:",
          {c: sorted(set(global_by_track[c].values())) for c in SER}, flush=True)

    # emit: for every frame, slot -> track -> global id
    frames_out = []
    for fi in range(F):
        assigns = []
        for c in SER:
            for slot, _ in valid_slots[c][fi]:
                tid = slot2track[c][fi].get(slot)
                if tid is None:
                    continue
                gid = global_by_track[c].get(tid)
                if gid is None:
                    continue      # lost/unmatched tracklet -> skip this camera's detection
                assigns.append([f"{c}:{slot}", int(gid)])
        if assigns:
            frames_out.append({"frame": fi, "assignments": assigns})
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump({"cameras": SER, "numAnimals": args.num_animals, "frames": frames_out,
               "method": "3D-MuPPET (SORT+xview)", "dmin_mm": args.dmin, "init_frame": int(init)},
              open(args.out, "w"))
    print("wrote", args.out, f"({len(frames_out)} frames)", flush=True)


if __name__ == "__main__":
    main()
