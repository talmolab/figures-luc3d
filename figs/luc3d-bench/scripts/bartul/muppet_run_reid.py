#!/usr/bin/env python
"""3D-MuPPET + re-ID (OURS — not the published method).

The faithful port (muppet_run.py) reproduces upstream exactly: the cross-view
triangulation match runs ONCE at an init frame, then `global_by_track` (SORT track id
-> global id) is frozen forever. When any camera's SORT tracklet dies (very common
over an 18-minute, real-interaction session — SORT is a simple Kalman+IoU tracker,
max_age=10), that camera's detections are dropped for the rest of the session because
there's no mechanism to re-link a freshly-born replacement tracklet back to its
identity. That's upstream's own disclosed limitation ("no re-ID"), not a port bug —
see the faithfulness note in outputs/subgroup_7_7_2026/README.md.

This variant asks: what if we just re-ran MuPPET's OWN matching_algorithm() again
whenever a new, not-yet-linked SORT track id shows up, instead of only once? Every
time a camera has a live track with no known global id, we re-triangulate ALL cameras'
current detections (the same Huang-2020 dynamic-matching clustering upstream already
uses), then reconcile the fresh clusters against already-known identities: a fresh
cluster is folded into an EXISTING global id if any of its members already carry one
(i.e. some other camera is still tracking that identity continuously); only a cluster
with zero prior evidence gets a brand-new id, and only if fewer than num_animals ids
have been allocated so far (otherwise we can't tell which known identity it is, so we
conservatively drop it rather than guess).

A retry cooldown (default 30 frames) avoids re-attempting an unresolved track every
single frame once we've already tried it once.

Emits the same LUC3D JSON schema as muppet_run.py. Reports "method":
"3D-MuPPET + re-ID (ours)" so it's never confused with the faithful port's numbers.
"""
import os, sys, json, argparse
import numpy as np
import h5py

REPO = "/root/vast/eric/luc3d-bench"
sys.path.insert(0, f"{REPO}/scripts/bartul")
sys.path.insert(0, f"{REPO}/outputs/muppet/Repositories/sort")
from sort import Sort
from build_gt_reproj import load_calibration
from muppet_run import bbox_from_kps, iou, matching_algorithm

RETRY_COOLDOWN = 30  # frames before re-attempting an unresolved live track


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--det-dir", default=f"{REPO}/outputs/bartul_smoke/det_h5")
    ap.add_argument("--calib", required=True)
    ap.add_argument("--cameras", required=True, help="comma-separated serials")
    ap.add_argument("--out", required=True)
    ap.add_argument("--num-animals", type=int, default=2)
    ap.add_argument("--session-idx", type=int, default=0)
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--dmin", type=float, default=100.0)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--retry-cooldown", type=int, default=RETRY_COOLDOWN)
    args = ap.parse_args()

    SER = args.cameras.split(",")
    cams = load_calibration(args.calib)
    det = {c: h5py.File(f"{args.det_dir}/{c}_predictions.h5")["tracks"][args.session_idx] for c in SER}
    F = min(v.shape[0] for v in det.values())
    if args.max_frames:
        F = min(F, args.max_frames)

    sorters = {c: Sort(max_age=10) for c in SER}
    slot2track = {c: [dict() for _ in range(F)] for c in SER}
    valid_slots = {c: [list() for _ in range(F)] for c in SER}
    for fi in range(F):
        for c in SER:
            dets = []; slots = []
            for a in range(det[c].shape[1]):
                b = bbox_from_kps(det[c][fi, a])
                if b is not None:
                    dets.append(b); slots.append(a)
            valid_slots[c][fi] = list(zip(slots, dets))
            if dets:
                out = sorters[c].update(np.array(dets))
                for slot, b in zip(slots, dets):
                    best, bt = 0.3, None
                    for row in out:
                        j = iou(b, row[:4])
                        if j > best:
                            best, bt = j, int(row[4])
                    if bt is not None:
                        slot2track[c][fi][slot] = bt
            else:
                sorters[c].update(np.empty((0, 5)))

    global_by_track = {c: {} for c in SER}
    next_retry = {c: {} for c in SER}
    frames_out = []
    n_rematch = 0
    for fi in range(args.warmup, F):
        live = {c: set(slot2track[c][fi].values()) for c in SER}
        todo = {c: [t for t in live[c] if t not in global_by_track[c] and next_retry[c].get(t, -1) <= fi]
                for c in SER}
        if any(todo.values()):
            n_rematch += 1
            kps_by_cam = {}; slotorder = {}
            for c in SER:
                vs = valid_slots[c][fi]
                kps_by_cam[c] = [det[c][fi, slot] for slot, _ in vs]
                slotorder[c] = [slot for slot, _ in vs]
            # inf_penalty: this variant matches at arbitrary frames, where a
            # detection with < 3 valid keypoints can make the Hungarian infeasible
            final = matching_algorithm(kps_by_cam, cams, args.dmin,
                                       inf_penalty=1e6)

            members_by_local_gid = {}
            for c in SER:
                for gid_local, di in final[c].items():
                    if di is None or di >= len(slotorder[c]):
                        continue
                    slot = slotorder[c][di]
                    tid = slot2track[c][fi].get(slot)
                    if tid is None:
                        continue
                    members_by_local_gid.setdefault(gid_local, []).append((c, tid))

            for _, members in members_by_local_gid.items():
                known = {global_by_track[c_].get(t_) for c_, t_ in members if t_ in global_by_track[c_]}
                known.discard(None)
                if len(known) > 1:
                    continue  # conflicting evidence across cams -> don't guess, leave existing links alone
                if len(known) == 1:
                    gid = next(iter(known))
                else:
                    used = set()
                    for c_ in SER:
                        used |= set(global_by_track[c_].values())
                    avail = [g for g in range(args.num_animals) if g not in used]
                    if not avail:
                        continue  # all identity slots already allocated elsewhere -> can't tell which; skip
                    gid = avail[0]
                for c_, t_ in members:
                    global_by_track[c_][t_] = gid

            for c in SER:
                for t in todo[c]:
                    next_retry[c][t] = fi + args.retry_cooldown

        assigns = []
        for c in SER:
            for slot, _ in valid_slots[c][fi]:
                tid = slot2track[c][fi].get(slot)
                if tid is None:
                    continue
                gid = global_by_track[c].get(tid)
                if gid is None:
                    continue
                assigns.append([f"{c}:{slot}", int(gid)])
        if assigns:
            frames_out.append({"frame": fi, "assignments": assigns})

    print(f"rematch attempts: {n_rematch} over {F} frames", flush=True)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump({"cameras": SER, "numAnimals": args.num_animals, "frames": frames_out,
               "method": "3D-MuPPET + re-ID (ours)", "dmin_mm": args.dmin,
               "retry_cooldown": args.retry_cooldown},
              open(args.out, "w"))
    print("wrote", args.out, f"({len(frames_out)} frames)", flush=True)


if __name__ == "__main__":
    main()
