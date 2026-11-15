#!/usr/bin/env python
"""Stage 1 (tracking-only, tail-blind, uncapped) for the Stefan-model BMimica
rerun. Input: untracked poses from bmimica_infer_stefan.py. Output: the same
poses (all 15 nodes, tail included) with track identity assigned -- but the
identity decision itself never looked at the tail.

Why this needs a custom two-pass trick instead of a single CLI call:
sleap-nn's default (non-Kalman) tracker has no flag to down-weight or exclude
specific nodes from its association score -- only --use_kalman exposes
--kf_node_indices, and that switches the whole tracker to a constant-velocity
Kalman motion model, which is not what "tracking window of 30" (a
FixedWindowCandidates/OKS-tracker concept) was asking for.

Instead: verified sleap_nn.evaluation.compute_oks(points_gt, points_pr, ...)
treats a NaN node on the *first* (points_gt) argument as fully EXCLUDED (not
penalized) from the score, and tracker.py's Tracker.get_scores calls
`scoring_method(f, x.feature)` with `f` = the CURRENT frame's instance as
points_gt. So NaN-ing the tail nodes on every instance uniformly before
tracking makes the OKS association score ignore them completely, without
otherwise changing the algorithm.

Steps per (session, camera):
  1. Load the untracked .slp from bmimica_infer_stefan.py's output.
  2. Write a temp copy with Tail_0/Tail_1/Tail_2/TailTip coordinates (and
     scores) set to NaN on every instance.
  3. Run `sleap_nn.cli predict --data_path <temp> --tracking
     --tracking_window_size 30 --output_path <temp_tracked>` with NO
     --model_paths (pure-tracking retrack path) and NO
     --tracking_target_instance_count / --max_instances (no cap).
  4. Verify the tracked temp file has the same per-frame instance count/order
     as the input (matched on non-tail coordinates) -- tracking must not have
     reordered or dropped anything -- then copy each instance's assigned
     `.track` back onto the ORIGINAL full-node (tail included) instance.
  5. Save the merged result: full 15-node poses + tail-blind, uncapped track
     identity.

Output: /root/vast/eric/stefan-bartul-model/BMimica_tracked/{session}/{serial}.slp
"""
import glob
import os
import subprocess
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import sleap_io as sio

IN = "/root/vast/eric/stefan-bartul-model/BMimica_predictions"
OUT = "/root/vast/eric/stefan-bartul-model/BMimica_tracked"
SLEAP = "/root/vast/eric/sleap_nn_030_env/bin/python"
SER = ["21241563", "21369048", "21372315", "21372316", "22085397"]
TAIL_NODES = {"TailTip", "Tail_0", "Tail_1", "Tail_2"}
TRACKING_WINDOW = 30


def _tail_indices(skeleton):
    return [i for i, n in enumerate(skeleton.nodes) if n.name in TAIL_NODES]


def _make_notail_copy(labels, tail_idx, dst_path):
    """Deep-copy via re-saving isn't needed -- mutate point arrays in place on
    a freshly reloaded Labels so the original in-memory object is untouched."""
    for lf in labels.labeled_frames:
        for inst in lf.instances:
            pts = inst.points
            if hasattr(pts, "dtype") and pts.dtype.names:
                for i in tail_idx:
                    pts["xy"][i] = np.nan
                    if "score" in pts.dtype.names:
                        pts["score"][i] = np.nan
                    if "visible" in pts.dtype.names:
                        pts["visible"][i] = False
    labels.tracks = []
    labels.save(str(dst_path))


def _non_tail_xy(inst, tail_idx):
    xy = np.array(inst.points["xy"], dtype=np.float64)
    mask = np.ones(len(xy), dtype=bool)
    mask[tail_idx] = False
    return xy[mask]


def track_one(src, dst, tmp_dir):
    orig = sio.load_slp(src, open_videos=False)
    tail_idx = _tail_indices(orig.skeletons[0])
    if not tail_idx:
        raise RuntimeError(f"no tail nodes matched in skeleton for {src}")

    notail_src = os.path.join(tmp_dir, "notail.slp")
    notail_tracked = os.path.join(tmp_dir, "notail_tracked.slp")

    notail = sio.load_slp(src, open_videos=False)
    _make_notail_copy(notail, tail_idx, notail_src)

    cmd = [
        SLEAP, "-m", "sleap_nn.cli", "predict",
        "--data_path", notail_src,
        "--tracking", "--tracking_window_size", str(TRACKING_WINDOW),
        "--output_path", notail_tracked,
    ]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
    if r.returncode != 0 or not os.path.exists(notail_tracked):
        raise RuntimeError("tail-blind tracking failed: " + r.stderr[-500:])

    tracked = sio.load_slp(notail_tracked, open_videos=False)
    orig_by_frame = {lf.frame_idx: lf for lf in orig.labeled_frames}
    tracked_by_frame = {lf.frame_idx: lf for lf in tracked.labeled_frames}

    if set(orig_by_frame) != set(tracked_by_frame):
        raise RuntimeError(f"frame set mismatch after tracking for {src}")

    for fidx, orig_lf in orig_by_frame.items():
        tr_lf = tracked_by_frame[fidx]
        if len(orig_lf.instances) != len(tr_lf.instances):
            raise RuntimeError(
                f"instance count mismatch at frame {fidx} in {src}: "
                f"{len(orig_lf.instances)} vs {len(tr_lf.instances)}"
            )
        for orig_inst, tr_inst in zip(orig_lf.instances, tr_lf.instances):
            a = _non_tail_xy(orig_inst, tail_idx)
            b = _non_tail_xy(tr_inst, tail_idx)
            close = np.isclose(a, b, equal_nan=True)
            if not np.all(close):
                raise RuntimeError(
                    f"instance order mismatch at frame {fidx} in {src} "
                    "(non-tail coordinates don't match after tracking -- "
                    "positional merge is unsafe here)"
                )
            orig_inst.track = tr_inst.track

    orig.tracks = tracked.tracks
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    orig.save(dst)
    return {"n_frames": len(orig.labeled_frames), "n_tracks": len(orig.tracks)}


def job(args):
    sid, cam = args
    src = f"{IN}/{sid}/{cam}.slp"
    dst = f"{OUT}/{sid}/{cam}.slp"
    if not os.path.exists(src):
        return (sid, cam, "no_src")
    if os.path.exists(dst) and os.path.getsize(dst) > 1000:
        return (sid, cam, "skip_done")
    with tempfile.TemporaryDirectory(prefix=f"notail_{sid}_{cam}_") as tmp_dir:
        try:
            info = track_one(src, dst, tmp_dir)
            return (sid, cam, f"ok:{info}")
        except Exception as e:
            return (sid, cam, "ERR:" + str(e)[:300])


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--only-session", default=None)
    ap.add_argument("--only-sessions", nargs="+", default=None)
    ap.add_argument("--only-cam", default=None)
    args = ap.parse_args()
    sids = sorted(os.path.basename(d) for d in glob.glob(f"{IN}/*") if os.path.isdir(d))
    if args.only_session:
        sids = [s for s in sids if s == args.only_session]
    if args.only_sessions:
        wanted = set(args.only_sessions)
        sids = [s for s in sids if s in wanted]
    cams = SER if not args.only_cam else [args.only_cam]
    jobs = [(s, c) for s in sids for c in cams]
    print(f"sessions={len(sids)} jobs={len(jobs)} workers={args.workers}", flush=True)
    done = fail = skip = 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(job, j) for j in jobs]
        for i, f in enumerate(as_completed(futs)):
            sid, cam, st = f.result()
            if st == "skip_done":
                skip += 1
            elif st.startswith("ok"):
                done += 1
            else:
                fail += 1
                print(f"  [{sid}/{cam}] {st}", flush=True)
            if (i + 1) % 5 == 0 or not (st == "skip_done" or st.startswith("ok")):
                print(f"  progress {i+1}/{len(jobs)}  ok={done} skip={skip} fail={fail}", flush=True)
    print(f"DONE ok={done} skip={skip} fail={fail} / {len(jobs)}", flush=True)


if __name__ == "__main__":
    main()
