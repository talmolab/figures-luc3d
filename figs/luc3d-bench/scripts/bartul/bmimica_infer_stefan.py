#!/usr/bin/env python
"""Stage 0 (real inference, NOT filter+track on existing predictions): run
Stefan's top-down BMimica model (centroid + centered-instance, trained on this
exact 15-node skeleton/dataset) fresh on the raw BMimica videos.

Optimized for feeding LUC3D (per 2026-07-21 instructions):
  - Permissive detection: low peak thresholds so weak/extra skeletons survive
    (LUC3D's own cross-view association, not this filter, resolves identity).
  - OKS de-dup at 0.25 (more aggressive than the old 0.50 filter+track recipe)
    to actually drop true double-detections of the same animal.
  - NO instance-count cap: no --max_instances (over-culling "extra skeletons"
    is exactly what this rerun is trying to avoid).

This stage produces UNTRACKED poses only (no --tracking here). Tracking is a
separate stage (bmimica_track_stefan_notail.py) that (a) also runs with no
instance-count cap and (b) additionally zeroes out the tail nodes'
contribution to the tracking association score -- something the sleap-nn CLI
has no direct flag for outside of --use_kalman, so it needs its own script.
Splitting inference and tracking avoids doing (then discarding) a
tail-including tracking pass here.

Model checkpoints: /root/vast/eric/stefan-bartul-model/OptimizedModels/
  RunA.FR2.50_centroid.n=38206                       (centroid stage)
  260706_042933.centered_instance.n=38206_FR2.25b     (centered-instance stage)

Output: /root/vast/eric/stefan-bartul-model/BMimica_predictions/{session}/{serial}.slp
Resumable + parallel (one job per (session, camera); real neural-net inference,
so parallelism should be capped to the number of GPUs, not CPU cores -- see
--workers).
"""
import glob
import os
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed

ROOT = "/root/vast/eric/BMimica"
MODELS = "/root/vast/eric/stefan-bartul-model/OptimizedModels"
CENTROID = f"{MODELS}/RunA.FR2.50_centroid.n=38206"
INSTANCE = f"{MODELS}/260706_042933.centered_instance.n=38206_FR2.25b"
OUT = "/root/vast/eric/stefan-bartul-model/BMimica_predictions"
SLEAP = "/root/vast/eric/sleap_nn_030_env/bin/python"
SER = ["21241563", "21369048", "21372315", "21372316", "22085397"]

# Stefan's exact recipe (2026-07-21 screenshot + written instructions):
#   SLEAP_FILTER_MIN_INSTANCE_SCORE=0.2
#   SLEAP_FILTER_MIN_MEAN_NODE_SCORE=0.2
#   SLEAP_FILTER_MIN_VISIBLE_NODES=7
#   SLEAP_FILTER_OVERLAPPING_METHOD=oks / SLEAP_FILTER_OVERLAPPING_THRESHOLD=0.25
#   SLEAP_TRACKING_WINDOW_SIZE=30 (used in bmimica_track_stefan_notail.py)
# peak_threshold/centroid_peak_threshold aren't in Stefan's recipe -- left at
# the CLI default (0.2) rather than guessed.
MIN_INSTANCE_SCORE = 0.2
MIN_MEAN_NODE_SCORE = 0.2
MIN_VISIBLE_NODES = 7
OVERLAP_METHOD = "oks"
OVERLAP_THRESHOLD = 0.25


def runnable_sessions():
    out = []
    for s in sorted(glob.glob(ROOT + "/*")):
        sid = os.path.basename(s)
        if not (os.path.isdir(s) and sid[0].isdigit()):
            continue
        if not glob.glob(s + "/calibration/*_calibration.toml"):
            continue
        if not glob.glob(s + "/*points3d*.h5"):
            continue
        if sum(1 for c in SER if glob.glob(f"{s}/{c}/*.mp4")) != 5:
            continue
        out.append(sid)
    return out


def infer_job(args):
    sid, cam, gpu = args
    src = glob.glob(f"{ROOT}/{sid}/{cam}/*.mp4")
    if not src:
        return (sid, cam, "no_src")
    src = src[0]
    dstdir = f"{OUT}/{sid}"
    os.makedirs(dstdir, exist_ok=True)
    dst = f"{dstdir}/{cam}.slp"
    if os.path.exists(dst) and os.path.getsize(dst) > 1000:
        return (sid, cam, "skip_done")
    cmd = [
        SLEAP, "-m", "sleap_nn.cli", "predict",
        "--data_path", src,
        "--model_paths", CENTROID,
        "--model_paths", INSTANCE,
        "--device", gpu,
        "--filter_min_instance_score", str(MIN_INSTANCE_SCORE),
        "--filter_min_mean_node_score", str(MIN_MEAN_NODE_SCORE),
        "--filter_min_visible_nodes", str(MIN_VISIBLE_NODES),
        "--filter_overlapping",
        "--filter_overlapping_method", OVERLAP_METHOD,
        "--filter_overlapping_threshold", str(OVERLAP_THRESHOLD),
        "--output_path", dst,
    ]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=10800)
        if r.returncode != 0 or not os.path.exists(dst):
            return (sid, cam, "FAIL:" + r.stderr[-400:].replace("\n", " "))
        return (sid, cam, "ok")
    except Exception as e:
        return (sid, cam, "ERR:" + str(e)[:200])


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=2, help="one job per GPU by default")
    ap.add_argument("--gpus", nargs="+", default=["cuda:0", "cuda:1"])
    ap.add_argument("--only-session", default=None)
    ap.add_argument("--only-sessions", nargs="+", default=None)
    ap.add_argument("--only-cam", default=None)
    args = ap.parse_args()
    sids = runnable_sessions()
    if args.only_session:
        sids = [s for s in sids if s == args.only_session]
    if args.only_sessions:
        wanted = set(args.only_sessions)
        sids = [s for s in sids if s in wanted]
    cams = SER if not args.only_cam else [args.only_cam]
    jobs = [(s, c, args.gpus[i % len(args.gpus)]) for i, (s, c) in
            enumerate((s, c) for s in sids for c in cams)]
    print(f"sessions={len(sids)} jobs={len(jobs)} workers={args.workers} gpus={args.gpus}", flush=True)
    done = fail = skip = 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(infer_job, j) for j in jobs]
        for i, f in enumerate(as_completed(futs)):
            sid, cam, st = f.result()
            if st == "ok":
                done += 1
            elif st == "skip_done":
                skip += 1
            else:
                fail += 1
                print(f"  [{sid}/{cam}] {st}", flush=True)
            if (i + 1) % 5 == 0 or st not in ("ok", "skip_done"):
                print(f"  progress {i+1}/{len(jobs)}  ok={done} skip={skip} fail={fail}", flush=True)
    print(f"DONE ok={done} skip={skip} fail={fail} / {len(jobs)}", flush=True)


if __name__ == "__main__":
    main()
