#!/usr/bin/env python
"""Stage 1: sleap-nn 0.3.0 FILTER+TRACK (the 'keeptrack' method) on all runnable
BMimica sessions × 5 cameras. Resumable (skips existing outputs), parallel.

Output: outputs/bmimica/retracked/{session}/{serial}.slp
Params match scripts/sleap_nn/retrack_one.py; target_instance_count=2 (2-animal).
"""
import os, glob, sys, subprocess, argparse
from concurrent.futures import ProcessPoolExecutor, as_completed

ROOT = "/root/vast/eric/BMimica"
OUT = "/root/vast/eric/luc3d-bench/outputs/bmimica/retracked"
SLEAP = os.environ.get("SLEAP_PY", "/root/vast/eric/sleap_nn_030_env/bin/python")
SER = ["21241563", "21369048", "21372315", "21372316", "22085397"]
TARGET = 2


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
        if sum(1 for c in SER if glob.glob(f"{s}/{c}/*.slp")) != 5:
            continue
        out.append(sid)
    return out


def retrack_job(args):
    sid, cam = args
    src = glob.glob(f"{ROOT}/{sid}/{cam}/*.slp")
    src = [p for p in src if p.endswith(".slp")]
    if not src:
        return (sid, cam, "no_src")
    src = src[0]
    dstdir = f"{OUT}/{sid}"
    os.makedirs(dstdir, exist_ok=True)
    dst = f"{dstdir}/{cam}.slp"
    if os.path.exists(dst) and os.path.getsize(dst) > 1000:
        return (sid, cam, "skip_done")
    cmd = [SLEAP, "-m", "sleap_nn.cli", "track", "--data_path", src,
           "--filter_min_instance_score", "0.85",
           "--filter_min_mean_node_score", "0.55",
           "--filter_min_visible_nodes", "8",
           "--filter_overlapping", "--filter_overlapping_method", "oks",
           "--filter_overlapping_threshold", "0.50",
           "--tracking", "--tracking_window_size", "15",
           "--tracking_target_instance_count", str(TARGET),
           "--post_connect_single_breaks",
           "--output_path", dst]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=5400)
        if r.returncode != 0 or not os.path.exists(dst):
            return (sid, cam, "FAIL:" + r.stderr[-300:].replace("\n", " "))
        return (sid, cam, "ok")
    except Exception as e:
        return (sid, cam, "ERR:" + str(e)[:200])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=20)
    ap.add_argument("--only-session", default=None)
    ap.add_argument("--only-cam", default=None)
    args = ap.parse_args()
    sids = runnable_sessions()
    if args.only_session:
        sids = [s for s in sids if s == args.only_session]
    jobs = [(s, c) for s in sids for c in (SER if not args.only_cam else [args.only_cam])]
    print(f"sessions={len(sids)} jobs={len(jobs)} workers={args.workers}", flush=True)
    done = fail = skip = 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(retrack_job, j) for j in jobs]
        for i, f in enumerate(as_completed(futs)):
            sid, cam, st = f.result()
            if st == "ok": done += 1
            elif st == "skip_done": skip += 1
            else: fail += 1; print(f"  [{sid}/{cam}] {st}", flush=True)
            if (i + 1) % 10 == 0 or st not in ("ok", "skip_done"):
                print(f"  progress {i+1}/{len(jobs)}  ok={done} skip={skip} fail={fail}", flush=True)
    print(f"DONE ok={done} skip={skip} fail={fail} / {len(jobs)}", flush=True)


if __name__ == "__main__":
    main()
