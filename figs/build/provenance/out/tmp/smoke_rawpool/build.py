# Smoke test (2026-10-16, Eric): Mouse-Dyad pool with NO body rule at all, 5 most-affected sessions.
import sys, glob, os
sys.path.insert(0, "/root/vast/eric/sleap-3d-gui/scratch/repos/lucid/figs")
import bmimica_build_lite_pool as b
from concurrent.futures import ProcessPoolExecutor
S = ["20250907_174343", "20250907_143021", "20250904_140306", "20250908_161812", "20250829_162253"]
OUT = "/root/vast/eric/luc3d-bench/outputs/bmimica/det_h5_raw_smoke"
def job(a):
    sid, cam = a
    dst = f"{OUT}/{sid}/{cam}_predictions.h5"
    assert not os.path.exists(dst), dst
    src = sorted(glob.glob(f"{b.RAW}/{sid}/{cam}/*.slp"))[0]
    k, d = b.build(src, dst, None)
    return sid, cam, k
if __name__ == "__main__":
    with ProcessPoolExecutor(25) as ex:
        for sid, cam, k in ex.map(job, [(s, c) for s in S for c in b.SER]):
            print(sid, cam, "kept", k, flush=True)
