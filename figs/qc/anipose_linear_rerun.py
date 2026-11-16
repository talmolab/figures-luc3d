#!/usr/bin/env python3
"""Re-run the per-session LINEAR accuracy pass of `figs/fig2_solvers_anipose.py`
(its `_block`, unchanged and imported, not copied) to recover the per-session pooled
samples that `fig4_anipose*.json` folds into `reproj_px` / `heldout_reproj_px` but
does not store.

Why: those two pooled percentile blocks are computed over a 20k-per-session random
sample (seed 0 per session), and the deposit keeps only the percentiles. Dropping a
session from them needs the samples. Anipose/aniposelib is the baseline, not LUC3D,
so re-running it is allowed; no LUC3D code runs here. Timing and the optim pass are
NOT re-run.

Writes an .npz with, per session (input block order): the row `_block` returns
(minus the samples) as JSON, and the two float32 samples. `figs/qc/reagg_fig3.py`
checks those rows against the deposit's `per_session` field for field, pools ALL
sessions and asserts the deposit's percentiles come back, then pools the kept ones.

Must run in the anipose env the deposit names (`python` field):
  OMP_NUM_THREADS=1 /root/vast/eric/luc3d-bench/anipose072_cv413_env/bin/python \
      figs/qc/anipose_linear_rerun.py <fig4_input_js.json> <out.npz> [--jobs N]
"""
import argparse
import json
import multiprocessing
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

FIGS = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(FIGS))
import fig2_solvers_anipose as fa  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("meta", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--jobs", type=int, default=16)
    ap.add_argument("--stride", type=int, default=1)
    a = ap.parse_args()
    meta = json.loads(a.meta.read_text())
    bin_path = a.meta.parent / meta["bin"]
    n = len(meta["blocks"])
    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=a.jobs, initializer=fa._init,
                             mp_context=multiprocessing.get_context("spawn"),
                             initargs=(str(a.meta), str(bin_path))) as ex:
        rows = list(ex.map(fa._block, [(i, a.stride) for i in range(n)]))
    print(f"{n} sessions in {time.perf_counter() - t0:.1f}s")
    err = [np.asarray(r.pop("_err_sample"), dtype=np.float32) for r in rows]
    ho = [np.asarray(r.pop("_ho_sample"), dtype=np.float32) for r in rows]
    a.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        a.out, rows=json.dumps(rows), meta=str(a.meta),
        err_len=np.array([len(x) for x in err]), err=np.concatenate(err),
        ho_len=np.array([len(x) for x in ho]), ho=np.concatenate(ho))
    print(f"wrote {a.out}")


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("NUMBA_NUM_THREADS", "1")
    main()
