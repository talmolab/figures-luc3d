#!/usr/bin/env python
"""Cross-view identity-consistency metric (per scripts/bartul/cross_view_metric.py),
extended over ALL 50 runnable BMimica sessions (the "Bartul" 1-session smoke test is
itself one of these 50 — Bartul and BMimica are the same rig/dataset family).

Four trackers, scored per session then aggregated:
  - LUC3D        (real talmolab/luc3d tracker via bench_crossview.mjs; global id)
  - 3D-MuPPET    (SORT + first-frame cross-view match; global id)
  - SLEAP per-camera 2D (ref)  (native SLEAP track id; camera-scoped)
  - ByteTrack    (per-camera motion/IoU tracker; camera-scoped)

The old standalone python "Liezl" reimplementation and the old broken flat-JS
"LUC3D (3D)" snapshot are retired — the real repo now IS the Liezl-model tracker.

Output: outputs/bmimica/bmimica_crossview_all_eval.csv (per session, per tracker).
"""
import os, glob, sys, json, argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np

REPO = "/root/vast/eric/luc3d-bench"
sys.path.insert(0, f"{REPO}/scripts")
import evaluate as ev
import motmetrics as mm
import h5py

BM = f"{REPO}/outputs/bmimica"
SER = ["21241563", "21369048", "21372315", "21372316", "22085397"]
LUC3D = f"{BM}/results/luc3d_real"
MUP = f"{BM}/results/muppet"
BYTE = f"{BM}/results/bytetrack"
GT = f"{BM}/gt"; DET = f"{BM}/det_h5"; SLEAP = f"{BM}/sleap_h5"


def score_session(sid):
    out = {"session": sid}
    gt = {}; det = {}; sleap_det = {}
    for c in SER:
        gp = f"{GT}/{sid}/{c}/proofread.analysis.h5"; dp = f"{DET}/{sid}/{c}_predictions.h5"
        sp = f"{SLEAP}/{sid}/{c}_predictions.h5"
        if not (os.path.exists(gp) and os.path.exists(dp) and os.path.exists(sp)):
            return None
        gt[c] = np.transpose(h5py.File(gp)["tracks"][:], (3, 0, 2, 1))
        det[c] = h5py.File(dp)["tracks"][0]
        sleap_det[c] = h5py.File(sp)["tracks"][0]

    luc3d_json = json.load(open(f"{LUC3D}/{sid}.json")) if os.path.exists(f"{LUC3D}/{sid}.json") else None
    muppet_json = json.load(open(f"{MUP}/{sid}.json")) if os.path.exists(f"{MUP}/{sid}.json") else None
    jsons = {"luc3d": luc3d_json, "muppet": muppet_json}

    def run(kind):
        percam = []; pooled = mm.MOTAccumulator(auto_id=False); sw2d = 0
        for ci, c in enumerate(SER):
            g = gt[c]
            if kind in ("luc3d", "muppet"):
                d = det[c]
            elif kind == "sleap":
                d = sleap_det[c]
            else:  # byte
                d = det[c]
            nf = min(g.shape[0], d.shape[0])
            if kind in ("luc3d", "muppet"):
                ids = ev.luc3d_assignments_for_cam(jsons[kind], c, nf, d.shape[1])
            elif kind == "byte":
                bp = f"{BYTE}/{sid}/{c}.h5"
                if not os.path.exists(bp): return None
                barr = h5py.File(bp)["tracks"][:nf]
            pc = mm.MOTAccumulator(auto_id=False)
            for fi in range(nf):
                gb, gi = [], []
                for t in range(g.shape[1]):
                    b = ev.bbox_from_kpts(g[fi, t])
                    if b is not None: gb.append(b); gi.append(t)
                prb, pl, pg = [], [], []
                if kind in ("luc3d", "muppet"):
                    for a in range(d.shape[1]):
                        b = ev.bbox_from_kpts(d[fi, a])
                        if b is None: continue
                        gid = int(ids[fi, a])
                        if gid < 0: continue
                        prb.append(b); pl.append(gid); pg.append(gid)
                elif kind == "sleap":
                    for a in range(d.shape[1]):
                        b = ev.bbox_from_kpts(d[fi, a])
                        if b is None: continue
                        prb.append(b); pl.append(a); pg.append(ci * 10 + a)   # camera-scoped
                else:  # byte
                    for a in range(min(d.shape[1], barr.shape[1])):
                        row = barr[fi, a]
                        if not np.isfinite(row[0]): continue
                        prb.append(row[1:5]); pl.append(int(row[0])); pg.append(ci * 100000 + int(row[0]))
                gn = np.array(gb) if gb else np.empty((0, 4))
                pn = np.array(prb) if prb else np.empty((0, 4))
                dist = mm.distances.iou_matrix(gn, pn, max_iou=0.5)
                pc.update(gi, pl, dist, frameid=fi)
                pooled.update(gi, pg, dist, frameid=ci * 10**7 + fi)
            s = mm.metrics.create().compute(pc, metrics=["idf1", "num_switches"], name="c")
            percam.append(s["idf1"]["c"]); sw2d += int(s["num_switches"]["c"])
        sp = mm.metrics.create().compute(pooled, metrics=["idf1", "num_switches"], name="p")
        return (float(np.mean(percam)), sw2d, float(sp["idf1"]["p"]), int(sp["num_switches"]["p"]))

    for kind in ("luc3d", "muppet", "sleap", "byte"):
        if kind in ("luc3d", "muppet") and jsons[kind] is None:
            continue
        r = run(kind)
        if r:
            out.update({f"{kind}_idf1": r[0], f"{kind}_sw2d": r[1],
                        f"{kind}_xview_idf1": r[2], f"{kind}_xview_sw": r[3]})
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--workers", type=int, default=12); a = ap.parse_args()
    sids = sorted(os.path.basename(os.path.dirname(p)) for p in glob.glob(f"{GT}/*/"))
    sids = [s for s in sids if os.path.exists(f"{GT}/{s}/{SER[0]}/proofread.analysis.h5")]
    print(f"eval: {len(sids)} sessions, {a.workers} workers", flush=True)
    rows = []
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for f in as_completed([ex.submit(score_session, s) for s in sids]):
            r = f.result()
            if r:
                rows.append(r)
                print(f"  {r['session']}: luc3d xview={r.get('luc3d_xview_idf1','-')} "
                      f"sleap xview={r.get('sleap_xview_idf1','-')} byte xview={r.get('byte_xview_idf1','-')}",
                      flush=True)
    import pandas as pd
    df = pd.DataFrame(rows)
    df.to_csv(f"{BM}/bmimica_crossview_all_eval.csv", index=False)
    agg = {}
    for tr in ["luc3d", "muppet", "sleap", "byte"]:
        for m in ["idf1", "xview_idf1"]:
            col = f"{tr}_{m}"
            if col in df: agg[col] = df[col].mean()
        for m in ["sw2d", "xview_sw"]:
            col = f"{tr}_{m}"
            if col in df: agg[col + "_total"] = df[col].sum()
    print("\n=== AGGREGATE over", len(df), "sessions ===")
    for k, v in agg.items():
        print(f"  {k}: {v:.3f}" if "idf1" in k else f"  {k}: {int(v)}")
    import json as J
    agg = {k: (int(v) if "idf1" not in k else float(v)) for k, v in agg.items()}
    J.dump(agg, open(f"{BM}/bmimica_crossview_all_aggregate.json", "w"), indent=2)
    print("wrote", f"{BM}/bmimica_crossview_all_eval.csv")


if __name__ == "__main__":
    main()
