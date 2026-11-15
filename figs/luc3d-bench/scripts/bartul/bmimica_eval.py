#!/usr/bin/env python
"""Stage 5: evaluate Liezl x6 + ByteTrack over all 50 BMimica sessions vs
points3d-reprojection GT. Computes BOTH metrics per session:
  - within-view: per-camera 2D IDF1 (mean over cams) + raw switches (sum)
  - cross-view : pooled one-global-id IDF1 + switches (Liezl=global id;
                 ByteTrack=camera-scoped id, so it fragments across views)
Aggregates over sessions -> CSV + figure. Run with .venv python (motmetrics)."""
import os, glob, sys, json, argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np

REPO = "/root/vast/eric/luc3d-bench"
sys.path.insert(0, f"{REPO}/scripts")
import evaluate as ev
import motmetrics as mm

BM = f"{REPO}/outputs/bmimica"
SER = ["21241563", "21369048", "21372315", "21372316", "22085397"]
LIEZL = f"{BM}/results/liezl_c6"
BYTE = f"{BM}/results/bytetrack"
MUP = f"{BM}/results/muppet"
GT = f"{BM}/gt"; DET = f"{BM}/det_h5"


def score_session(sid):
    out = {"session": sid}
    # load per-cam GT + det once
    gt = {}; det = {}
    for c in SER:
        gp = f"{GT}/{sid}/{c}/proofread.analysis.h5"; dp = f"{DET}/{sid}/{c}_predictions.h5"
        if not (os.path.exists(gp) and os.path.exists(dp)):
            return None
        import h5py
        gt[c] = np.transpose(h5py.File(gp)["tracks"][:], (3, 0, 2, 1))
        det[c] = h5py.File(dp)["tracks"][0]
    # LUC3D-schema JSONs (global id): Liezl + 3D-MuPPET
    liezl = json.load(open(f"{LIEZL}/{sid}.json")) if os.path.exists(f"{LIEZL}/{sid}.json") else None
    muppet = json.load(open(f"{MUP}/{sid}.json")) if os.path.exists(f"{MUP}/{sid}.json") else None
    jsons = {"liezl": liezl, "muppet": muppet}

    def run(kind):
        percam = []; pooled = mm.MOTAccumulator(auto_id=False); sw2d = 0
        for ci, c in enumerate(SER):
            g = gt[c]; d = det[c]
            nf = min(g.shape[0], d.shape[0])
            if kind in ("liezl", "muppet"):
                ids = ev.luc3d_assignments_for_cam(jsons[kind], c, nf, d.shape[1])
            elif kind == "byte":
                import h5py
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
                if kind in ("liezl", "muppet"):
                    for a in range(d.shape[1]):
                        b = ev.bbox_from_kpts(d[fi, a])
                        if b is None: continue
                        gid = int(ids[fi, a])
                        if gid < 0: continue
                        prb.append(b); pl.append(gid); pg.append(gid)
                else:
                    for a in range(min(d.shape[1], barr.shape[1])):
                        row = barr[fi, a]
                        if not np.isfinite(row[0]): continue
                        prb.append(row[1:5]); pl.append(int(row[0])); pg.append(ci*100000+int(row[0]))
                gn = np.array(gb) if gb else np.empty((0, 4))
                pn = np.array(prb) if prb else np.empty((0, 4))
                dist = mm.distances.iou_matrix(gn, pn, max_iou=0.5)
                pc.update(gi, pl, dist, frameid=fi)
                pooled.update(gi, pg, dist, frameid=ci*10**7+fi)
            s = mm.metrics.create().compute(pc, metrics=["idf1", "num_switches"], name="c")
            percam.append(s["idf1"]["c"]); sw2d += int(s["num_switches"]["c"])
        sp = mm.metrics.create().compute(pooled, metrics=["idf1", "num_switches"], name="p")
        return (float(np.mean(percam)), sw2d, float(sp["idf1"]["p"]), int(sp["num_switches"]["p"]))

    if liezl is not None:
        r = run("liezl")
        if r: out.update(dict(liezl_idf1=r[0], liezl_sw2d=r[1], liezl_xview_idf1=r[2], liezl_xview_sw=r[3]))
    if muppet is not None:
        rm = run("muppet")
        if rm: out.update(dict(muppet_idf1=rm[0], muppet_sw2d=rm[1], muppet_xview_idf1=rm[2], muppet_xview_sw=rm[3]))
    rb = run("byte")
    if rb: out.update(dict(byte_idf1=rb[0], byte_sw2d=rb[1], byte_xview_idf1=rb[2], byte_xview_sw=rb[3]))
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--workers", type=int, default=10); a = ap.parse_args()
    sids = sorted(os.path.basename(os.path.dirname(p)) for p in glob.glob(f"{GT}/*/"))
    sids = [s for s in sids if os.path.exists(f"{GT}/{s}/{SER[0]}/proofread.analysis.h5")]
    print(f"eval: {len(sids)} sessions, {a.workers} workers", flush=True)
    rows = []
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for f in as_completed([ex.submit(score_session, s) for s in sids]):
            r = f.result()
            if r: rows.append(r); print(f"  {r['session']}: liezl xview={r.get('liezl_xview_idf1','-')} byte xview={r.get('byte_xview_idf1','-')}", flush=True)
    import pandas as pd
    df = pd.DataFrame(rows); df.to_csv(f"{BM}/bmimica_eval_per_session.csv", index=False)
    # aggregate
    agg = {}
    for tr in ["liezl", "byte"]:
        for m in ["idf1", "xview_idf1"]:
            col = f"{tr}_{m}"
            if col in df: agg[col] = df[col].mean()
        for m in ["sw2d", "xview_sw"]:
            col = f"{tr}_{m}"
            if col in df: agg[col+"_total"] = df[col].sum()
    print("\n=== AGGREGATE over", len(df), "sessions ===")
    for k, v in agg.items(): print(f"  {k}: {v:.3f}" if 'idf1' in k else f"  {k}: {int(v)}")
    import json as J; J.dump(agg, open(f"{BM}/bmimica_eval_aggregate.json", "w"), indent=2)
    print("wrote", f"{BM}/bmimica_eval_per_session.csv")


if __name__ == "__main__":
    main()
