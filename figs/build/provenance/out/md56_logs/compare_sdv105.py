#!/usr/bin/env python3
"""SDV105: old (lucid-switch-correct c9b2a325) vs new (LUC3D v1.0.5, 2dc001fe)
social-DANNCE numbers, for every s-DANNCE number in luc3d_newest.tex
(abstract L103 "88%", L423 Supp Sec, L485 Supp Fig 6 caption G) and the
Supp Fig 6G-I panel footnotes. Reads only; writes nothing.

    compare_sdv105.py OLD_AGG NEW_AGG
"""
import csv, json, statistics, sys
from pathlib import Path

OLD, NEW = Path(sys.argv[1]), Path(sys.argv[2])
FAM = {"triads": "TRIADS", "bedding": "BEDDING", "soc1": "SCN2A", "soc3": "SCN2A"}
DS = ["TRIADS", "BEDDING", "SCN2A"]
CAM = {2: "C7c_cams2_sigma0", 3: "C7c_cams3_sigma0", 4: "C7c_cams4_sigma0",
       5: "C7c_cams5_sigma0", 6: "C1_sigma0"}


def rows(agg, cell):
    out = {}
    for r in csv.DictReader(open(agg / "summary.csv")):
        if r["cell"] == cell:
            out[(r["dataset"], r["session_name"])] = dict(
                ds=FAM[r["dataset"]], idf1=float(r["idf1"]),
                sw=int(r["switches"]), n=int(r["n_gt_dets"]), frames=int(r["frames"]))
    return out


def swapdiag(agg):
    out = {}
    for f in sorted((agg / "swapdiag").glob("*.json")):
        d = json.loads(f.read_text())
        out[Path(d["session"]).name] = d
    return out


def numbers(agg):
    c1 = rows(agg, "C1_sigma0")
    v = list(c1.values())
    N = {"n_sessions": len(v)}
    zero = [x for x in v if x["sw"] == 0]
    N["zero_switch_sessions (36 of 41)"] = len(zero)
    N["zero_switch_pct (abstract 88%)"] = round(100 * len(zero) / len(v), 1)
    N["idf1_exactly_1.0 (35 of 41)"] = sum(x["idf1"] == 1.0 for x in v)
    N["idf1_>=0.9999"] = sum(x["idf1"] >= 0.9999 for x in v)
    for d in DS:
        s = [x for x in v if x["ds"] == d]
        N[f"{d} zero-switch (5/6, 5/6, 26/29)"] = f"{sum(x['sw'] == 0 for x in s)}/{len(s)}"
        N[f"{d} median IDF1 (1.0)"] = statistics.median(x["idf1"] for x in s)
        N[f"{d} min IDF1"] = min(x["idf1"] for x in s)
    N["total switches"] = sum(x["sw"] for x in v)
    sd = swapdiag(agg)
    swaps = []
    for (ds, sess), x in sorted(c1.items()):
        if x["sw"] == 0:
            continue
        d = sd.get(sess, {})
        ev = d.get("events", [])
        fr = d.get("switch_frames", [])
        swaps.append(dict(
            session=f"{ds}/{sess}", idf1=x["idf1"], switches=x["sw"],
            switch_frames=fr, frames=x["frames"],
            frac=[round(f / x["frames"], 3) for f in sorted(set(fr))],
            min_dist_within_50f_mm=min((e["min_dist_within_50f_mm"] for e in ev), default=None),
            reverts=d.get("reverts_in_session"),
            swapdiag_matches_scored=d.get("switch_count_matches_scored")))
    N["swap sessions (text: 4 within 6 mm, 1 at 45 mm; 0.970/0.972 near ends; "
      "0.792 TRIADS @41,298/60,000; 0.648 SCN2A @31,312/89,000)"] = swaps
    try:
        merge = json.loads((agg / "panel_10f_merge.json").read_text())
        clean = [m for m in merge if m["switches_sigma0"] == 0]
        N["frames<50mm in zero-switch sessions (68,560)"] = sum(m["frames_lt50mm"] for m in clean)
        N["frames<20mm in zero-switch sessions"] = sum(m["frames_lt20mm"] for m in clean)
    except FileNotFoundError:
        N["frames<50mm"] = "panel_10f_merge.json missing"
    try:
        c = json.loads((agg / "controls.json").read_text())
        N["controls sample"] = f"{c['session']} , first {c['frames']} frames, {c['animals']} animals, prep seed 99"
        N["controls baseline idf1/switches"] = (c["baseline"]["idf1"], c["baseline"]["switches"])
        N["control random-GT idf1 (0.500)"] = c["random_gt"]["idf1"]
        N["control splice idf1/switches (0.5 / 2)"] = (c["splice"]["idf1"], c["splice"]["switches"])
        N["control wrong-calib idf1 (0.513)"] = c["bad_calibration"]["idf1"]
        N["control wrong-calib switches/coverage"] = (c["bad_calibration"]["switches"],
                                                      c["bad_calibration"]["coverage"])
    except FileNotFoundError:
        N["controls"] = "controls.json missing"
    for k, cell in CAM.items():
        r = list(rows(agg, cell).values())
        if not r:
            continue
        dets = sum(x["n"] for x in r)
        N[f"6H k={k}: %perfect(>=0.9999) / pooled IDF1 / switches"] = (
            round(100 * sum(x["idf1"] >= 0.9999 for x in r) / len(r), 1),
            round(sum(x["idf1"] * x["n"] for x in r) / dets, 4),
            sum(x["sw"] for x in r))
    try:
        a = json.loads((agg / "c0_audit.json").read_text())
        N["c0 audit"] = {ds: (r["status"], r.get("effectiveConfig", {}).get("poseRoot"),
                              r.get("effectiveConfig", {}).get("tracker", {}).get("matchGate"))
                         for ds, r in a.items()}
    except FileNotFoundError:
        pass
    return N, c1


old, oc1 = numbers(OLD)
new, nc1 = numbers(NEW)
print(f"OLD = {OLD}\nNEW = {NEW}\n")
for k in list(dict.fromkeys(list(old) + list(new))):
    o, n = old.get(k, "-"), new.get(k, "-")
    flag = "" if o == n else "   <-- CHANGED"
    if isinstance(o, list) or isinstance(n, list):
        print(f"== {k}{flag}\n  OLD:")
        for e in (o if isinstance(o, list) else []):
            print("    ", e)
        print("  NEW:")
        for e in (n if isinstance(n, list) else []):
            print("    ", e)
    else:
        print(f"{k}\n    old: {o}\n    new: {n}{flag}")
print("\n== per-session C1_sigma0 changes (idf1, switches)")
ch = 0
for key in sorted(set(oc1) | set(nc1)):
    o, n = oc1.get(key), nc1.get(key)
    if not o or not n or o["idf1"] != n["idf1"] or o["sw"] != n["sw"]:
        ch += 1
        print(f"  {key[0]}/{key[1]}: old {o and (o['idf1'], o['sw'])}  new {n and (n['idf1'], n['sw'])}")
print(f"  {ch} of {len(set(oc1) | set(nc1))} sessions changed")
