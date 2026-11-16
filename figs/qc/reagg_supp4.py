#!/usr/bin/env python3
"""q series, Supp Fig 4 family (repo fig6: within- vs cross-view identity, panels a-d):
re-aggregate the b deposits with the sessions in src/qc_exclude.py left out.

Deposits re-aggregated (b -> q):
  out/fig7_variant_best_b.json    Mouse-Dyad-10M block (`bmimica_sessions`,
                                  `bmimica_50_sessions`, `bmimica_wins`). Producer:
                                  figs/fig6_variant_best_rebuild.py (`luc_entry`,
                                  `fair_entry`, the wins loop) with `stats`/`sign_p` from
                                  figs/fig6_variant_tracker.py, imported, not copied.
                                  SLAP-2M blocks are not Mouse-Dyad and are carried over.
  out/fig7_fair_baselines_b.json  `sessions` + every arm. Producer: figs/fig6_fair_baselines.py
                                  (`arm`, `ci95` -- the seeded bootstrap, copied verbatim
                                  because they are closures inside its main()).

Every IDF1 aggregate is recomputed from the per-session arrays the b deposit holds, which
are positionally aligned to its session list and are filtered together with it. The
switch totals and SLEAP's unscoped cross-view mean are SUMS/MEANS over per-session values
the deposits do not store, so they are read from the producers' own per-session inputs:

  LUC3D, LUC3D (shipped)  out/fig8_methods_50_b.json cells sync_stale20_dist25 / stale0_dist25
                          (`within_switches`, `cross_switches`); the producer takes
                          `cell["switches"]`, which equals the sum of `within_switches`
                          over all 56 sessions (asserted below).
  SLEAP (both deposits)   <lucid>/figs/out/fig7_sleap_max2_c6b3.json  (= the `_c6b3nd`
                          name fig7_fair_baselines_b's `sources` gives; it is a symlink)
  ByteTrack, fair arms    <lucid>/figs/out/fig7_bytetrack_max2_c6b3nd.json  (B1s, B1)
  ByteTrack, variant_best <lucid>/figs/out/fig7_bytetrack_max2_c6b3.json  (B1s). NOTE: the
                          b build's variant_best `ByteTrack` arm is NOT the fair deposit's
                          B1s arm: it matches the pre-`nd` c6b3 ByteTrack run (within
                          0.8206, 1927 switches) while fig7_fair_baselines_b carries c6b3nd
                          (0.8251, 2039). The panel draws the fair arm's values and takes
                          only `bmimica_wins` from variant_best. That split is reproduced
                          here as found, not repaired.

Each source is used only after its per-session IDF1 arrays are proven equal, element for
element and in the deposit's session order, to the arrays the deposit holds.

    cd figs && $PY qc/reagg_supp4.py            # identity check, then write out/<stem>_q.json
    cd figs && $PY qc/reagg_supp4.py --check    # identity check only, write nothing

Identity check (always run first): the same code with an EMPTY exclusion list must
reproduce every field of each _b deposit EXACTLY (parsed JSON, floats ==).
"""
from __future__ import annotations

import copy
import json
import math
import sys
from pathlib import Path

import numpy as np

FIGS = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(FIGS))
from src.qc_exclude import excluded, provenance  # noqa: E402
from fig6_variant_tracker import sign_p, stats  # noqa: E402  (the producer's own)

OUT = FIGS / "out"
LUCID_OUT = Path("/root/vast/eric/sleap-3d-gui/scratch/repos/lucid/figs/out")
SLEAP_SRC = LUCID_OUT / "fig7_sleap_max2_c6b3.json"
BYTE_FAIR_SRC = LUCID_OUT / "fig7_bytetrack_max2_c6b3nd.json"
BYTE_VB_SRC = LUCID_OUT / "fig7_bytetrack_max2_c6b3.json"
F8_SRC = OUT / "fig8_methods_50_b.json"
TAG = " + figs/qc/reagg_supp4.py (qc_exclude)"

#: variant_best key -> fig8_methods_50 cell (its `luc3d_source` names these)
LUC_CELLS = {"LUC3D": "sync_stale20_dist25", "LUC3D (shipped)": "stale0_dist25"}
FAIR = {"SLEAP per-camera": "SLEAP per-camera (2 tracks enforced)",
        "ByteTrack": "ByteTrack (2 identities, B1s)"}


def never(_s):
    return False


def _rows(path):
    return {r["session"]: r for r in json.loads(Path(path).read_text())["per_session"]}


def _prove(name, got, want):
    if got != want:
        bad = sum(1 for a, b in zip(got, want) if a != b) + abs(len(got) - len(want))
        raise SystemExit(f"[reagg] {name}: source does not reproduce the deposit's "
                         f"per-session array ({bad} mismatches) -- refusing to use it")


# --------------------------------------------------------------------------- sources
def load_sources(vb, fb):
    """Per-session values the deposits do not store, keyed by session, after proving each
    source's IDF1 arrays equal the deposit's."""
    f8 = json.loads(F8_SRC.read_text())
    cells = {c["config"]: c for c in f8["cells"]}
    S = vb["bmimica_sessions"]
    src = {}
    for key, cfg in LUC_CELLS.items():
        c = cells[cfg]
        per = c["per_session"]
        _prove(f"f8 {cfg} sessions", [q["session"] for q in per], S)
        _prove(f"f8 {cfg} within", [q["within_idf1"] for q in per],
               vb["bmimica_50_sessions"][key]["per_session_within"])
        _prove(f"f8 {cfg} cross", [q["cross_idf1"] for q in per],
               vb["bmimica_50_sessions"][key]["per_session_cross"])
        if c["switches"] != sum(q["within_switches"] for q in per):
            raise SystemExit(f"[reagg] f8 {cfg}: switches != sum(within_switches)")
        src[key] = {q["session"]: {"sw2d": q["within_switches"], "swx": q["cross_switches"]}
                    for q in per}

    sl = _rows(SLEAP_SRC)
    bf = _rows(BYTE_FAIR_SRC)
    bv = _rows(BYTE_VB_SRC)
    FS = fb["sessions"]
    a = fb["arms"]["SLEAP per-camera (2 tracks enforced)"]
    _prove("sleap within", [sl[s]["within_idf1"] for s in FS], a["per_session_within"])
    _prove("sleap cross", [sl[s]["cross_idf1_scoped"] for s in FS], a["per_session_cross"])
    for arm in ("B1s", "B1"):
        a = fb["arms"]["ByteTrack (2 identities, B1s)" if arm == "B1s"
                       else "ByteTrack (own knob only, B1)"]
        _prove(f"bytetrack {arm} within", [bf[s][f"arm_{arm}"]["within_idf1"] for s in FS],
               a["per_session_within"])
        _prove(f"bytetrack {arm} cross",
               [bf[s][f"arm_{arm}"]["cross_idf1_scoped"] for s in FS], a["per_session_cross"])
    vbb = vb["bmimica_50_sessions"]
    _prove("vb sleap within", [sl[s]["within_idf1"] for s in S],
           vbb["SLEAP per-camera"]["per_session_within"])
    _prove("vb sleap cross", [sl[s]["cross_idf1_scoped"] for s in S],
           vbb["SLEAP per-camera"]["per_session_cross"])
    _prove("vb bytetrack within", [bv[s]["arm_B1s"]["within_idf1"] for s in S],
           vbb["ByteTrack"]["per_session_within"])
    _prove("vb bytetrack cross", [bv[s]["arm_B1s"]["cross_idf1_scoped"] for s in S],
           vbb["ByteTrack"]["per_session_cross"])
    src["SLEAP per-camera"] = {s: {"sw2d": sl[s]["within_switches"]} for s in sl}
    src["ByteTrack"] = {s: {"sw2d": bv[s]["arm_B1s"]["within_sw"]} for s in bv}
    src["fair"] = {
        "SLEAP per-camera (2 tracks enforced)": {
            s: {"sw2d": sl[s]["within_switches"], "unscoped": sl[s]["cross_idf1_unscoped"]}
            for s in sl},
        "ByteTrack (2 identities, B1s)": {s: {"sw2d": bf[s]["arm_B1s"]["within_sw"]}
                                          for s in bf},
        "ByteTrack (own knob only, B1)": {s: {"sw2d": bf[s]["arm_B1"]["within_sw"]}
                                          for s in bf},
    }
    return src


# --------------------------------------------------------------------------- variant_best
def reagg_variant_best(b, src, ex):
    """figs/fig6_variant_best_rebuild.py build(), Mouse-Dyad block, over the kept sessions."""
    out = copy.deepcopy(b)
    S = b["bmimica_sessions"]
    K = [i for i, s in enumerate(S) if not ex(s)]
    sess = [S[i] for i in K]
    out["bmimica_sessions"] = sess
    bm = out["bmimica_50_sessions"]
    if set(bm) != set(LUC_CELLS) | set(FAIR):
        raise SystemExit(f"[reagg] variant_best arms {sorted(bm)} -- unexpected")
    for key, e in bm.items():
        w = [e["per_session_within"][i] for i in K]
        c = [e["per_session_cross"][i] for i in K]
        e["within"], e["cross"] = stats(w), stats(c)
        e["switches_2d_total"] = sum(src[key][s]["sw2d"] for s in sess)
        if key in LUC_CELLS:                                   # luc_entry()
            e["switches_xview_total"] = sum(src[key][s]["swx"] for s in sess)
            e["per_session_within"], e["per_session_cross"] = w, c
            drift = sorted(abs(y - x) for x, y in zip(w, c))
            e["drift_abs_max"] = drift[-1]
            e["drift_abs_median"] = float(drift[len(drift) // 2])
        else:                                                  # fair_entry()
            e["per_session_within"], e["per_session_cross"] = w, c
        e["cross_over_within"]["mean"] = sum(c) / max(1e-12, sum(w))

    luc = {s: (x, y) for s, x, y in zip(sess, bm["LUC3D"]["per_session_within"],
                                        bm["LUC3D"]["per_session_cross"])}
    wins = out["bmimica_wins"]
    for key in FAIR:
        for i, lbl in enumerate(("within", "cross")):
            vals = bm[key][f"per_session_{lbl}"]
            k = sum(1 for s, v in zip(sess, vals) if luc[s][i] > v)
            wins[key][lbl] = {"wins": k, "n": len(vals), "sign_p": sign_p(k, len(vals))}
    sh = bm["LUC3D (shipped)"]
    for i, lbl in enumerate(("within", "cross")):
        shv = sh[f"per_session_{lbl}"]
        wins["LUC3D"][lbl] = {"wins": sum(1 for s, v in zip(sess, shv) if luc[s][i] > v),
                              "n": len(sess)}
    old = f"over {len(S)} sessions"
    if old not in out["generated_by"]:
        raise SystemExit("[reagg] variant_best generated_by lost its session count")
    out["generated_by"] = out["generated_by"].replace(old, f"over {len(sess)} sessions")
    return out


# --------------------------------------------------------------------------- fair baselines
def _ci95(v, n_boot=10000):
    """figs/fig6_fair_baselines.py main().ci95, verbatim."""
    rng = np.random.default_rng(20260814)
    idx = rng.integers(0, v.size, size=(n_boot, v.size))
    means = v[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def _arm_core(vals_within, vals_cross):
    """figs/fig6_fair_baselines.py main().arm, minus **extra."""
    w, c = np.asarray(vals_within, float), np.asarray(vals_cross, float)
    w_lo, w_hi = _ci95(w)
    c_lo, c_hi = _ci95(c)
    return {"within": {"mean": float(w.mean()), "median": float(np.median(w)),
                       "ci95_lo": w_lo, "ci95_hi": w_hi,
                       "min": float(w.min()), "max": float(w.max()),
                       "n_sessions": int(w.size)},
            "cross": {"mean": float(c.mean()), "median": float(np.median(c)),
                      "ci95_lo": c_lo, "ci95_hi": c_hi,
                      "min": float(c.min()), "max": float(c.max()),
                      "n_sessions": int(c.size)},
            "retention_cross_over_within": float(c.mean() / w.mean()),
            "per_session_within": [float(v) for v in w],
            "per_session_cross": [float(v) for v in c]}


def reagg_fair(b, src, ex):
    out = copy.deepcopy(b)
    S = b["sessions"]
    K = [i for i, s in enumerate(S) if not ex(s)]
    sess = [S[i] for i in K]
    out["sessions"] = sess
    for name, a in out["arms"].items():
        core = _arm_core([a["per_session_within"][i] for i in K],
                         [a["per_session_cross"][i] for i in K])
        a.update(core)                         # existing keys keep their order
        ps = src["fair"][name]
        a["switches_2d_total"] = sum(ps[s]["sw2d"] for s in sess)
        if "cross_unscoped_mean" in a:
            a["cross_unscoped_mean"] = float(np.mean([ps[s]["unscoped"] for s in sess]))
    return out


# --------------------------------------------------------------------------- driver
DEPOSITS = {"fig7_variant_best": reagg_variant_best, "fig7_fair_baselines": reagg_fair}


def diff(a, b, path=""):
    """Every leaf that differs (exact)."""
    if isinstance(a, dict) and isinstance(b, dict):
        out = [f"{path}.{k}: key set" for k in set(a) ^ set(b)]
        for k in a.keys() & b.keys():
            out += diff(a[k], b[k], f"{path}.{k}")
        return out
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return [f"{path}: len {len(a)} vs {len(b)}"]
        return [d for i, (x, y) in enumerate(zip(a, b)) for d in diff(x, y, f"{path}[{i}]")]
    if isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b):
        return []
    return [] if (a == b and type(a) is type(b)) else [f"{path}: {a!r} vs {b!r}"]


def main(argv):
    check_only = "--check" in argv
    b = {stem: json.loads((OUT / f"{stem}_b.json").read_text()) for stem in DEPOSITS}
    src = load_sources(b["fig7_variant_best"], b["fig7_fair_baselines"])
    ok = True
    for stem, fn in DEPOSITS.items():
        d = diff(fn(b[stem], src, never), b[stem])
        print(f"[identity] {stem}: {'EXACT' if not d else 'FAILED'}")
        for x in d[:20]:
            print("   ", x)
        ok &= not d
    if not ok:
        sys.exit("[reagg] identity check FAILED -- nothing written")
    if check_only:
        return
    for stem, fn in DEPOSITS.items():
        q = fn(b[stem], src, excluded)
        q["generated_by"] = q["generated_by"] + TAG
        q["qc_excluded"] = provenance()
        dest = OUT / f"{stem}_q.json"
        if dest.is_symlink():
            raise SystemExit(f"[reagg] {dest} is a symlink -- refusing to write through it")
        with open(dest, "w") as fh:
            fh.write(json.dumps(q, indent=1))
        print(f"[reagg] wrote {dest.name}")


if __name__ == "__main__":
    main(sys.argv[1:])
