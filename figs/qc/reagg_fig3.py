#!/usr/bin/env python3
"""Re-aggregate the paper's Fig 3 (labeling and triangulation) deposits without the
sessions on `src/qc_exclude.py`'s list -- the "q" series.

Naming is pre-swap: the repo's `fig4_*` deposits feed the paper's Fig 3.

    out/fig4_by_views_b.json        -> out/fig4_by_views_q.json
    out/fig4_robust_sessions_b.json -> out/fig4_robust_sessions_q.json
    out/fig4_anipose_b.json         -> out/fig4_anipose_q.json
    out/fig4_b.json                 -> out/fig4_q.json

No LUC3D code runs here. Every aggregate is recomputed from the deposit's own
per-session records with its producer's formula:

  fig4_by_views     `summarize()` in figs/fig2_solvers_by_views.mjs: median ACROSS
                    session p25/p50/p75 (JS `med`: finite values, mean of the middle
                    two when even), n_sessions, n_values = sum of session n. `sessions`,
                    `keypoints_used` = sum n_keypoints, `solves` = sum over k of the
                    session's dlt.n + ba.n (one DLT and one refined solve per scored
                    subset; exact on the b deposit, i.e. no DLT returned null).
  fig4_robust_sessions  per-session records only; `gate` is the run-integrity check
                    against fig4.json's pooled strata on the SAME run and is carried.
  fig4_anipose      per_session / per_session_optim dropped; `input.n_sessions`,
                    `input.keypoints`; `reproj_px` / `heldout_reproj_px` are pooled
                    over a 20k-per-session random sample the deposit does not store, so
                    they come from re-running the producer's own `_block` (aniposelib
                    baseline, Python) via figs/qc/anipose_linear_rerun.py -> the .npz
                    below, checked row-for-row against the deposit and pooled over all
                    sessions back to the deposit's numbers before the kept pool is used.
                    `timing` is session-independent and carried.
  fig4              `sessions`, `n_sessions`, `keypoints`, `per_session`, and every
                    COUNT that is a sum of per-session quantities (worse_than_dlt,
                    heldout_paired counts, guard, heldout_by_views.n_keypoints). The
                    pooled-over-keypoints percentile blocks (methods.*.reproj_px /
                    err3d_mm, reference_reproj_px, distortion_px, legacy_regression_px,
                    diff_vs_dlt_px, heldout_reproj_px, heldout_paired.diff_px /
                    median_diff_px, heldout_by_views.by_k, triage, robust,
                    by_worst_view) need per-keypoint LUC3D values that are not on disk
                    and LUC3D may not be re-run, so they are NOT reproducible: with an
                    exclusion they are set to null and listed in `qc_unreproducible`
                    (none is read by a placed Fig 3 panel). `methods.*.us_per_keypoint`
                    is timing and is carried.

Identity check (mandatory, run every time): with an EMPTY exclusion list the same code
must give back every field of the `_b` deposit. Fields that are not reproducible are
carried in that mode and named in the printout, so the check never claims them.

    cd figs && <python> qc/reagg_fig3.py           # identity check, then write _q
    cd figs && <python> qc/reagg_fig3.py --check   # identity check only
"""
import argparse
import copy
import json
import math
import sys
from pathlib import Path

import numpy as np

FIGS = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(FIGS))
from src.qc_exclude import excluded, provenance  # noqa: E402

OUT = FIGS / "out"
ANIPOSE_NPZ = OUT / "tmp" / "fig4_anipose_linear_rerun.npz"
STAMP = " + figs/qc/reagg_fig3.py (qc_exclude)"
NONE = lambda s: False  # noqa: E731  -- the empty exclusion list


def med(xs):
    """JS `med` in fig2_solvers_by_views.mjs."""
    v = sorted(x for x in xs if isinstance(x, (int, float)) and math.isfinite(x))
    if not v:
        return None
    h = len(v) >> 1
    return v[h] if len(v) % 2 else 0.5 * (v[h - 1] + v[h])


def stamp(d, ex, producer):
    if ex is NONE:
        return d
    d["generated_by"] = d.get("generated_by", producer) + STAMP
    d["qc_excluded"] = provenance()
    return d


def unrepro(d, ex, paths, reason):
    """Null the fields a session drop cannot recompute; carry them when nothing is
    dropped (the identity mode), so the check exercises everything else."""
    dropped = ex is not NONE
    if dropped:
        d.setdefault("qc_unreproducible", {})
    for p in paths:
        *head, last = p.split(".")
        o = d
        for h in head:
            o = o[h]
        if last not in o:
            continue
        if dropped:
            o[last] = None
            d["qc_unreproducible"][p] = reason
    return d


# ---------------------------------------------------------------- fig4_by_views --
def by_views(b, ex):
    d = copy.deepcopy(b)
    ps = [r for r in d["per_session"] if not ex(r["session"])]
    C = d["cameras"]
    d["per_session"] = ps
    d["sessions"] = len(ps)
    d["keypoints_used"] = sum(r["n_keypoints"] for r in ps)
    d["solves"] = sum(r["by_k"][k][m]["n"] for r in ps for k in r["by_k"]
                      for m in ("dlt", "ba") if r["by_k"][k][m])
    for arm, field, kmax in (("err3d_mm_across_sessions", "by_k", C),
                             ("heldout_px_across_sessions", "heldout_px_by_k", C - 1)):
        agg = {}
        for k in range(2, kmax + 1):
            agg[str(k)] = {}
            for m in ("dlt", "ba"):
                g = [r[field][str(k)][m] for r in ps if r[field][str(k)][m]]
                agg[str(k)][m] = {
                    "p25": med([x["p25"] for x in g]), "p50": med([x["p50"] for x in g]),
                    "p75": med([x["p75"] for x in g]), "n_sessions": len(g),
                    "n_values": sum(x["n"] for x in g)}
        d[arm] = agg
    return stamp(d, ex, d.get("driver", "figs/fig2_solvers_by_views.mjs"))


# -------------------------------------------------------- fig4_robust_sessions --
def robust_sessions(b, ex):
    d = copy.deepcopy(b)
    d["per_session"] = [r for r in d["per_session"] if not ex(r["session"])]
    return stamp(d, ex, "figs/fig2_solvers_robust_sessions.mjs")


# ----------------------------------------------------------------- fig4_anipose --
def pct(a):
    """`pct` in figs/fig2_solvers_anipose.py."""
    if not len(a):
        return None
    v = np.asarray(a, dtype=np.float64)
    return {"n": int(v.size), "mean": float(v.mean()),
            **{f"p{p}": float(np.percentile(v, p)) for p in (5, 25, 50, 75, 90, 95, 99)}}


def anipose_samples(b):
    """Per-session pooled samples from the re-run, after checking the re-run's rows
    against the deposit field for field."""
    if not ANIPOSE_NPZ.exists():
        sys.exit(f"{ANIPOSE_NPZ} missing -- run figs/qc/anipose_linear_rerun.py first")
    z = np.load(ANIPOSE_NPZ)
    rows = json.loads(str(z["rows"]))
    if rows != b["per_session"]:
        bad = [r["session"] for r, s in zip(rows, b["per_session"]) if r != s]
        sys.exit(f"anipose re-run does not reproduce the deposit's per_session: {bad[:5]}")
    out = []
    for key in ("err", "ho"):
        lens = z[f"{key}_len"]
        out.append(np.split(z[key], np.cumsum(lens)[:-1]))
    return [r["session"] for r in rows], out[0], out[1]


def anipose(b, ex, samples):
    d = copy.deepcopy(b)
    d["per_session"] = [r for r in d["per_session"] if not ex(r["session"])]
    d["per_session_optim"] = [r for r in d["per_session_optim"] if not ex(r["session"])]
    d["input"]["n_sessions"] = len(d["per_session"])
    d["input"]["keypoints"] = sum(r["n_keypoints"] for r in d["per_session"])
    sess, err, ho = samples
    keep = [i for i, s in enumerate(sess) if not ex(s)]
    # the producer pools Python lists of float32 values in session order
    d["reproj_px"] = pct(np.concatenate([err[i] for i in keep]).astype(np.float64))
    d["heldout_reproj_px"] = pct(np.concatenate([ho[i] for i in keep]).astype(np.float64))
    return stamp(d, ex, "figs/fig2_solvers_anipose.py")


# ------------------------------------------------------------------------- fig4 --
FIG4_UNREPRO = [
    "methods.dlt.reproj_px", "methods.dlt.err3d_mm",
    "methods.ba.reproj_px", "methods.ba.err3d_mm",
    "methods.ba_legacy.reproj_px", "methods.ba_legacy.err3d_mm",
    "reference_reproj_px", "distortion_px", "legacy_regression_px",
    "diff_vs_dlt_px.ba", "diff_vs_dlt_px.ba_legacy",
    "heldout_reproj_px.dlt", "heldout_reproj_px.ba",
    "heldout_paired.diff_px", "heldout_paired.median_diff_px",
    "heldout_by_views.by_k", "triage", "robust", "by_worst_view",
]
FIG4_REASON = ("pooled over individual keypoints of the whole LUC3D run; the "
               "per-keypoint values are not deposited and LUC3D may not be re-run, so "
               "this cannot be recomputed without the excluded session(s)")


def zero_sum(total):
    """A non-negative count whose corpus total is 0 is 0 in every session, so it is 0
    for any subset. Anything else is not derivable from the deposit."""
    return 0 if total == 0 else None


def fig4(b, ex):
    d = copy.deepcopy(b)
    ps = [r for r in d["per_session"] if not ex(r["session"])]
    d["per_session"] = ps
    d["sessions"] = [s for s in d["sessions"] if not ex(s)]
    if d["sessions"] != [r["session"] for r in ps]:
        sys.exit("fig4: sessions[] and per_session[] disagree")
    d["n_sessions"] = len(ps)
    K = sum(r["n_keypoints"] for r in ps)
    d["keypoints"] = K
    # worse_than_dlt: ba_legacy = sum(legacy_worse_frac * legN), legN = the session's
    # finite legacy solves (= n_keypoints on this run; the identity check pins it).
    w = d["worse_than_dlt"]
    w["ba"] = zero_sum(b["worse_than_dlt"]["ba"])
    w["ba_legacy"] = sum(round(r["legacy_worse_frac"] * r["n_keypoints"]) for r in ps)
    w["of"] = K
    hp = d["heldout_paired"]
    n = sum(r["heldout_n"] for r in ps)
    better = sum(round(r["heldout_ba_better_frac"] * r["heldout_n"]) for r in ps)
    hp["n"] = n
    hp["ba_better"] = better
    hp["tie"] = zero_sum(b["heldout_paired"]["tie"])
    hp["dlt_better"] = n - better - hp["tie"] if hp["tie"] is not None else None
    hp["ba_better_frac"] = better / n if n else None
    g = d["guard"]
    g["returned_seed"] = zero_sum(b["guard"]["returned_seed"])
    g["of"] = K
    g["frac"] = g["returned_seed"] / K if (K and g["returned_seed"] is not None) else None
    # subsample_every = 1 on this run, so every keypoint entered the by-views pass
    hv = d["heldout_by_views"]
    if hv["subsample_every"] != 1:
        sys.exit("fig4: heldout_by_views.subsample_every != 1; n_keypoints not derivable")
    hv["n_keypoints"] = K
    unrepro(d, ex, FIG4_UNREPRO, FIG4_REASON)
    return stamp(d, ex, d.get("driver", "figs/fig2_solvers_measure.mjs"))


# ------------------------------------------------------------------------ main --
def diff(a, b, path=""):
    """Paths where two parsed JSON values differ (exact)."""
    if isinstance(a, dict) and isinstance(b, dict):
        out = []
        for k in set(a) | set(b):
            if k not in a or k not in b:
                out.append(f"{path}.{k} (missing on one side)")
            else:
                out += diff(a[k], b[k], f"{path}.{k}")
        return out
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return [f"{path} (len {len(a)} vs {len(b)})"]
        return [x for i, (u, v) in enumerate(zip(a, b)) for x in diff(u, v, f"{path}[{i}]")]
    if type(a) is not type(b) and not (isinstance(a, (int, float)) and isinstance(b, (int, float))):
        return [f"{path} ({a!r} vs {b!r})"]
    return [] if a == b else [f"{path} ({a!r} vs {b!r})"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="identity check only")
    a = ap.parse_args()
    jobs = [("fig4_by_views", by_views), ("fig4_robust_sessions", robust_sessions),
            ("fig4_anipose", None), ("fig4", fig4)]
    ok = True
    results = {}
    for stem, fn in jobs:
        b = json.loads((OUT / f"{stem}_b.json").read_text())
        if fn is None:
            samples = anipose_samples(b)
            fn = lambda b_, ex, s=samples: anipose(b_, ex, s)  # noqa: E731
        ident = fn(b, NONE)
        bad = diff(ident, b)
        carried = [p for p in FIG4_UNREPRO] if stem == "fig4" else []
        print(f"[identity] {stem}: {'EXACT' if not bad else f'{len(bad)} DIFFS'}"
              + (f" (carried, not reproduced: {', '.join(carried)})" if carried else ""))
        for x in bad[:20]:
            print("    ", x)
        ok &= not bad
        results[stem] = fn(b, excluded)
    if not ok:
        sys.exit("identity check FAILED -- nothing written")
    if a.check:
        return
    for stem, q in results.items():
        p = OUT / f"{stem}_q.json"
        if p.is_symlink():
            sys.exit(f"{p} is a symlink -- refusing to write through it")
        with open(p, "w") as fh:
            fh.write(json.dumps(q, indent=1))
        print(f"[wrote] {p.relative_to(FIGS)}")


if __name__ == "__main__":
    main()
