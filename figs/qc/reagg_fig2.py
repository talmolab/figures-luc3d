#!/usr/bin/env python3
"""q series, Fig 2 family (cross-view association / tracking): re-aggregate the b deposits
with the sessions in src/qc_exclude.py left out.

Every aggregate is recomputed from the per-session records the b deposit already holds,
with the producer's own formula (cited per function). Nothing is re-tracked or re-scored.
One input is not in a deposit: fig3_frame_matched_bmimica's per-arm `cross_switches_total`
needs each session's cross-view switch count, which the deposit's per_session rows do not
carry; it is read from the producer's own per-session cache
(<lucid>/figs/out/tmp/headtohead_b3/A2_C5_bmimica/<session>/score_framematched_<arm>.json),
and every other field of that cache is checked against the deposit before use.

    cd figs && $PY qc/reagg_fig2.py            # identity check, then write out/<stem>_q.json
    cd figs && $PY qc/reagg_fig2.py --check    # identity check only, write nothing

Identity check (always run first): the same code with an EMPTY exclusion list must
reproduce every field of each _b deposit. Floats must be equal, except where the producer
summed in a different order than the deposit records (listed in TOLERANT below), where a
1e-12 relative tolerance is allowed and the fields that needed it are printed.

Timing is NOT re-aggregated. fig3_headtohead_clean{,_j18}'s clean-sitting fields
(`seconds_per_frame_exhaustive`, `seconds_per_frame_greedy_clean`, `clean_frames_per_round`,
`clean_sessions`, `clean_timing_note`, `sitting`) are wall-clock solve times of a sitting
over all 56 Mouse-Dyad-10M sessions. They are not scored against the reference, they pair
with fig3_runtime_clean_j18.json's LUC3D times (which name no session and are not part of
this re-aggregation), and dropping one session's 1000 frames from one arm only would make
Fig 2g compare two different frame sets. They are carried over unchanged.
"""
from __future__ import annotations

import copy
import json
import math
import re
import sys
from pathlib import Path

import numpy as np

FIGS = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(FIGS))
from src.qc_exclude import excluded, provenance  # noqa: E402

OUT = FIGS / "out"
#: the producers' per-session caches (fig3_frame_matched.py, FIG3_VARIANT=b3)
LUCID_TMP = Path("/root/vast/eric/sleap-3d-gui/scratch/repos/lucid/figs/out/tmp")
FM_CACHE = LUCID_TMP / "headtohead_b3" / "A2_C5_bmimica"
TAG = " + figs/qc/reagg_fig2.py (qc_exclude)"
N_CAMERAS_BM = 5


def never(_s):
    return False


def _keep(xs, ex, key=None):
    return [x for x in xs if not ex(x[key] if key else x)]


# --------------------------------------------------------------------------- headtohead
def reagg_headtohead(d, ex, clean=False):
    """fig3_headtohead.py main() phase 3 (lines 533-646). For the clean deposits
    (fig3f_timing_clean.py main(), lines 282-319) the head-to-head part is the same and the
    clean-sitting timing fields are carried over (module docstring)."""
    d = copy.deepcopy(d)
    g_tot = {"idf1_within": [], "idf1_cross": [], "switches": 0}
    e_tot = {"idf1_within": [], "idf1_cross": [], "switches": 0}
    tot_comp = tot_agree = 0
    all_sessions = []
    for c in d["configs"]:
        if c["status"] == "intractable":
            c["sessions_available"] = _keep(c["sessions_available"], ex)
            all_sessions += c["sessions_available"]
            continue
        c["sessions"] = _keep(c["sessions"], ex)
        all_sessions += c["sessions"]
        c["n_sessions"] = len(c["sessions"])
        c["session"] = c["sessions"][0] if len(c["sessions"]) == 1 else None
        c["per_session"] = _keep(c["per_session"], ex, "session")
        if c["status"] != "ok":
            continue
        ok = [r for r in c["per_session"] if r.get("status") == "ok"]
        c["frames_considered"] = sum(r["frames_considered"] for r in ok)
        c["frames_clean"] = sum(r["frames_clean"] for r in ok)
        c["frames_computed"] = sum(r["frames_computed"] for r in ok)
        runtime = sum(r["runtime_seconds_exhaustive"] for r in ok)
        spf = runtime / c["frames_computed"] if c["frames_computed"] else None
        if clean:
            # fig3f_timing_clean.py:300 -- `_previous` is the head-to-head value; the
            # un-suffixed field is the clean sitting's timing, carried over.
            c["seconds_per_frame_exhaustive_previous"] = spf
        else:
            c["seconds_per_frame_exhaustive"] = spf
        c["exhaustive_wall_seconds"] = runtime
        c["n_agree"] = sum(r["n_agree"] for r in ok)
        c["n_compared"] = sum(r["n_compared"] for r in ok)
        c["agreement_rate"] = c["n_agree"] / c["n_compared"] if c["n_compared"] else None
        tot_agree += c["n_agree"]
        tot_comp += c["n_compared"]
        for r in ok:
            sc = r.get("scores")
            if not isinstance(sc, dict) or "greedy" not in sc:
                continue
            for tot, arm in ((g_tot, "greedy"), (e_tot, "exhaustive")):
                tot["idf1_within"].append(sc[arm]["idf1_within"])
                tot["idf1_cross"].append(sc[arm]["idf1_cross"])
                tot["switches"] += sc[arm]["switches"]

    def mean_or_none(xs):
        return float(sum(xs) / len(xs)) if xs else None

    d["sessions"] = list(dict.fromkeys(all_sessions))
    d["n_sessions"] = len(set(all_sessions))
    d["frames_compared"] = tot_comp
    d["agreement_rate"] = tot_agree / tot_comp if tot_comp else None
    d["frames_agree"] = tot_agree
    d["n_idf1_sessions"] = len(g_tot["idf1_within"])
    for k, tot in (("greedy", g_tot), ("exhaustive", e_tot)):
        d[k] = {"idf1_within": mean_or_none(tot["idf1_within"]),
                "idf1_cross": mean_or_none(tot["idf1_cross"]),
                "switches": tot["switches"]}
    d["caps"]["max_frames"] = {c["key"]: c.get("frames_considered", 0) for c in d["configs"]}
    return d


# --------------------------------------------------------------------------- quality
def reagg_quality(d, ex):
    """fig3_quality.py main() (lines 462-519)."""
    d = copy.deepcopy(d)
    for c in d["configs"]:
        c["sessions"] = _keep(c["sessions"], ex)
        c["n_sessions_planned"] = len(c["sessions"])
        c["session"] = c["sessions"][0] if len(c["sessions"]) == 1 else None
        if c.get("status") != "ok":
            continue
        ok = _keep(c["per_session"], ex, "session")   # deposit rows are the producer's `ok`
        c["per_session"] = ok
        c["n_sessions"] = len(ok)
        c["n_compared"] = sum(r["n_compared"] for r in ok)
        c["n_agree"] = sum(r["n_agree"] for r in ok)
        c["agreement_rate"] = c["n_agree"] / c["n_compared"] if c["n_compared"] else None
        c["disagree_frames"] = [{"session": r["session"], "frame": fi}
                                for r in ok for fi in r["disagree_frames"]]
        for m in ("greedy", "exhaustive"):
            frames = sum(r["gt"][m]["frames"] for r in ok)
            exact = sum(r["gt"][m]["exact_match_frames"] for r in ok)
            pair_n = sum(r["gt"][m]["frames_with_pairs"] for r in ok)
            pair_s = sum(r["gt"][m].get("pair_accuracy_sum", 0.0) for r in ok)
            c["gt"][m] = {"exact_match_frames": exact, "frames": frames,
                          "exact_match_rate": exact / frames if frames else None,
                          "pair_accuracy_mean": pair_s / pair_n if pair_n else None,
                          "frames_with_pairs": pair_n}
        c["gt_matching"] = {k: sum(r["gt_matching"][k] for r in ok) for k in c["gt_matching"]}
        c["disagreement_detail"] = _keep(c["disagreement_detail"], ex, "session")
    all_s = {s for c in d["configs"] for s in c.get("sessions", [])}
    d["sessions"] = sorted(all_s)
    d["n_sessions"] = len(all_s)
    return d


# --------------------------------------------------------------------------- exhaustive
def reagg_exhaustive_bm(d, ex):
    """fig3_exhaustive_bmimica.py main() (lines 64-89); `denominator_note` as amended by
    its x6 correction (the full-session denominator is frames_considered x 5)."""
    d = copy.deepcopy(d)
    rows = _keep(d["per_session"], ex, "session")
    d["per_session"] = rows
    w = np.array([r["idf1_within"] for r in rows])
    c = np.array([r["idf1_cross"] for r in rows])
    sw = sum(r["switches"] for r in rows)
    frames = sum(r["frames_computed"] for r in rows)
    considered = sum(r["frames_considered"] for r in rows)
    camf = frames * N_CAMERAS_BM
    d["claim"] = re.sub(r"restricted to the \d+ BMimica", f"restricted to the {len(rows)} BMimica",
                        d["claim"])
    d["n_sessions"] = len(rows)
    d["idf1_within_mean"] = float(w.mean())
    d["idf1_within_median"] = float(np.median(w))
    d["idf1_cross_mean"] = float(c.mean())
    d["switches_total"] = int(sw)
    d["frames_computed_total"] = int(frames)
    d["camera_frames_computed"] = int(camf)
    d["switches_per_100k_camera_frames"] = float(sw / camf * 1e5)
    d["denominator_note"] = (
        f"clean frames only (every camera holds exactly 2 detections): {frames:,} of the "
        f"{considered:,} video frames = {camf:,} camera-frames. Fig 3 G/H score the greedy "
        f"sweep frame-matched on these same frames. The full-session denominator is "
        f"{considered * N_CAMERAS_BM:,} camera-frames.")
    return d


# --------------------------------------------------------------------------- frame-matched
def _fm_cache(session, arm):
    return json.loads((FM_CACHE / session / f"score_framematched_{arm}.json").read_text())


def reagg_frame_matched(d, ex):
    """fig3_frame_matched.py main() (lines 151-183). cross_switches per session come from
    the producer's per-session cache; every field the deposit also holds is checked
    against it first."""
    d = copy.deepcopy(d)
    rows = _keep(d["per_session"], ex, "session")
    d["per_session"] = rows
    d["n_sessions"] = len(rows)
    for arm in ("exhaustive", "greedy"):
        recs = []
        for r in rows:
            cch = _fm_cache(r["session"], arm)
            for dk, ck in ((f"{arm}_idf1_cross", "idf1_cross"),
                           (f"{arm}_idf1_within", "idf1_within"),
                           (f"{arm}_switches", "switches"), ("frames_scored", "frames_scored"),
                           ("frames_considered", "frames_considered")):
                if r[dk] != cch[ck]:
                    sys.exit(f"[reagg] frame-matched cache disagrees with the deposit: "
                             f"{r['session']} {arm} {dk} {r[dk]} vs {cch[ck]}")
            if cch["camera_frames_scored"] != r["frames_scored"] * N_CAMERAS_BM:
                sys.exit(f"[reagg] {r['session']}: camera_frames_scored != 5 x frames_scored")
            recs.append(cch)
        w = np.array([x["idf1_within"] for x in recs])
        c = np.array([x["idf1_cross"] for x in recs])
        sw = sum(x["switches"] for x in recs)
        csw = sum(x["cross_switches"] for x in recs)
        camf = sum(x["camera_frames_scored"] for x in recs)
        d["arms"][arm] = {
            "idf1_within_mean": float(w.mean()),
            "idf1_within_median": float(np.median(w)),
            "idf1_cross_mean": float(c.mean()),
            "idf1_cross_median": float(np.median(c)),
            "switches_total": int(sw), "cross_switches_total": int(csw),
            "camera_frames_scored": int(camf),
            "switches_per_100k_camera_frames": float(sw / camf * 1e5),
            "cross_switches_per_100k_camera_frames": float(csw / camf * 1e5),
        }
    return d


# --------------------------------------------------------------------------- sweeps / fig8
def _score_cell(per):
    """fig3_score.score_cell (lines 191-209), verbatim."""
    n = len(per)
    if n == 0:
        return None

    def mean_of(key):
        vals = [s[key] for s in per if s.get(key) is not None]
        return float(np.mean(vals)) if vals else None

    return {
        "idf1_within": float(np.mean([s["within_idf1"] for s in per])),
        "idf1_cross": float(np.mean([s["cross_idf1"] for s in per])),
        "idp_within": mean_of("within_idp"), "idr_within": mean_of("within_idr"),
        "idp_cross": mean_of("cross_idp"), "idr_cross": mean_of("cross_idr"),
        "switches": int(sum(s["within_switches"] for s in per)),
        "n_sessions": n,
    }


def _cf_total(cf_by_session):
    return sum(sum(v.values()) if isinstance(v, dict) else v for v in cf_by_session.values())


def _n_text(s, n):
    """The producers' f-string session counts in caveat text."""
    s = re.sub(r"The corpus is all \d+ proofread", f"The corpus is all {n} proofread", s)
    s = re.sub(r"through figs/fig3-bench/fig3_bench\.mjs, on all \d+ sessions",
               f"through figs/fig3-bench/fig3_bench.mjs, on all {n} sessions", s)
    return s


def reagg_sweep(d, ex):
    """fig3_sweep50.py main() phase 3 (lines 539-627), and fig6_methods.py's identical
    aggregation (lines 851-916; same score_cell, same rate)."""
    d = copy.deepcopy(d)
    d["sessions"] = _keep(d["sessions"], ex)
    d["camera_frames_by_session"] = {s: v for s, v in d["camera_frames_by_session"].items()
                                     if not ex(s)}
    total_cf = _cf_total(d["camera_frames_by_session"])
    d["total_camera_frames"] = total_cf
    d["caveats"] = [_n_text(s, len(d["sessions"])) for s in d["caveats"]]
    for c in d["cells"]:
        c["per_session"] = _keep(c["per_session"], ex, "session")
        if c.get("status") != "ok":
            continue
        c.update(_score_cell(c["per_session"]))
        if c["switches"] is not None and total_cf:
            c["switches_per_100k_camera_frames"] = c["switches"] * 100_000 / total_cf
    return d


def reagg_sweep_framematched(d, ex, full_q):
    """fig3_sweep_framematched.py main() (lines 104-129). `full_session` and
    `total_camera_frames_full_session` come from the (re-aggregated) full-session sweep
    deposit, which is what the producer copied them from."""
    d = copy.deepcopy(d)
    d["sessions"] = _keep(d["sessions"], ex)
    d["caveats"] = [_n_text(s, len(d["sessions"])) for s in d["caveats"]]
    d["total_camera_frames_full_session"] = full_q["total_camera_frames"]
    fcells = {(c["corr2d"], c["corr3d"]): c for c in full_q["cells"]}
    tot = None
    for c in d["cells"]:
        recs = _keep(c["per_session"], ex, "session")
        if [r["session"] for r in recs] != d["sessions"]:
            sys.exit("[reagg] framematched per_session does not follow `sessions`")
        camf = sum(r["camera_frames"] for r in recs)
        if tot is None:
            tot = camf
        assert camf == tot, "frame-matched denominators differ between cells"
        fc = fcells[(c["corr2d"], c["corr3d"])]
        c["full_session"] = {k: fc.get(k) for k in ("idf1_within", "idf1_cross", "switches",
                                                     "switches_per_100k_camera_frames")}
        mean = lambda k: sum(r[k] for r in recs) / len(recs)  # noqa: E731
        sw = sum(r["within_switches"] for r in recs)
        c.update({"idf1_within": mean("within_idf1"), "idf1_cross": mean("cross_idf1"),
                  "idp_within": mean("within_idp"), "idr_within": mean("within_idr"),
                  "idp_cross": mean("cross_idp"), "idr_cross": mean("cross_idr"),
                  "switches": sw, "switches_per_100k_camera_frames": sw / camf * 1e5,
                  "per_session": recs})
        # n_sessions is score_cell's, carried from the full-session sweep's cell
        c["n_sessions"] = len(recs)
    d["total_camera_frames"] = tot
    d["camera_frames_by_session"] = {r["session"]: r["camera_frames"]
                                     for r in d["cells"][0]["per_session"]}
    return d


# --------------------------------------------------------------------------- driver
SWEEP_FULL = "fig3_sweep50__distanceThreshold25-stale20-sync_e508a7ab"
JOBS = [
    ("fig3_headtohead", lambda d, ex, _: reagg_headtohead(d, ex)),
    ("fig3_headtohead_clean_j18", lambda d, ex, _: reagg_headtohead(d, ex, clean=True)),
    # not placed by the b/q build (FIG3F_CLEAN_TAG=_j18), but the same family and it names
    # the session, so its q copy is re-aggregated too rather than left stale
    ("fig3_headtohead_clean", lambda d, ex, _: reagg_headtohead(d, ex, clean=True)),
    ("fig3_quality", lambda d, ex, _: reagg_quality(d, ex)),
    ("fig3_quality__distanceThreshold25-stale20-sync_e508a7ab",
     lambda d, ex, _: reagg_quality(d, ex)),
    ("fig3_exhaustive_bmimica", lambda d, ex, _: reagg_exhaustive_bm(d, ex)),
    ("fig3_frame_matched_bmimica", lambda d, ex, _: reagg_frame_matched(d, ex)),
    ("fig3_sweep50", lambda d, ex, _: reagg_sweep(d, ex)),
    (SWEEP_FULL, lambda d, ex, _: reagg_sweep(d, ex)),
    (SWEEP_FULL + "_framematched",
     lambda d, ex, done: reagg_sweep_framematched(d, ex, done[SWEEP_FULL])),
    ("fig8_methods_50", lambda d, ex, _: reagg_sweep(d, ex)),
]
#: fields whose producer summed in an order the deposit does not record exactly
TOLERANT: set = set()


def compare(a, b, path="", tol_hits=None):
    """Exact structural equality; floats exact or within 1e-12 relative (recorded)."""
    if isinstance(a, dict) and isinstance(b, dict):
        if set(a) != set(b):
            return [f"{path}: keys differ {sorted(set(a) ^ set(b))}"]
        return [m for k in a for m in compare(a[k], b[k], f"{path}.{k}", tol_hits)]
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return [f"{path}: length {len(a)} vs {len(b)}"]
        return [m for i, (x, y) in enumerate(zip(a, b))
                for m in compare(x, y, f"{path}[{i}]", tol_hits)]
    if isinstance(a, float) and isinstance(b, float):
        if a == b or (math.isnan(a) and math.isnan(b)):
            return []
        if abs(a - b) <= 1e-12 * max(abs(a), abs(b)):
            tol_hits.append(f"{path}: {a!r} vs {b!r}")
            return []
        return [f"{path}: {a!r} vs {b!r}"]
    if type(a) is not type(b) or a != b:
        return [f"{path}: {a!r} vs {b!r}"]
    return []


def names_session(o):
    s = json.dumps(o)
    return any(excluded(m.group(0)) for m in re.finditer(r"\d{8}_?\d{6}", s))


def main(argv):
    check_only = "--check" in argv
    ok = True
    done_id, done_q = {}, {}
    for stem, fn in JOBS:
        b = json.loads((OUT / f"{stem}_b.json").read_text())
        ident = fn(b, never, done_id)
        done_id[stem] = ident
        tol = []
        diffs = compare(ident, b, stem, tol)
        status = "EXACT" if not diffs and not tol else ("TOLERANCE" if not diffs else "FAIL")
        print(f"[identity] {stem}: {status}")
        for m in diffs[:20]:
            print(f"    differs {m}")
        for m in tol[:20]:
            print(f"    within 1e-12 {m}")
        if diffs:
            ok = False
        done_q[stem] = fn(b, excluded, done_q)
    if not ok:
        sys.exit("[reagg] identity check FAILED -- nothing written")
    if check_only:
        return
    for stem, _ in JOBS:
        q = done_q[stem]
        if names_session(q):
            sys.exit(f"[reagg] {stem}: an excluded session is still named in the q deposit")
        q["generated_by"] = q["generated_by"] + TAG
        q["qc_excluded"] = provenance()
        p = OUT / f"{stem}_q.json"
        if p.is_symlink():
            sys.exit(f"[reagg] {p} is a symlink -- refusing to write through it")
        with open(p, "w") as fh:
            json.dump(q, fh, indent=1)
        print(f"[reagg] wrote {p.relative_to(FIGS)}")


if __name__ == "__main__":
    main(sys.argv[1:])
