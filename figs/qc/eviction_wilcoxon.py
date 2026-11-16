#!/usr/bin/env python3
"""The paired Wilcoxon P values of the staleness (eviction) paragraphs, made exact.

The Supplementary eviction text quotes five session-level P values: N = 20 against no
eviction, N = 1, N = 10 and N = 30 on the 42 multi-animal SLAP-2M sessions, and no
eviction against N = 20 on MouseDyad-10M. No script in this repo produced them. They
reproduce to the printed digits from `stats.wilcoxon(d)` at its defaults, on each
session's cross-view IDF1 difference. Many sessions do not change (zero differences),
and with zeros present scipy's default `method="auto"` silently uses the normal
approximation. This script is the producer: it drops the zeros (zero_method="wilcox",
the counts the text quotes as "improved / worsened") and computes the exact two-sided
signed-rank P over the nonzero differences. When the nonzero |d| tie, it uses scipy's
full sign-flip permutation instead. The deposit also records the old approximate P, so
each printed value can be traced from old to new.

    cd figs && $PY qc/eviction_wilcoxon.py     # writes out/eviction_wilcoxon_q.json

Inputs: out/stale_sweep_slap2m.json (SLAP-2M has no excluded session, so it is the same
in both series) and out/fig8_methods_50_{b,q}.json (MouseDyad-10M, 56 and 55 sessions).
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

FIGS = Path(__file__).resolve().parents[1]
OUT = FIGS / "out"
REF = "sync_stale20_dist25"          # the app default, N = 20


def exact_p(d):
    nz = d[d != 0]
    if len(np.unique(np.abs(nz))) == len(nz):
        return float(stats.wilcoxon(nz, method="exact").pvalue), "exact"
    p = stats.wilcoxon(nz, method=stats.PermutationMethod(n_resamples=np.inf)).pvalue
    return float(p), "exact permutation (tied |d|)"


def compare(cells, a, b, sessions, label):
    d = np.array([cells[a][s]["cross_idf1"] - cells[b][s]["cross_idf1"] for s in sessions])
    p, method = exact_p(d)
    return {"label": label, "a": a, "b": b, "statistic": "cross_idf1, a minus b",
            "n_sessions": len(d), "a_better": int((d > 0).sum()),
            "b_better": int((d < 0).sum()), "unchanged": int((d == 0).sum()),
            "p_exact": p, "p_method": method,
            "p_scipy_default_approx": float(stats.wilcoxon(d).pvalue)}


def by_config(path):
    d = json.loads(path.read_text())
    return {c["config"]: {r["session"]: r for r in c["per_session"]} for c in d["cells"]}


def main():
    out = {"generated_by": "figs/qc/eviction_wilcoxon.py", "test":
           "two-sided Wilcoxon signed-rank on per-session differences, zeros dropped, exact",
           "slap2m": [], "mousedyad10m": {}}
    s = by_config(OUT / "stale_sweep_slap2m.json")
    multi = sorted(k for k, r in s["stale0_dist25"].items() if r["animals"] > 1)
    assert len(multi) == 42, len(multi)
    for b, lab in (("stale0_dist25", "N = 20 vs no eviction"),
                   ("sync_stale1_dist25", "N = 20 vs N = 1"),
                   ("sync_stale10_dist25", "N = 20 vs N = 10"),
                   ("sync_stale30_dist25", "N = 20 vs N = 30")):
        out["slap2m"].append(compare(s, REF, b, multi, lab))
    for series in ("b", "q"):
        m = by_config(OUT / f"fig8_methods_50_{series}.json")
        ss = sorted(m["stale0_dist25"])
        out["mousedyad10m"][series] = compare(m, "stale0_dist25", REF, ss,
                                              "no eviction vs N = 20")
    assert out["mousedyad10m"]["q"]["n_sessions"] == 55
    (OUT / "eviction_wilcoxon_q.json").write_text(json.dumps(out, indent=1) + "\n")
    for r in out["slap2m"] + [out["mousedyad10m"]["b"], out["mousedyad10m"]["q"]]:
        print(f"  {r['label']:24s} n={r['n_sessions']:2d} {r['a_better']:2d} better "
              f"{r['b_better']:2d} worse {r['unchanged']:2d} same  P exact {r['p_exact']:.3g} "
              f"(scipy default {r['p_scipy_default_approx']:.3g})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
