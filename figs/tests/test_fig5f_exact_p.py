"""Fig 5f's paired Wilcoxon P is the exact one, not scipy's tie fallback.

The female shares tie, so stats.wilcoxon(method="auto") switches to the normal
approximation (P = 4.0e-5 on the q series). The panel now computes the exact
sign-flip null of the tied ranks; see panels/fig5_10_leader.py.
"""
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import stats

FIGS = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("fig5_10_leader", FIGS / "panels/fig5_10_leader.py")
leader = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(leader)


@pytest.mark.parametrize("n", [6, 11, 22, 30])
def test_matches_scipy_exact_without_ties(n):
    d = np.random.default_rng(n).normal(0.3, 1.0, n)
    assert leader.exact_signed_rank_p(d) == pytest.approx(stats.wilcoxon(d, method="exact").pvalue,
                                                          rel=1e-12)


def test_matches_full_permutation_with_ties():
    d = np.round(np.random.default_rng(0).normal(0.4, 1.0, 16), 1)
    d = d[d != 0]
    assert len(np.unique(np.abs(d))) < len(d)
    perm = stats.wilcoxon(d, method=stats.PermutationMethod(n_resamples=np.inf)).pvalue
    assert leader.exact_signed_rank_p(d) == pytest.approx(perm, rel=1e-12)


def test_deposited_fig5f_p_is_exact():
    t = pd.read_csv(FIGS / "data/fig5/fig5f_leader_by_session.csv")
    row = t[t.session == f"TEST_SESSION_ge{leader.MIN_DISPLAYS}"].iloc[0]
    n = int(row.displays)
    sessions = t[t.session.str.match(r"^\d{8}_\d{6}$")]
    big = sessions[sessions.displays >= leader.MIN_DISPLAYS]
    assert n == len(big)
    # every one of these sessions is female-led, so the exact P is the two extreme flips
    assert (big.female_lead > big.male_lead).all()
    assert row.wilcoxon_p == pytest.approx(2 / 2 ** n, rel=1e-12)


def test_eviction_p_values_are_exact_and_match_the_text():
    """qc/eviction_wilcoxon.py: the Supplementary eviction P values, exact.

    The scipy-default (normal approximation) values are the ones the text used to print,
    which is how they were traced; the exact ones are what it prints now."""
    import json
    d = json.loads((FIGS / "out/eviction_wilcoxon_q.json").read_text())
    s = {r["label"]: r for r in d["slap2m"]}
    printed_old = {"N = 20 vs no eviction": "0.012", "N = 20 vs N = 1": "0.003",
                   "N = 20 vs N = 10": "0.72", "N = 20 vs N = 30": "0.49"}
    printed_new = {"N = 20 vs no eviction": "0.011", "N = 20 vs N = 1": "0.002",
                   "N = 20 vs N = 10": "0.73", "N = 20 vs N = 30": "0.50"}
    for k, old in printed_old.items():
        nd = len(old.split(".")[1])
        assert f"{s[k]['p_scipy_default_approx']:.{nd}f}" == old
        assert f"{s[k]['p_exact']:.{nd}f}" == printed_new[k]
    assert (s["N = 20 vs no eviction"]["a_better"], s["N = 20 vs no eviction"]["b_better"]) == (20, 6)
    md = d["mousedyad10m"]
    assert f"{md['b']['p_scipy_default_approx']:.3f}" == "0.017"
    assert f"{md['q']['p_scipy_default_approx']:.3f}" == "0.031"
    assert f"{md['q']['p_exact']:.3f}" == "0.030"
    assert (md["q"]["n_sessions"], md["q"]["a_better"], md["q"]["b_better"]) == (55, 11, 4)
