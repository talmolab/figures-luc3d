#!/usr/bin/env python3
"""F5FIX old-vs-new report (2026-10-15 audit). Reads only; writes nothing.

  1. Fig 5 controls: text values vs the new FIG5_MIN_DISPLAYS=6 run, and the >=5
     numbers the script used to print, from data/fig5f5fix/fig5s_controls.csv
     (old deposit data/fig5/fig5s_controls.csv shown alongside).
  2. Fig 5G: every number the tex quotes, from the published deposit
     out/fig5_rear_coupling_2animal.json and the new per-direction deposit
     out/fig5_rear_coupling_2animal_perdir.json, plus the new per-direction far / null.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

FIGS = Path(__file__).resolve().parents[2]
OLD_CSV = FIGS / "data/fig5/fig5s_controls.csv"
NEW_CSV = FIGS / "data/fig5f5fix/fig5s_controls.csv"
OLD_J = FIGS / "out/fig5_rear_coupling_2animal.json"
NEW_J = FIGS / "out/fig5_rear_coupling_2animal_perdir.json"


def controls(csv, M):
    d = pd.read_csv(csv)
    base = d[d["mode"] == "per_animal"]
    big = base[base.n >= M]
    fs = big.female_lead / (big.female_lead + big.male_lead)
    fr = big.female_rear_frac / (big.female_rear_frac + big.male_rear_frac)
    r, p = stats.pearsonr(fs, fr)
    o = {"n_sessions": len(big), "r": round(r, 3), "P": round(p, 2)}
    for mode in ("abs", "shared"):
        a = d[d["mode"] == mode].set_index("session").loc[big.session]
        o[f"{mode}_female"] = f"{int((a.leader == 'female').sum())}/{len(big)}"
        o[f"{mode}_same"] = f"{int((a.leader.values == big.leader.values).sum())}/{len(big)}"
    allb = base[base.n >= 1]
    o["male_med_mm"] = round(allb.L_male_mm.median(), 1)
    o["female_med_mm"] = round(allb.L_female_mm.median(), 1)
    o["male_longer"] = f"{int((allb.L_male_mm > allb.L_female_mm).sum())}/{len(allb)}"
    uniq = allb[allb.leader != "tie"]
    sh = ((uniq.leader == "female") & (uniq.L_female_mm < uniq.L_male_mm)) | \
         ((uniq.leader == "male") & (uniq.L_male_mm < uniq.L_female_mm))
    o["leader_shorter"] = f"{int(sh.sum())}/{len(uniq)} P={stats.binomtest(int(sh.sum()), len(uniq)).pvalue:.4f}"
    fl = allb[allb.L_female_mm > allb.L_male_mm]
    k, n = int(fl.female_lead.sum()), int(fl.female_lead.sum() + fl.male_lead.sum())
    o["female_longer"] = f"{len(fl)} sessions, {100 * k / n:.0f}% P={stats.binomtest(k, n).pvalue:.1e}"
    return o


print("=== 1. Fig 5 controls (tex L183 / L319 / L323 / L403) ===")
TEXT = {"n_sessions": 23, "r": -0.011, "P": 0.96, "abs_female": "23/23",
        "shared_female": "20/23", "male_med_mm": 90.1, "female_med_mm": 84.2,
        "male_longer": "29/37", "leader_shorter": "28/36 P=0.0012",
        "female_longer": "8 sessions, 78% P=6e-9"}
cols = {"text": TEXT, "old CSV >=5 (script default)": controls(OLD_CSV, 5),
        "old CSV >=6": controls(OLD_CSV, 6)}
if NEW_CSV.exists():
    cols["NEW CSV >=6"] = controls(NEW_CSV, 6)
    cols["NEW CSV >=5"] = controls(NEW_CSV, 5)
else:
    print(f"  (missing {NEW_CSV} -- new run not done)")
print(pd.DataFrame(cols).to_string())

print("\n=== 2. Fig 5G (tex L185 + Fig 5 caption G) ===")
old = json.load(open(OLD_J))["corpora"]["BMimica"]
if not NEW_J.exists():
    sys.exit(f"missing {NEW_J}")
new = json.load(open(NEW_J))["corpora"]["BMimica"]
t = np.asarray(old["t"])
assert np.allclose(t, new["t"]), "time axes differ"
z = int(np.argmin(np.abs(t)))


def stats_of(g, label):
    if not g:
        return f"{label:42s} --"
    m = np.asarray(g["p50"])
    k, w = int(np.argmax(m)), np.abs(t) <= 1.0
    pre = m[(t >= -1.0) & (t <= 0)]
    return (f"{label:42s} n={g['n_sessions']:2d}  lag0 {m[z]:.3f}  peak {m[k]:.3f} @ {t[k]:+.2f}s"
            f"  range|t|<=1s [{m[w].min():.3f}, {m[w].max():.3f}]  max on [-1,0] {pre.max():.3f}")


print("-- unchanged-by-design (near curves, counts): max |old-new| per curve")
for k in ("all", "near", "far", "null", "near_i0", "near_i1"):
    d = np.max(np.abs(np.asarray(old[k]["p50"]) - np.asarray(new[k]["p50"])))
    print(f"  {k:8s} max|dp50| = {d:.3g}   n_sessions {old[k]['n_sessions']} -> {new[k]['n_sessions']}")
print(f"  onsets all/near: {old['n_onsets']['all']:,}/{old['n_onsets']['near']:,} -> "
      f"{new['n_onsets']['all']:,}/{new['n_onsets']['near']:,}   (caption: 9,354 / 2,915; 56 sessions -> {new['n_sessions']})")
for a, txt in (("i0", "50.2% (male onset, female already up)"), ("i1", "6.2% (female onset, male already up)")):
    print(f"  already-mid-bout {a}: {old['already_mid_bout'][a]['pooled']:.1%} -> "
          f"{new['already_mid_bout'][a]['pooled']:.1%}   text {txt}")

print("\n-- right sub-axis: FEMALE rearing around MALE onset (near_i0); text 4.7x at onset, 5.2x ~1/3 s later")
print(" ", stats_of(new["near_i0"], "near_i0 (female | male onset)"))
print(" ", stats_of(old["far"], "OLD far (both directions, all sessions)"))
print(" ", stats_of(new.get("far_i0"), "NEW far_i0 (same sessions as near_i0)"))
print(" ", stats_of(old["null"], "OLD null (both directions, all sessions)"))
print(" ", stats_of(new.get("null_i0"), "NEW null_i0"))
print(" ", stats_of(new.get("null_near_i0"), "NEW null_near_i0"))
print("\n-- left sub-axis: MALE rearing around FEMALE onset (near_i1); text below baseline before/at onset, 1.7x ~0.8 s later")
print(" ", stats_of(new["near_i1"], "near_i1 (male | female onset)"))
print(" ", stats_of(old["far"], "OLD far (both directions, all sessions)"))
print(" ", stats_of(new.get("far_i1"), "NEW far_i1 (same sessions as near_i1)"))
print(" ", stats_of(old["null"], "OLD null (both directions, all sessions)"))
print(" ", stats_of(new.get("null_i1"), "NEW null_i1"))
print(" ", stats_of(new.get("null_near_i1"), "NEW null_near_i1"))
for dk in ("i0", "i1"):
    g = new.get(f"far_{dk}")
    if g:
        print(f"  far_{dk}: min far onsets among admitted sessions = {g['min_far_onsets_in_admitted']}; "
              f"sessions without a far curve = {g['sessions_without_curve']}")
print(f"  onsets far_i0 / far_i1: {new['n_onsets'].get('far_i0')} / {new['n_onsets'].get('far_i1')}")
