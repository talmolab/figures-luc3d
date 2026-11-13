"""Add a UDMT bar to figure_method_comparison.png (per Eric's request).

UDMT is OFF-POOL and computed on ONLY the hardest 4-animal session (6 cam-sessions),
whereas every other bar aggregates 252 multi-animal cam-sessions. So the UDMT bar is drawn
with a distinct hatch + an explicit "(off-pool, hardest session only)" label and a caption
caveat — it is NOT a like-for-like aggregate bar. The apples-to-apples comparison lives in
figure_udmt_hardsession_percam.png.

Reuses make_plots_compare.py's ROWS/idf1 machinery so the existing bars are identical.
Output: outputs/PAF_3d_kalman/plots/figure_method_comparison.{png,pdf} (regenerated with UDMT).
"""
from __future__ import annotations
import sys
from pathlib import Path
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))
import make_plots_compare as mpc  # noqa: E402

UDMT_CSV = mpc.PAF / "_eval_udmt_hardsession.csv"
PLOTS = mpc.PLOTS


def main():
    bd = pd.read_csv(mpc.BD, sep="\t", header=None, names=["k", "raw", "sus"]).drop_duplicates("k", keep="last").set_index("k")
    recs = []
    for key, label, src, color, hi in mpc.ROWS:
        if key not in bd.index:
            continue
        f1, ci = mpc.idf1(src)
        if f1 is None:
            continue
        recs.append(dict(label=label, color=color, hi=hi, hatch="",
                         raw=int(bd.loc[key, "raw"]), sus=int(bd.loc[key, "sus"]), idf1=f1, ci=ci))

    udmt_ok = False
    if UDMT_CSV.exists():
        u = pd.read_csv(UDMT_CSV)
        if len(u):
            f1 = u.idf1.dropna().mean()
            a = u.idf1.dropna().values
            ci = 1.96 * a.std(ddof=1) / np.sqrt(len(a)) if len(a) > 1 else 0
            raw = int(u.num_switches.fillna(0).sum())
            recs.append(dict(label="UDMT — off-pool, hardest session only",
                             color="#5d4037", hi=True, hatch="///",
                             raw=raw, sus=raw, idf1=f1, ci=ci))
            udmt_ok = True
            print(f"UDMT: meanIDF1={f1:.3f} rawSwitches={raw} over {len(u)} cam-sessions")
    if not udmt_ok:
        print("UDMT csv missing/empty; regenerating base figure without UDMT")

    s = pd.DataFrame(recs).sort_values("idf1").reset_index(drop=True)
    mpl.rcParams["font.family"] = "DejaVu Sans"
    fig, (axF, axS) = plt.subplots(1, 2, figsize=(16, 6.8))
    y = np.arange(len(s))
    edge = ["black" if h else "none" for h in s.hi]
    lw = [1.8 if h else 0.5 for h in s.hi]

    axF.barh(y, s.idf1, xerr=s.ci, color=s.color, edgecolor=edge, linewidth=lw,
             hatch=list(s.hatch), error_kw=dict(ecolor="#333", capsize=3))
    for yy, v, hi in zip(y, s.idf1, s.hi):
        axF.text(v + 0.008, yy, f"{v:.3f}", va="center", fontsize=9, fontweight="bold" if hi else "normal")
    axF.set_yticks(y); axF.set_yticklabels(s.label, fontsize=9)
    axF.set_xlim(0, 1.0); axF.set_xlabel("IDF1 — identity accuracy (higher better)")
    axF.set_title("A. Identity F1", fontweight="bold")
    axF.grid(axis="x", alpha=0.3)

    hh = 0.38
    axS.barh(y + hh / 2, s.raw, height=hh, color=s.color, alpha=0.35, edgecolor=edge, linewidth=lw,
             hatch=list(s.hatch), label="raw (all transitions)")
    axS.barh(y - hh / 2, s.sus, height=hh, color=s.color, edgecolor=edge, linewidth=lw,
             hatch=list(s.hatch), label="sustained (>=30-frame runs)")
    for yy, rv, sv, hi in zip(y, s.raw, s.sus, s.hi):
        axS.text(rv, yy + hh / 2, f" {rv:,}", va="center", fontsize=7.5, alpha=0.7)
        axS.text(sv, yy - hh / 2, f" {sv:,}", va="center", fontsize=8.5, fontweight="bold" if hi else "normal")
    axS.set_yticks(y); axS.set_yticklabels([])
    axS.set_xlabel("ID switches (lower better)")
    axS.set_title("B. ID switches — raw vs SUSTAINED", fontweight="bold", fontsize=10)
    axS.set_xlim(0, s.raw.max() * 1.13)
    axS.legend(loc="lower right", fontsize=8)
    axS.grid(axis="x", alpha=0.3)

    fig.suptitle("Improving detections-only ID tracking — multi-animal sessions (252 cam-sessions, ≥2 animals)\n"
                 "HATCHED 'UDMT' bar is OFF-POOL (own detector) and computed on the hardest 4-animal session ONLY "
                 "(6 cam-sessions) — not a like-for-like aggregate; see figure_udmt_hardsession_percam.png",
                 fontsize=11, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    PLOTS.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(PLOTS / f"figure_method_comparison.{ext}", dpi=150, bbox_inches="tight")
    print("wrote", PLOTS / "figure_method_comparison.png")
    plt.close(fig)


if __name__ == "__main__":
    main()
