"""Method-comparison figure (multi-animal, honest metrics).

Compares the detections-only ID trackers and the improvement methods on the two
metrics that matter: IDF1 (identity accuracy) and SUSTAINED switches (real identity
errors, brief <30-frame flicker removed). MULTI-ANIMAL sessions only (animals>=2).

Headline: feeding Liezl the sleap-nn FILTER+TRACK pool (keeptrack) instead of the
filter-only pool is the big win; 3D-weight tuning is a smaller stackable gain.

Inputs: _liezlsweep/switch_breakdown.tsv (raw,sustained) + eval CSVs / metrics CSV.
Output: outputs/PAF_3d_kalman/plots/figure_method_comparison.{png,pdf}
"""
from __future__ import annotations
from pathlib import Path
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
PAF = REPO / "outputs" / "PAF_3d_kalman"
METRICS = PAF / "metrics" / "per_camera_session_metrics.csv"
BD = PAF / "_liezlsweep" / "switch_breakdown.tsv"
PLOTS = PAF / "plots"

# breakdown-key, label, idf1-source, color, highlight
ROWS = [
    ("detok_bytetrack", "ByteTrack", ("metrics", "detok_bytetrack"), "#c2185b", False),
    ("detok_luc3d_results", "LUC3D (3D)", ("csv", PAF/"_eval_detok_luc3d.csv"), "#1565c0", False),
    ("detok_sleap_ref", "SLEAP per-camera 2D (ref)", ("metrics", "detok_sleap_ref"), "#9e9e9e", False),
    ("detok_liezl_results", "Liezl — filter-only detections", ("csv", PAF/"_eval_detok_liezl.csv"), "#66bb6a", False),
    ("detok_liezl_reid_results", "Liezl + appearance ReID (unsup.)", ("csv", PAF/"_eval_detok_liezl_reid.csv"), "#6a1b9a", True),
    ("sweep_C_3dheavy", "Liezl — filter-only + 3D-weight×3", (
        "csv", PAF/"_liezlsweep/eval_C_3dheavy.csv"), "#2e7d32", False),
    ("keeptrack_liezl", "Liezl — sleap-nn FILTER+TRACK pool", (
        "csv", PAF/"_eval_keeptrack_liezl.csv"), "#e65100", True),
    # optional combine (Method 1 config G); included if present
    ("G_keeptrack_3d6", "Liezl — FILTER+TRACK + 3D-weight×6", (
        "csv", PAF/"_liezlsweep/eval_G_keeptrack_3d6.csv"), "#bf360c", True),
    ("_swapdp_keeptrack_3d6", "Liezl — FILTER+TRACK + 3D swap-fix (swap-dp)", (
        "csv", PAF/"_eval_swapdp_keeptrack_3d6.csv"), "#00838f", True),
]
MIN = 2


def idf1(src):
    kind, ref = src
    if kind == "csv":
        p = Path(ref)
        if not p.exists():
            return None, None
        df = pd.read_csv(p); df = df[df.tracker == "luc3d"]
    else:
        df = pd.read_csv(METRICS); df = df[df.tracker == ref]
    df = df[df.animals >= MIN]
    a = df.idf1.dropna().values
    return df.idf1.mean(), (1.96*a.std(ddof=1)/np.sqrt(len(a)) if len(a) > 1 else 0)


def main():
    bd = pd.read_csv(BD, sep="\t", header=None, names=["k", "raw", "sus"]).drop_duplicates("k", keep="last").set_index("k")
    recs = []
    for key, label, src, color, hi in ROWS:
        if key not in bd.index:
            print(f"skip {key} (no breakdown)"); continue
        f1, ci = idf1(src)
        if f1 is None:
            print(f"skip {key} (no eval csv yet)"); continue
        recs.append(dict(label=label, color=color, hi=hi,
                         raw=int(bd.loc[key, "raw"]), sus=int(bd.loc[key, "sus"]), idf1=f1, ci=ci))
    s = pd.DataFrame(recs)
    # order by IDF1 ascending so best is on top after invert
    s = s.sort_values("idf1").reset_index(drop=True)
    print(s[["label", "raw", "sus", "idf1"]].to_string(index=False))

    mpl.rcParams["font.family"] = "DejaVu Sans"
    fig, (axF, axS) = plt.subplots(1, 2, figsize=(16, 6.5))
    y = np.arange(len(s))
    edge = ["black" if h else "none" for h in s.hi]
    lw = [1.8 if h else 0.5 for h in s.hi]

    axF.barh(y, s.idf1, xerr=s.ci, color=s.color, edgecolor=edge, linewidth=lw,
             error_kw=dict(ecolor="#333", capsize=3))
    for yy, v, hi in zip(y, s.idf1, s.hi):
        axF.text(v+0.008, yy, f"{v:.3f}", va="center", fontsize=9, fontweight="bold" if hi else "normal")
    axF.set_yticks(y); axF.set_yticklabels(s.label, fontsize=9)
    axF.set_xlim(0, 1.0); axF.set_xlabel("IDF1 — identity accuracy (higher better)")
    axF.set_title("A. Identity F1", fontweight="bold")
    axF.grid(axis="x", alpha=0.3)

    hh = 0.38
    axS.barh(y + hh/2, s.raw, height=hh, color=s.color, alpha=0.35,
             edgecolor=edge, linewidth=lw, label="raw (all transitions)")
    axS.barh(y - hh/2, s.sus, height=hh, color=s.color,
             edgecolor=edge, linewidth=lw, label="sustained (>=30-frame runs)")
    for yy, rv, sv, hi in zip(y, s.raw, s.sus, s.hi):
        axS.text(rv, yy + hh/2, f" {rv:,}", va="center", fontsize=7.5, alpha=0.7)
        axS.text(sv, yy - hh/2, f" {sv:,}", va="center", fontsize=8.5, fontweight="bold" if hi else "normal")
    axS.set_yticks(y); axS.set_yticklabels([])
    axS.set_xlabel("ID switches (lower better)")
    axS.set_title("B. ID switches — raw vs SUSTAINED\n(sustained = honest error count, <30-frame flicker removed)",
                  fontweight="bold", fontsize=10)
    axS.set_xlim(0, s.raw.max()*1.13)
    axS.legend(loc="lower right", fontsize=8)
    axS.grid(axis="x", alpha=0.3)

    fig.suptitle("Improving detections-only ID tracking — multi-animal sessions (252 cam-sessions, ≥2 animals)\n"
                 "orange = the big win: feed Liezl the sleap-nn FILTER+TRACK pool.  A 'switch' = one identity "
                 "TRANSITION (a 300-frame swap counts as 1, not 300); SUSTAINED drops <30-frame flicker = honest count",
                 fontsize=11.5, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    PLOTS.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        p = PLOTS / f"figure_method_comparison.{ext}"
        fig.savefig(p, dpi=150, bbox_inches="tight")
    print("wrote", PLOTS / "figure_method_comparison.png")
    plt.close(fig)


if __name__ == "__main__":
    main()
