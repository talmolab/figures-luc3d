"""Single-session method comparison — the HONEST apples-to-apples version for UDMT.

Same methods/layout as figure_method_comparison.png, but every tracker (incl. UDMT) is
scored on the SAME ONE hard video (session 10072022145420, the hardest 4-animal session)
over the SAME set of cameras (the cameras UDMT has completed). Because everyone is on the
identical sessions+cameras here, UDMT's bar IS comparable (unlike the 252-cam-session
aggregate, where a 1-session UDMT bar is meaningless).

IDF1 = mean over the comparison cameras. Switches = SUM of raw num_switches over those
cameras (raw per-camera, since the "sustained" filter is only computed in aggregate).

Output: outputs/PAF_3d_kalman/plots/figure_method_comparison_small.{png,pdf}
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
PLOTS = PAF / "plots"
SESSION = "10072022145420"
UDMT_CSV = PAF / "_eval_udmt_hardsession.csv"

# (label, idf1/switch source, color, highlight) — mirrors make_plots_compare.ROWS
ROWS = [
    ("ByteTrack", ("metrics", "detok_bytetrack"), "#c2185b", False),
    ("LUC3D (3D)", ("csv", PAF / "_eval_detok_luc3d.csv"), "#1565c0", False),
    ("SLEAP per-camera 2D (ref)", ("metrics", "detok_sleap_ref"), "#9e9e9e", False),
    ("Liezl — filter-only detections", ("csv", PAF / "_eval_detok_liezl.csv"), "#66bb6a", False),
    ("Liezl + appearance ReID (unsup.)", ("csv", PAF / "_eval_detok_liezl_reid.csv"), "#6a1b9a", False),
    ("Liezl — filter-only + 3D-weight×3", ("csv", PAF / "_liezlsweep/eval_C_3dheavy.csv"), "#2e7d32", False),
    ("Liezl — sleap-nn FILTER+TRACK pool", ("csv", PAF / "_eval_keeptrack_liezl.csv"), "#e65100", True),
    ("Liezl — FILTER+TRACK + 3D-weight×6", ("csv", PAF / "_liezlsweep/eval_G_keeptrack_3d6.csv"), "#bf360c", True),
    ("Liezl — FILTER+TRACK + 3D swap-fix", ("csv", PAF / "_eval_swapdp_keeptrack_3d6.csv"), "#00838f", True),
]


def load(src) -> pd.DataFrame | None:
    kind, ref = src
    if kind == "csv":
        p = Path(ref)
        if not p.exists():
            return None
        df = pd.read_csv(p); df = df[df.tracker == "luc3d"]
    else:
        df = pd.read_csv(METRICS); df = df[df.tracker == ref]
    df = df[df.session.astype(str) == SESSION]
    return df if len(df) else None


def main():
    # comparison camera set = cameras UDMT has scored (strict apples-to-apples); fallback all
    udmt = pd.read_csv(UDMT_CSV) if UDMT_CSV.exists() else None
    if udmt is not None:
        udmt = udmt[udmt.session.astype(str) == SESSION]
    cams = sorted(udmt.camera.unique()) if udmt is not None and len(udmt) else \
        ["back", "backL", "mid", "midL", "top", "topL"]

    recs = []
    for label, src, color, hi in ROWS:
        df = load(src)
        if df is None:
            print("skip", label); continue
        d = df[df.camera.isin(cams)]
        if not len(d):
            continue
        recs.append(dict(label=label, color=color, hi=hi,
                         idf1=d.idf1.dropna().mean(),
                         sw=int(d.num_switches.fillna(0).sum())))
    if udmt is not None and len(udmt):
        recs.append(dict(label="UDMT (off-pool, own detector)", color="#5d4037", hi=True,
                         idf1=udmt.idf1.dropna().mean(),
                         sw=int(udmt.num_switches.fillna(0).sum())))

    s = pd.DataFrame(recs).sort_values("idf1").reset_index(drop=True)
    print(f"cameras compared ({len(cams)}): {cams}")
    print(s[["label", "idf1", "sw"]].to_string(index=False))

    mpl.rcParams["font.family"] = "DejaVu Sans"
    fig, (axF, axS) = plt.subplots(1, 2, figsize=(16, 6.5))
    y = np.arange(len(s))
    edge = ["black" if h else "none" for h in s.hi]
    lw = [1.8 if h else 0.5 for h in s.hi]
    hatch = ["///" if "UDMT" in l else "" for l in s.label]

    axF.barh(y, s.idf1, color=s.color, edgecolor=edge, linewidth=lw, hatch=hatch)
    for yy, v, hi in zip(y, s.idf1, s.hi):
        axF.text(v + 0.008, yy, f"{v:.3f}", va="center", fontsize=9, fontweight="bold" if hi else "normal")
    axF.set_yticks(y); axF.set_yticklabels(s.label, fontsize=9)
    axF.set_xlim(0, 1.0); axF.set_xlabel("IDF1 — identity accuracy (higher better)")
    axF.set_title("A. Identity F1", fontweight="bold")
    axF.grid(axis="x", alpha=0.3)

    axS.barh(y, s.sw, color=s.color, edgecolor=edge, linewidth=lw, hatch=hatch)
    for yy, sv in zip(y, s.sw):
        axS.text(sv, yy, f" {sv:,}", va="center", fontsize=8.5)
    axS.set_yticks(y); axS.set_yticklabels([])
    axS.set_xlabel("ID switches — raw, summed over the compared cameras (lower better)")
    axS.set_title("B. ID switches (raw)", fontweight="bold")
    axS.set_xlim(0, max(s.sw) * 1.15)
    axS.grid(axis="x", alpha=0.3)

    fig.suptitle(f"All trackers on ONE hard video — session {SESSION} (hardest 4-animal), "
                 f"{len(cams)} cameras: {', '.join(cams)}\n"
                 "Apples-to-apples: every method (incl. hatched UDMT, off-pool) scored on the "
                 "same session+cameras vs proofread GT. Switches are RAW per-camera summed.",
                 fontsize=11, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    PLOTS.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(PLOTS / f"figure_method_comparison_small.{ext}", dpi=150, bbox_inches="tight")
    print("wrote", PLOTS / "figure_method_comparison_small.png")
    plt.close(fig)


if __name__ == "__main__":
    main()
