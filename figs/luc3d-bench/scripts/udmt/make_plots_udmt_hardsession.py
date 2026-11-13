"""Per-camera comparison ON the hardest 4-animal session (10072022145420), all methods
restricted to THAT session's 6 cameras, with UDMT alongside.

This is the apples-to-apples home for UDMT: every method is scored on the same session,
same cameras, same proofread GT. (The aggregate figure averages 252 cam-sessions, where a
single-session UDMT bar would be misleading.)

Existing trackers come from their per-method eval CSVs / the metrics CSV, filtered to
session 10072022145420. UDMT comes from _eval_udmt_hardsession.csv (scripts/udmt/score_udmt.py),
labeled OFF-POOL (own detector, not the shared sleap-nn pool).

Output: outputs/PAF_3d_kalman/plots/figure_udmt_hardsession_percam.{png,pdf}
"""
from __future__ import annotations
from pathlib import Path
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
PAF = REPO / "outputs" / "PAF_3d_kalman"
METRICS = PAF / "metrics" / "per_camera_session_metrics.csv"
PLOTS = PAF / "plots"
SESSION = "10072022145420"
CAMERAS = ["back", "backL", "mid", "midL", "top", "topL"]
UDMT_CSV = PAF / "_eval_udmt_hardsession.csv"

METHODS = [
    ("ByteTrack", ("metrics", "detok_bytetrack")),
    ("LUC3D (3D)", ("csv", PAF / "_eval_detok_luc3d.csv")),
    ("SLEAP per-cam 2D (ref)", ("metrics", "detok_sleap_ref")),
    ("Liezl — filter-only", ("csv", PAF / "_eval_detok_liezl.csv")),
    ("Liezl + ReID (unsup.)", ("csv", PAF / "_eval_detok_liezl_reid.csv")),
    ("Liezl — FILTER+TRACK pool", ("csv", PAF / "_eval_keeptrack_liezl.csv")),
    ("Liezl — FILTER+TRACK + swap-dp", ("csv", PAF / "_eval_swapdp_keeptrack_3d6.csv")),
]


def load(src):
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
    labels, idf1_rows, sw_rows = [], [], []
    for label, src in METHODS:
        df = load(src)
        if df is None:
            print("skip", label); continue
        idf1_rows.append([df[df.camera == c].idf1.mean() if len(df[df.camera == c]) else np.nan for c in CAMERAS])
        sw_rows.append([int(df[df.camera == c].num_switches.fillna(0).sum()) if len(df[df.camera == c]) else 0 for c in CAMERAS])
        labels.append(label)

    # UDMT (off-pool) if scored
    if UDMT_CSV.exists():
        u = pd.read_csv(UDMT_CSV)
        u = u[u.session.astype(str) == SESSION]
        if len(u):
            idf1_rows.append([u[u.camera == c].idf1.mean() if len(u[u.camera == c]) else np.nan for c in CAMERAS])
            sw_rows.append([int(u[u.camera == c].num_switches.fillna(0).sum()) if len(u[u.camera == c]) else 0 for c in CAMERAS])
            labels.append("UDMT (off-pool, own detector)")
            print("UDMT included")
    else:
        print("UDMT csv not present yet:", UDMT_CSV)

    idf1 = np.array(idf1_rows, float); sw = np.array(sw_rows, float)
    n = len(labels)
    mpl.rcParams["font.family"] = "DejaVu Sans"
    fig, (axF, axS) = plt.subplots(1, 2, figsize=(15, 1.05 * n + 2.6))

    im1 = axF.imshow(idf1, aspect="auto", cmap="YlGn", vmin=np.nanmin(idf1), vmax=np.nanmax(idf1))
    axF.set_xticks(range(6)); axF.set_xticklabels(CAMERAS, rotation=30, ha="right")
    axF.set_yticks(range(n)); axF.set_yticklabels(labels, fontsize=9)
    axF.set_title("A. IDF1 per camera (higher better)", fontweight="bold")
    for i in range(n):
        for j in range(6):
            v = idf1[i, j]
            if not np.isnan(v):
                axF.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8,
                         color="black" if v < 0.55 else "white")
    fig.colorbar(im1, ax=axF, fraction=0.046, pad=0.04, label="IDF1")

    im2 = axS.imshow(sw, aspect="auto", cmap="Reds", vmin=0, vmax=np.nanmax(sw) if np.nanmax(sw) > 0 else 1)
    axS.set_xticks(range(6)); axS.set_xticklabels(CAMERAS, rotation=30, ha="right")
    axS.set_yticks(range(n)); axS.set_yticklabels([])
    axS.set_title("B. raw ID switches per camera (lower better)", fontweight="bold")
    mx = np.nanmax(sw) if np.nanmax(sw) > 0 else 1
    for i in range(n):
        for j in range(6):
            axS.text(j, i, f"{int(sw[i, j])}", ha="center", va="center", fontsize=8,
                     color="black" if sw[i, j] < 0.6 * mx else "white")
    fig.colorbar(im2, ax=axS, fraction=0.046, pad=0.04, label="switches")

    fig.suptitle(f"Hardest 4-animal session {SESSION} — per-camera comparison (incl. UDMT)\n"
                 "All methods scored on the same session/cameras/GT. UDMT is OFF-POOL "
                 "(its own detector via SEEM+XMem+TrDiMP, not the shared sleap-nn pool).",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    PLOTS.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(PLOTS / f"figure_udmt_hardsession_percam.{ext}", dpi=150, bbox_inches="tight")
    print("wrote", PLOTS / "figure_udmt_hardsession_percam.png")
    plt.close(fig)


if __name__ == "__main__":
    main()
