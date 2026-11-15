"""PER-CAMERA version of the method-comparison analysis (multi-animal sessions).

The aggregate figure (`make_plots_compare.py`) averages every metric over all 6
cameras. This script breaks the SAME methods out per camera, so you can see which
views are hard (e.g. side-on `backL` vs overhead `top`) and whether a method's win
is uniform across the camera array or carried by a few views.

Two heatmaps (methods x cameras), multi-animal only (animals>=2):
  A. IDF1 per camera (higher = better identity accuracy)
  B. raw ID switches per camera, summed over sessions (lower = better)

IDF1 per (method,camera) = mean over multi-animal cam-sessions for that camera.
Switches per (method,camera) = SUM of raw num_switches over those cam-sessions.
(Sustained switches are only computed in aggregate by switch_breakdown.py, so this
per-camera view uses raw switches and says so.)

Output: outputs/PAF_3d_kalman/plots/figure_method_comparison_percam.{png,pdf}
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
CAMERAS = ["back", "backL", "mid", "midL", "top", "topL"]
MIN_ANIMALS = 2

# (label, idf1-source) — mirrors make_plots_compare.ROWS.
# source: ("csv", path)  -> that method's per-camera eval CSV (tracker=="luc3d" rows)
#         ("metrics", tracker_name) -> rows of the shared metrics CSV
METHODS = [
    ("ByteTrack", ("metrics", "detok_bytetrack")),
    ("LUC3D (3D)", ("csv", PAF / "_eval_detok_luc3d.csv")),
    ("SLEAP per-cam 2D (ref)", ("metrics", "detok_sleap_ref")),
    ("Liezl — filter-only", ("csv", PAF / "_eval_detok_liezl.csv")),
    ("Liezl + ReID (unsup.)", ("csv", PAF / "_eval_detok_liezl_reid.csv")),
    ("Liezl — filter + 3D×3", ("csv", PAF / "_liezlsweep/eval_C_3dheavy.csv")),
    ("Liezl — FILTER+TRACK pool", ("csv", PAF / "_eval_keeptrack_liezl.csv")),
    ("Liezl — FILTER+TRACK + 3D×6", ("csv", PAF / "_liezlsweep/eval_G_keeptrack_3d6.csv")),
    ("Liezl — FILTER+TRACK + swap-dp", ("csv", PAF / "_eval_swapdp_keeptrack_3d6.csv")),
]


def load(src) -> pd.DataFrame | None:
    kind, ref = src
    if kind == "csv":
        p = Path(ref)
        if not p.exists():
            return None
        df = pd.read_csv(p)
        df = df[df.tracker == "luc3d"]
    else:
        df = pd.read_csv(METRICS)
        df = df[df.tracker == ref]
    df = df[df.animals >= MIN_ANIMALS]
    return df if len(df) else None


def main():
    labels, idf1_rows, sw_rows = [], [], []
    for label, src in METHODS:
        df = load(src)
        if df is None:
            print(f"skip {label} (no data)")
            continue
        idf1_by_cam, sw_by_cam = [], []
        for cam in CAMERAS:
            d = df[df.camera == cam]
            idf1_by_cam.append(d.idf1.dropna().mean() if len(d) else np.nan)
            sw_by_cam.append(int(d.num_switches.fillna(0).sum()) if len(d) else 0)
        labels.append(label)
        idf1_rows.append(idf1_by_cam)
        sw_rows.append(sw_by_cam)

    idf1 = np.array(idf1_rows, dtype=float)
    sw = np.array(sw_rows, dtype=float)
    n = len(labels)
    print(f"{n} methods x {len(CAMERAS)} cameras")

    mpl.rcParams["font.family"] = "DejaVu Sans"
    fig, (axF, axS) = plt.subplots(1, 2, figsize=(15, 1.1 * n + 2.5))

    # A. IDF1 heatmap (higher better -> greens)
    im1 = axF.imshow(idf1, aspect="auto", cmap="YlGn", vmin=np.nanmin(idf1), vmax=np.nanmax(idf1))
    axF.set_xticks(range(len(CAMERAS))); axF.set_xticklabels(CAMERAS, rotation=30, ha="right")
    axF.set_yticks(range(n)); axF.set_yticklabels(labels, fontsize=9)
    axF.set_title("A. IDF1 per camera (higher better)", fontweight="bold")
    for i in range(n):
        for j in range(len(CAMERAS)):
            v = idf1[i, j]
            if not np.isnan(v):
                axF.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8,
                         color="black" if v < 0.6 else "white")
    fig.colorbar(im1, ax=axF, fraction=0.046, pad=0.04, label="IDF1")

    # B. raw switches heatmap (lower better -> reds, log-ish via sqrt for contrast)
    sw_disp = np.sqrt(sw)
    im2 = axS.imshow(sw_disp, aspect="auto", cmap="Reds", vmin=0, vmax=np.nanmax(sw_disp))
    axS.set_xticks(range(len(CAMERAS))); axS.set_xticklabels(CAMERAS, rotation=30, ha="right")
    axS.set_yticks(range(n)); axS.set_yticklabels([])
    axS.set_title("B. raw ID switches per camera (lower better)", fontweight="bold")
    for i in range(n):
        for j in range(len(CAMERAS)):
            axS.text(j, i, f"{int(sw[i, j]):,}", ha="center", va="center", fontsize=8,
                     color="black" if sw_disp[i, j] < 0.6 * np.nanmax(sw_disp) else "white")
    fig.colorbar(im2, ax=axS, fraction=0.046, pad=0.04, label="sqrt(switches)")

    fig.suptitle("Per-camera method comparison — multi-animal sessions (animals>=2)\n"
                 "Same methods as figure_method_comparison.png, broken out by camera view. "
                 "Switches are RAW per-camera (summed over sessions).",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    PLOTS.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(PLOTS / f"figure_method_comparison_percam.{ext}", dpi=150, bbox_inches="tight")
    print("wrote", PLOTS / "figure_method_comparison_percam.png")
    plt.close(fig)


if __name__ == "__main__":
    main()
