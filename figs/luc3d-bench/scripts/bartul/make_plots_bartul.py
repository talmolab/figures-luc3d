#!/usr/bin/env python
"""Bartul 2-animal method-comparison figure (+ per-camera breakdown), styled
after outputs/PAF_3d_kalman/plots/figure_method_comparison_small.png."""
import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path("/root/vast/eric/luc3d-bench/outputs/bartul_smoke")
PLOTS = OUT / "plots"; PLOTS.mkdir(exist_ok=True)
CSV = OUT / "bartul_eval_per_camera.csv"
SERIALS = ["21241563","21369048","21372315","21372316","22085397"]
CAMLAB = {s: f"Cam {chr(65+i)}\n{s}" for i, s in enumerate(SERIALS)}

COLORS = {
    "Liezl — 3D-weight×6": "#d24a00", "Liezl ×6 + 3D swap-fix": "#0f7d8c",
    "Liezl ×6 + appearance ReID": "#6a1fa0", "Liezl — 3D-weight×3": "#2e7d32",
    "Liezl (3D)": "#8bc34a", "LUC3D (3D)": "#1f6fd0",
    "SLEAP per-camera 2D (ref)": "#8a8a8a", "ByteTrack": "#c2185b",
    "UDMT (off-pool, own det.)": "#5d4037", "UDMT (off-pool, centroid-scored)": "#5d4037",
    "3D-MuPPET (SORT+xview)": "#00897b",
}
HATCH = {"UDMT (off-pool, own det.)": "///"}


def main():
    df = pd.read_csv(CSV, dtype={"camera": str})
    g = df.groupby("tracker").agg(idf1=("idf1","mean"),
                                  sw=("num_switches","sum")).reset_index()
    g = g.sort_values("idf1", ascending=True)  # bottom->top ascending for barh (best on top)
    labels = g.tracker.tolist()
    colors = [COLORS.get(l, "#777") for l in labels]

    fig, (a, b) = plt.subplots(1, 2, figsize=(15.5, 0.55*len(labels)+2.2))
    y = np.arange(len(labels))
    # A: IDF1
    for i, l in enumerate(labels):
        a.barh(i, g.idf1.iloc[i], color=colors[i], hatch=HATCH.get(l, ""),
               edgecolor="black", linewidth=0.7)
        a.text(g.idf1.iloc[i]+0.008, i, f"{g.idf1.iloc[i]:.3f}", va="center",
               fontweight="bold" if g.idf1.iloc[i] >= g.idf1.max()-1e-9 else "normal", fontsize=10)
    a.set_yticks(y); a.set_yticklabels(labels, fontsize=10)
    a.set_xlim(0, 1.0); a.set_xlabel("IDF1 — identity accuracy (higher better)")
    a.set_title("A. Identity F1", fontweight="bold")
    # B: switches
    for i, l in enumerate(labels):
        b.barh(i, g.sw.iloc[i], color=colors[i], hatch=HATCH.get(l, ""),
               edgecolor="black", linewidth=0.7)
        b.text(g.sw.iloc[i]+g.sw.max()*0.01, i, f"{int(g.sw.iloc[i]):,}", va="center", fontsize=10)
    b.set_yticks(y); b.set_yticklabels([])
    b.set_xlabel("ID switches — raw, summed over 5 cameras (lower better)")
    b.set_title("B. ID switches (raw)", fontweight="bold")
    b.set_xlim(0, g.sw.max()*1.18 + 1)

    nfr = int(df.num_frames.mean())
    fig.suptitle(
        "All trackers on ONE Bartul video — session 20250827_141755 (2-animal), "
        "5 cameras: " + ", ".join(SERIALS) + "\n"
        f"Detections = RAW SLEAP model keypoints, track identity STRIPPED (each tracker re-derives identity). "
        f"GT = reprojected proofread 3D. Same session+cameras, first {nfr:,} frames; switches RAW per-camera summed.",
        fontsize=10.5, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    for ext in ("png", "pdf"):
        fig.savefig(PLOTS / f"figure_method_comparison_bartul.{ext}", dpi=140, bbox_inches="tight")
    print("wrote", PLOTS / "figure_method_comparison_bartul.png")

    # ---- per-camera breakdown ----
    order = g.sort_values("idf1", ascending=False).tracker.tolist()
    piv_id = df.pivot_table(index="tracker", columns="camera", values="idf1").reindex(order)[SERIALS]
    piv_sw = df.pivot_table(index="tracker", columns="camera", values="num_switches").reindex(order)[SERIALS]

    fig2, (h1, h2) = plt.subplots(1, 2, figsize=(15.5, 0.5*len(order)+2.4))
    for ax, piv, title, fmt, cmap in [
        (h1, piv_id, "Per-camera IDF1 (higher better)", "{:.2f}", "YlGn"),
        (h2, piv_sw, "Per-camera raw ID switches (lower better)", "{:.0f}", "YlOrRd")]:
        im = ax.imshow(piv.values, aspect="auto", cmap=cmap)
        ax.set_xticks(range(len(SERIALS))); ax.set_xticklabels([CAMLAB[s] for s in SERIALS], fontsize=8)
        ax.set_yticks(range(len(order))); ax.set_yticklabels(order, fontsize=9)
        for i in range(piv.shape[0]):
            for j in range(piv.shape[1]):
                v = piv.values[i, j]
                if np.isfinite(v):
                    ax.text(j, i, fmt.format(v), ha="center", va="center", fontsize=8)
        ax.set_title(title, fontweight="bold")
        fig2.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig2.suptitle("Bartul session 20250827_141755 — per-camera breakdown", fontsize=12, fontweight="bold")
    fig2.tight_layout(rect=[0, 0, 1, 0.95])
    for ext in ("png", "pdf"):
        fig2.savefig(PLOTS / f"figure_per_camera_breakdown_bartul.{ext}", dpi=140, bbox_inches="tight")
    print("wrote", PLOTS / "figure_per_camera_breakdown_bartul.png")


if __name__ == "__main__":
    main()
