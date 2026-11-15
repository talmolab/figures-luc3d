"""Where do the switches concentrate? Stratify by #animals and session difficulty.

Shows switches-per-session and IDF1 broken down by num_animals (2/3/4) and by the
master-sheet difficulty rating, for the winning keeptrack pipeline vs the filter-only
baseline. Multi-animal sessions only. Tells us whether the residual swaps are the
tractable kind (spread out) or the irreducible close-encounter kind (concentrated in
hard / many-animal sessions).

Output: outputs/PAF_3d_kalman/plots/figure_switches_stratified.{png,pdf}
"""
from __future__ import annotations
from pathlib import Path
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
PAF = REPO / "outputs" / "PAF_3d_kalman"
PLOTS = PAF / "plots"

SERIES = [
    ("Liezl — filter-only", PAF / "_eval_detok_liezl.csv", "#66bb6a"),
    ("Liezl — FILTER+TRACK (keeptrack)", PAF / "_eval_keeptrack_liezl.csv", "#e65100"),
    ("Liezl — FILTER+TRACK + 3D-weight×6 (champion)", PAF / "_liezlsweep/eval_G_keeptrack_3d6.csv", "#bf360c"),
]


def per_session(csv):
    df = pd.read_csv(csv); df = df[(df.tracker == "luc3d") & (df.animals >= 2)]
    # collapse 6 cameras -> per session (sum switches, mean idf1), keep strata
    g = df.groupby("session").agg(sw=("num_switches", "sum"), idf1=("idf1", "mean"),
                                  animals=("animals", "first"), difficulty=("difficulty", "first"))
    return g


def main():
    data = {label: per_session(csv) for label, csv, _ in SERIES}
    colors = {label: c for label, _, c in SERIES}

    mpl.rcParams["font.family"] = "DejaVu Sans"
    fig, axes = plt.subplots(2, 2, figsize=(15, 9))

    def grouped_bar(ax, by, agg, ylabel, title, fmt="{:.0f}"):
        keys = sorted(set().union(*[set(d[by].unique()) for d in data.values()]))
        x = np.arange(len(keys)); n = len(data); w = 0.8 / n
        for i, (label, d) in enumerate(data.items()):
            vals, ns = [], []
            for k in keys:
                s = d[d[by] == k]
                vals.append(getattr(s[agg], "mean")() if len(s) else 0)
                ns.append(len(s))
            off = (i - (n - 1) / 2) * w
            ax.bar(x + off, vals, w, color=colors[label], label=label,
                   edgecolor="black", linewidth=0.5)
            for xi, v in zip(x + off, vals):
                ax.text(xi, v, fmt.format(v), ha="center", va="bottom", fontsize=6.5)
        ax.set_xticks(x)
        ax.set_xticklabels([f"{k}\n(n={n})" for k, n in zip(keys, ns)], fontsize=8)
        ax.set_ylabel(ylabel); ax.set_title(title, fontweight="bold", fontsize=10)
        ax.grid(axis="y", alpha=0.3)

    grouped_bar(axes[0, 0], "animals", "sw", "switches / session",
                "A. Switches per session, by # animals")
    grouped_bar(axes[0, 1], "difficulty", "sw", "switches / session",
                "B. Switches per session, by difficulty rating")
    grouped_bar(axes[1, 0], "animals", "idf1", "IDF1",
                "C. IDF1 by # animals", fmt="{:.2f}")
    grouped_bar(axes[1, 1], "difficulty", "idf1", "IDF1",
                "D. IDF1 by difficulty rating", fmt="{:.2f}")
    axes[1, 0].set_ylim(0, 1); axes[1, 1].set_ylim(0, 1)
    axes[0, 0].set_xlabel("# animals"); axes[1, 0].set_xlabel("# animals")
    axes[0, 1].set_xlabel("difficulty (master sheet)"); axes[1, 1].set_xlabel("difficulty (master sheet)")
    axes[0, 0].legend(fontsize=8, loc="upper left")

    fig.suptitle("Where do the ID switches concentrate? — multi-animal sessions, by #animals & difficulty\n"
                 "switches pile up in hard (diff-7) / many-animal sessions — i.e. at frequent CLOSE encounters; "
                 "3D-weight×6 lowers them further",
                 fontsize=12.5, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    PLOTS.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(PLOTS / f"figure_switches_stratified.{ext}", dpi=150, bbox_inches="tight")
    print("wrote", PLOTS / "figure_switches_stratified.png")
    plt.close(fig)


if __name__ == "__main__":
    main()
