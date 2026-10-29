#!/usr/bin/env python3
"""
Fig 3c -- LUC3D's per-frame association loop, as one strip
(Eric, 2026-10-27: put the Hungarian step back after the anchor diagram replaced
the old grouping-strategies panel, which was its only drawing).

    frame start            one Hungarian solve per camera              frame end
    freeze anchors,  ->  [cam 1] -> [cam 2] -> ... -> [cam C]  ->  re-triangulate
    evict > N frames       anchors x detections, chosen pairs bold     (>= 2 views)
           ^------------------------- next frame ---------------------------'

Each camera block is a small bipartite graph: the frozen 3D anchors (rings, in the
identity hues of panels b-d) against that camera's detections (ink dots, no identity
yet). Grey lines are every pairing the cost matrix scores (the 2D + 3D terms of
panel b); the coloured lines are the assignment the Hungarian solve keeps. Different
cameras keep different permutations, because detection order is arbitrary per view.
The exhaustive count (A!)^C sits beside the O(C A^3) label so the contrast the old
panel drew survives in one line. Track initialisation (epipolar matching of
unmatched detections) is Methods-only and not drawn.

Writes figures/fig2/fig2c_algorithm_strip.pdf (+ .png) directly, the compact strip
that composites.fig3 places; --full writes the full-width fig2c_algorithm_strip_full.

    .venv/bin/python panels/fig2_19_algorithm_strip.py [--full] [--h=40]
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.diagram import icon  # noqa: E402
from src.style import INK, MUTED, SALMON, TEAL, mm, use  # noqa: E402

FIGS = Path(__file__).resolve().parent.parent
# provenance marker for make_docs.panel_script: save(fig, 2, "c", "algorithm_strip")
OUT = FIGS / "figures" / "fig2" / "fig2c_algorithm_strip_full.pdf"
#: compact (Eric, 2026-10-27: "put the old exhaustive hypothesis explosion next to
#: c"): the strip at two-thirds width beside the hypotheses-per-frame plot, so the
#: plot carries the exhaustive count and the strip drops its own (A!)^C note.
OUT_COMPACT = FIGS / "figures" / "fig2" / "fig2c_algorithm_strip.pdf"
COMPACT_W = 180.0 - 4.0 - (180.0 - 8.0) / 3         # PAGE_W - GUTTER - third span

W, H = 180.0, 31.0                     # mm; the axes ARE millimetres
#: tab10 blue / orange / green -- the animal hues of the hypothesis panel (b) and
#: the anchor diagram beside it, not the app palette of the retired panel a
ANIMALS = ["#1F77B4", "#FF7F0E", "#2CA02C"]
ROWS = [19.0, 14.5, 10.0]              # y of the three rows in every block
GREY_EDGE = "#D4D4D4"
#: per camera: anchor row a -> detection row KEEP[c][a]
KEEP = [(1, 0, 2), (0, 2, 1), (2, 1, 0)]
CAM_LABEL = ["cam 1", "cam 2", "cam C"]
#: x geometry per layout: frame-start centre, camera blocks' anchor columns,
#: anchor -> detection spacing, frame-end centre
GEOM = {"full": dict(xs=14.0, cams=[40.0, 66.0, 101.0], dx=12.0, xe=140.0),
        "compact": dict(xs=11.6, cams=[29.5, 48.0, 73.0], dx=10.0, xe=103.5)}
CAM_X, PAIR_DX = GEOM["full"]["cams"], GEOM["full"]["dx"]
FS, FS_SMALL = 7.0, 6.2


def arrow(ax, x0, x1, y, color=MUTED, lw=0.8):
    ax.add_patch(FancyArrowPatch((x0, y), (x1, y), arrowstyle="-|>",
                                 mutation_scale=6, color=color, lw=lw,
                                 shrinkA=0, shrinkB=0))


def anchor(ax, x, y, color, ms=5.2):
    ax.plot([x], [y], "o", ms=ms, mfc="white", mec=color, mew=1.4, zorder=5)


def detection(ax, x, y, ms=4.0):
    ax.plot([x], [y], "o", ms=ms, mfc=INK, mec="white", mew=0.5, zorder=5)


def camera_block(ax, x, keep, label, first):
    xd = x + PAIR_DX
    for a in range(3):
        for d in range(3):
            ax.plot([x, xd], [ROWS[a], ROWS[d]], color=GREY_EDGE, lw=0.6, zorder=2)
    for a, d in enumerate(keep):
        ax.plot([x, xd], [ROWS[a], ROWS[d]], color=ANIMALS[a], lw=1.5, zorder=3,
                solid_capstyle="round")
    for a in range(3):
        anchor(ax, x, ROWS[a], ANIMALS[a])
        detection(ax, xd, ROWS[a])
    icon(ax, "camera", xd - 1.9, 21.2, s=3.8, color=INK)
    ax.text(xd - 2.6, 23.1, label, ha="right", va="center", fontsize=FS_SMALL,
            color=INK)
    ax.text(x + PAIR_DX / 2, 6.4, "Hungarian", ha="center", va="center",
            fontsize=FS_SMALL, color=TEAL, fontweight="bold")
    if first:
        ax.text(x, 3.6, "anchors", ha="center", va="center", fontsize=FS_SMALL - 0.6,
                color=MUTED)
        ax.text(xd, 3.6, "detections", ha="center", va="center",
                fontsize=FS_SMALL - 0.6, color=MUTED)


def main(compact=False, h_out=None):
    global CAM_X, PAIR_DX
    g = GEOM["compact" if compact else "full"]
    CAM_X, PAIR_DX = g["cams"], g["dx"]
    xs, xe = g["xs"], g["xe"]
    w = COMPACT_W if compact else W
    h = h_out or H
    use()
    fig = plt.figure(figsize=(mm(w), mm(h)))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, w)
    ax.set_ylim(-(h - H) / 2, H + (h - H) / 2)        # content centred vertically
    ax.set_aspect("equal")
    ax.axis("off")

    # ---- frame start: the frozen anchors, stale views evicted
    ax.text(xs, 25.6, "frame start", ha="center", va="center", fontsize=FS,
            color=INK, fontweight="bold")
    for a, (dx, dy) in enumerate([(-5.0, 0.0), (0.0, -0.8), (5.0, 0.4)]):
        anchor(ax, xs + dx, 18.6 + dy, ANIMALS[a])
    ax.text(xs, 12.4, "freeze 3D anchors", ha="center", va="center",
            fontsize=FS_SMALL, color=INK)
    ax.text(xs, 8.9, "drop views not seen", ha="center", va="center",
            fontsize=FS_SMALL, color=MUTED)
    ax.text(xs, 6.0, "for > N frames", ha="center", va="center",
            fontsize=FS_SMALL, color=MUTED)
    arrow(ax, xs + 15.0 if not compact else xs + 11.4, CAM_X[0] - 3.5, ROWS[1])

    # ---- one Hungarian solve per camera, against the same frozen anchors
    for i, (x, keep, lab) in enumerate(zip(CAM_X, KEEP, CAM_LABEL)):
        camera_block(ax, x, keep, lab, first=(i == 0))
    arrow(ax, CAM_X[0] + PAIR_DX + 2.5, CAM_X[1] - 3.5, ROWS[1])
    xm = (CAM_X[1] + PAIR_DX + CAM_X[2]) / 2
    ax.text(xm, ROWS[1], "···", ha="center", va="center", fontsize=9, color=MUTED)
    arrow(ax, CAM_X[1] + PAIR_DX + 2.5, xm - 3.0, ROWS[1])
    arrow(ax, xm + 3.0, CAM_X[2] - 3.5, ROWS[1])
    span0, span1 = CAM_X[0] - 2.0, CAM_X[2] + PAIR_DX + 2.0
    ax.plot([span0, span0, span1, span1], [27.0, 28.0, 28.0, 27.0], color=TEAL, lw=0.7)
    ax.text((span0 + span1) / 2, 29.6, "C Hungarian solves per frame, O(C·A³)",
            ha="center", va="center", fontsize=FS, color=TEAL, fontweight="bold")
    arrow(ax, CAM_X[2] + PAIR_DX + 2.5, xe - 15.5 if not compact else xe - 12.5,
          ROWS[1])

    # ---- frame end: one re-triangulation per anchor
    ax.text(xe, 25.6, "frame end", ha="center", va="center", fontsize=FS,
            color=INK, fontweight="bold")
    icon(ax, "triangulate", xe - 3.4, 15.0, s=6.8, color=INK)
    ax.text(xe, 12.4, "re-triangulate anchors", ha="center", va="center",
            fontsize=FS_SMALL, color=INK)
    ax.text(xe, 8.9, "from ≥ 2 retained views", ha="center", va="center",
            fontsize=FS_SMALL, color=MUTED)

    # ---- the exhaustive count, for contrast with the bracket's O(C A^3)
    # (compact: the hypotheses-per-frame plot beside the strip says it instead)
    if not compact:
        _exhaustive_note(ax)

    _loop(ax, xs, xe)

    out = OUT_COMPACT if compact else OUT
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"), dpi=600)
    plt.close(fig)
    print(f"  wrote {out.relative_to(FIGS)}  ({w:.1f} x {h:.1f} mm)")


def _exhaustive_note(ax):
    ax.text(178.5, 20.0, "exhaustive:", ha="right", va="center", fontsize=FS_SMALL,
            color=SALMON, fontweight="bold")
    ax.text(178.5, 16.6, "(A!)$^C$ hypotheses", ha="right", va="center",
            fontsize=FS_SMALL, color=SALMON)
    ax.text(178.5, 13.4, "per frame", ha="right", va="center", fontsize=FS_SMALL,
            color=SALMON)


def _loop(ax, xs, xe):
    """Loop back to the next frame, under everything, clear of the
    anchors/detections key at y 3.6."""
    yl = 0.9
    ax.plot([xe, xe, xs], [6.6, yl, yl], color=MUTED, lw=0.7, ls=(0, (3, 2)))
    ax.add_patch(FancyArrowPatch((xs, yl), (xs, 4.6), arrowstyle="-|>",
                                 mutation_scale=6, color=MUTED, lw=0.7,
                                 shrinkA=0, shrinkB=0))
    ax.text(xe + 1.5, 2.6, "next frame", ha="left", va="center",
            fontsize=FS_SMALL - 0.6, color=MUTED)


if __name__ == "__main__":
    # compact at 40 mm is the placed panel, so a bare run (make_figures.py) builds it
    if "--full" in sys.argv:
        main()
    else:
        h = next((float(a.split("=")[1]) for a in sys.argv if a.startswith("--h=")),
                 40.0)
        main(compact=True, h_out=h)
