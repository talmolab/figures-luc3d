#!/usr/bin/env python3
"""
Fig 3d -- the exhaustive hypothesis explosion (the old Fig 3b,
panels/fig2_03_cost_model.py) redrawn at THIRD span and a short row, to sit beside
the compact per-frame loop strip (Eric, 2026-10-27: "put the old exhaustive
hypothesis explosion next to c"). Same build(), same cap, same drawing -- only the
panel size differs, so nothing is re-measured. Writes
figures/fig2/fig2d_cost_model_third.pdf; fig2_03_cost_model.py's full-size panel and
its deposit are not touched (no deposit() call here).

    .venv/bin/python panels/fig2_20_cost_model_third.py [--h=40]
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import fig2_03_cost_model as cm  # noqa: E402
from src.data_loader import load  # noqa: E402
from src.style import use  # noqa: E402

FIGS = Path(__file__).resolve().parent.parent
# provenance marker for make_docs.panel_script: save(fig, 2, "d", "cost_model_third")
OUT = FIGS / "figures" / "fig2" / "fig2d_cost_model_third.pdf"


def main(h_mm=40.0):
    use()
    orig_panel = cm.panel
    # the four camera counts as a 2 x 2 key, so the band is two lines, not four
    cm.panel = lambda span, row, **kw: orig_panel("third", h_mm, **{**kw, "key": 2})
    orig_legend = cm.text_legend

    def legend_2x2(ax, entries, loc="above", **kw):
        orig_legend(ax, entries[0::2], "above", **kw)
        orig_legend(ax, entries[1::2], "above", **kw)
        # the second call drew at the same x: shift its texts right
        for t in ax.figure.texts[-len(entries[1::2]):]:
            t.set_x(t.get_position()[0] + 0.40)

    cm.text_legend = legend_2x2

    def save(fig, fig_no, letter, slug):
        OUT.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(OUT, dpi=600)
        fig.savefig(OUT.with_suffix(".png"), dpi=600)
        print(f"  wrote {OUT.relative_to(FIGS)}")

    cm.save = save
    cap = load("fig3_headtohead.json")["caps"]["max_hypotheses_per_frame"]
    cm.render(cm.build(), cap, "third", 2)


if __name__ == "__main__":
    main(next((float(a.split("=")[1]) for a in sys.argv if a.startswith("--h=")), 40.0))
