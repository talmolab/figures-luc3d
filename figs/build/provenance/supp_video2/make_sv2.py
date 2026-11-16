"""Re-render Supplementary Video 2 with the 55-session dataset mean (read-only on Talmo's sandbox).

Runs talmo/2026-09-13-sfari-sandbox/projects/LUC3D/supp_composite.py unchanged, with
height_trace_anim's module paths pointed at this machine and its fig5 event source set to
the series being built. Only the bottom-right panel's dataset-mean curve and its
"mean of N displays" label depend on the series; the orbit, overlay and side-view frames of
session 20250905_210254 are reused as rendered.

    cd figs && $PY build/provenance/supp_video2/make_sv2.py check      # rebuild the b mean, compare with the sandbox cache
    cd figs && PATH=<dir with ffmpeg>:$PATH $PY build/provenance/supp_video2/make_sv2.py q OUT.mp4  # render with the 55-session events
"""
import json, sys
from pathlib import Path

import numpy as np
from matplotlib import font_manager

SANDBOX = Path("/root/vast/talmo/2026-09-13-sfari-sandbox")
LUC3D = SANDBOX / "projects/LUC3D"
FIGS = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(LUC3D))

import height_trace_anim as hta  # noqa: E402

for ttf in sorted((FIGS / "src" / "fonts").glob("*.ttf")):
    font_manager.fontManager.addfont(str(ttf))
hta.FIGS = FIGS
hta.BMIMICA = "/root/vast/eric/BMimica"


def use(series):
    hta.EVENTS_JSON = FIGS / "out" / f"fig5_upright_{series}.json"
    hta.POP_CACHE = HERE / f"peak_aligned_mean_{series}.npz"


mode = sys.argv[1]
if mode == "check":
    use("b")
    t, m, n = hta.population_curve()
    ref = np.load(LUC3D / "data/peak_aligned_mean.npz")
    print("b rebuild n", n, "cache n", int(ref["n"]),
          "| t equal", np.array_equal(t, ref["t"]),
          "| mean max abs diff", float(np.nanmax(np.abs(m - ref["mean"]))))
    ev = {s: [(e["start_frame"], e["end_frame"]) for e in json.load(open(FIGS / f"out/fig5_upright_{s}.json"))["events"]
              if e["session"] == "20250905_210254"] for s in "bq"}
    print("210254 events identical in b and q:", ev["b"] == ev["q"], len(ev["q"]))
    use("q")
    t, m, n = hta.population_curve()
    print("q n", n)
else:
    use(mode)
    import os
    os.chdir(SANDBOX)                     # supp_composite reads its frames from a relative R
    import supp_composite as sc
    sys.argv = ["supp_composite.py", "--session", "20250905_210254",
                "--start", "165303", "--end", "167420", "--out", sys.argv[2]]
    sc.main()
