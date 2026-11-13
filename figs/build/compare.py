#!/usr/bin/env python3
"""Compare a rebuilt figs/ tree against a reference one (run_FINALBUILD.sh --verify).

    python build/compare.py BUILT_FIGS REF_FIGS [--since MARKER]

Pass/fail is decided on what the paper shows:
  * the seven composites figures/figN/figN.pdf must be PIXEL-IDENTICAL (rendered at 150 dpi);
  * every data/figN/*.csv the build wrote must be byte-identical.
Also reported, without failing: panel PDFs that differ (unplaced leftovers such as
fig2a_association do), and data CSVs the panel build does not write (with --since MARKER,
files older than MARKER). Those come from the measurement scripts, e.g. fig5s_*, fig7s*.
"""
import hashlib
import os
import sys
from pathlib import Path

import fitz
import numpy as np


def render(p, dpi=150):
    out = []
    for pg in fitz.open(p):
        pm = pg.get_pixmap(dpi=dpi, alpha=False)
        out.append(np.frombuffer(pm.samples, np.uint8).reshape(pm.h, pm.w, pm.n))
    return out


def pdf_diff(a, b):
    ra, rb = render(a), render(b)
    if len(ra) != len(rb):
        return f"pages {len(ra)} vs {len(rb)}"
    worst, npx = 0, 0
    for x, y in zip(ra, rb):
        if x.shape != y.shape:
            return f"size {x.shape[:2]} vs {y.shape[:2]}"
        d = np.abs(x.astype(int) - y.astype(int)).max(axis=2)
        worst, npx = max(worst, int(d.max())), npx + int((d > 0).sum())
    return "PIXEL-IDENTICAL" if worst == 0 else f"max {worst}/255 on {npx} px"


def md5(p):
    return hashlib.md5(Path(p).read_bytes()).hexdigest()


def main(argv):
    built, ref = Path(argv[1]), Path(argv[2])
    since = os.stat(argv[argv.index("--since") + 1]).st_mtime if "--since" in argv else None
    bad, info = [], []
    for n in range(1, 8):
        comp = Path("figures") / f"fig{n}" / f"fig{n}.pdf"
        if not (built / comp).exists():
            bad.append(f"{comp}: not built")
            continue
        r = "PIXEL-IDENTICAL" if md5(built / comp) == md5(ref / comp) else pdf_diff(built / comp, ref / comp)
        print(f"{comp}: {r}")
        if r != "PIXEL-IDENTICAL":
            bad.append(f"{comp}: {r}")
        for p in sorted((ref / "figures" / f"fig{n}").glob(f"fig{n}?*.pdf")):
            q = built / "figures" / f"fig{n}" / p.name
            if not q.exists():
                info.append(f"panel not rebuilt (unplaced): {q.relative_to(built)}")
            elif md5(p) != md5(q) and (r := pdf_diff(q, p)) != "PIXEL-IDENTICAL":
                info.append(f"panel differs (unplaced): {q.relative_to(built)}: {r}")
    for p in sorted(ref.glob("data/fig[1-7]/*.csv")):
        q = built / p.relative_to(ref)
        if not q.exists() or (since and q.stat().st_mtime < since):
            info.append(f"csv not written by the panel build: {p.relative_to(ref)}")
        elif md5(p) != md5(q):
            bad.append(f"{p.relative_to(ref)}: bytes differ")
    for line in info:
        print("  info:", line)
    print()
    for line in bad:
        print("FAIL:", line)
    print("VERIFY PASS: all seven composites pixel-identical, every rebuilt CSV identical"
          if not bad else f"VERIFY FAIL: {len(bad)} problem(s)")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
