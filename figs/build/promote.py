"""Promote a FIG_TAG=<t> series into the main tree: figures/fig{N}<t>/fig{N}<t><rest>
-> figures/fig{N}/fig{N}<rest>, data/fig{N}<t>/* -> data/fig{N}/*.

    python promote.py FIGS_DIR TAG FIGNUMS(comma)
"""
import shutil
import sys
from pathlib import Path

figs, tag, nums = Path(sys.argv[1]), sys.argv[2], sys.argv[3].split(",")
for n in nums:
    src, dst = figs / "figures" / f"fig{n}{tag}", figs / "figures" / f"fig{n}"
    k = 0
    for p in sorted(src.iterdir()):
        name = p.name
        pre = f"fig{n}{tag}"
        if name.startswith(pre):
            name = f"fig{n}" + name[len(pre):]
        if name == f"fig{n}.pdf" or name == f"fig{n}.png":
            continue                      # the tagged composite; main is re-assembled
        shutil.copy2(p, dst / name)
        k += 1
    dsrc, ddst = figs / "data" / f"fig{n}{tag}", figs / "data" / f"fig{n}"
    m = 0
    if dsrc.exists():
        ddst.mkdir(parents=True, exist_ok=True)
        for p in sorted(dsrc.iterdir()):
            shutil.copy2(p, ddst / p.name)
            m += 1
    print(f"fig{n}: {k} figure files, {m} data files promoted")
