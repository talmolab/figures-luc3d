"""Fig 3E on main (E2) vs what the md56 b-figure used. Run from the repo root."""
import hashlib
import json
from pathlib import Path

OUT = Path("figs/out")
NEW = sorted((OUT / "md56main").glob("fig3_*__distanceThreshold25-stale20_*_md56.json"))
VAR = sorted((OUT / "tmp").glob("headtohead_var_md56_2dc001fe*"))


def payload(p):
    b = p.read_bytes()
    i, j = b.find(b'"identities":'), b.find(b',"framesProcessed"')
    return hashlib.sha256(b[i:j] if i >= 0 and j > i else b).hexdigest()


def frames(p):
    d = json.loads(p.read_text())
    return d.get("identities"), d.get("frames")


print("new deposits:", [p.name for p in NEW])
# 1. greedy on main at {distanceThreshold 25, stale 20} vs the head-to-head greedy on main
#    (defaults, which are the same values): expected identical in every session.
if VAR:
    root = next(iter(sorted(VAR[0].glob("distanceThreshold25-stale20_*"))), None)
    same = diff = missing = 0
    bad = []
    for g in sorted(root.glob("*/*/greedy.json")) if root else []:
        ref = OUT / "tmp" / "headtohead_md56" / g.parent.parent.name / g.parent.name / "greedy.json"
        if not ref.exists():
            missing += 1
            continue
        if frames(g) == frames(ref):
            same += 1
        else:
            diff += 1
            bad.append(f"{g.parent.parent.name}/{g.parent.name}")
    print(f"greedy (E2, main) vs head-to-head greedy (main): identical {same}, "
          f"different {diff}, no reference {missing}  {bad[:8]}")


# 2. field-by-field: new quality deposit vs the one fig3b drew (eric/figs greedy + eric/figs
#    rescore), and vs fig3_quality_md56 (head-to-head greedy on main, eric/figs rescore).
def walk(o, path, out):
    if isinstance(o, dict):
        for k, v in o.items():
            walk(v, f"{path}.{k}", out)
    elif isinstance(o, list) and len(o) < 200:
        for i, v in enumerate(o):
            walk(v, f"{path}[{i}]", out)
    else:
        out[path] = o


SKIP = ("seconds", "variant", "generated", "wall", "runtime", "lucid", "driver", "cache", "date")
for kind, olds in (("quality", ["fig3_quality__distanceThreshold25-stale20-sync_e508a7ab_md56.json",
                                "fig3_quality_md56.json"]),
                   ("headtohead", ["fig3_headtohead__distanceThreshold25-stale20-sync_e508a7ab_md56.json",
                                   "fig3_headtohead_md56.json"])):
    new = [p for p in NEW if p.name.startswith(f"fig3_{kind}__")]
    if not new:
        print(f"!! no new {kind} deposit")
        continue
    a = {}
    walk(json.loads(new[0].read_text()), "", a)
    for old in olds:
        b = {}
        walk(json.loads((OUT / old).read_text()), "", b)
        keys = [k for k in set(a) & set(b) if not any(s in k.lower() for s in SKIP)]
        changed = sorted(k for k in keys if a[k] != b[k])
        print(f"\n{kind}: new vs {old}: {len(keys)} comparable fields, {len(changed)} differ")
        for k in changed[:25]:
            print(f"   {k}: {b[k]!r:.60} -> {a[k]!r:.60}")
