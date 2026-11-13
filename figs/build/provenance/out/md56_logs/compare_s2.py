"""Compare the main-2dc001fe sweep (S2, *_md56main) against the eric/figs md56 deposits."""
import json
import sys
from pathlib import Path

OUT = Path("/root/vast/eric/sleap-3d-gui/scratch/repos/lucid/figs/out")
PAIRS = [
    ("current tracker",
     "fig3_sweep50__distanceThreshold25-stale20-sync_e508a7ab_md56.json",
     "fig3_sweep50__distanceThreshold25-stale20_f7733dbf_md56main.json"),
    ("previous default (old pre-#210) vs previous settings on main",
     "fig3_sweep50_md56.json",
     "fig3_sweep50__distanceThreshold50-stale0_ce1d7f8d_md56main.json"),
]
KEYS = ("within_idf1", "cross_idf1", "within_switches", "cross_switches", "within_idp", "within_idr",
        "cross_idp", "cross_idr", "per_camera_idf1", "camera_frames")


def rows(d):
    return {(c["corr2d"], c["corr3d"], r["session"]): r
            for c in d["cells"] for r in c["per_session"]}


rc = 0
for label, old, new in PAIRS:
    a, b = json.loads((OUT / old).read_text()), json.loads((OUT / new).read_text())
    print(f"\n== {label}\n   old {old}\n   new {new}")
    print(f"   new: sessions {len(b['sessions'])}, total_camera_frames {b['total_camera_frames']:,}, "
          f"commit {b.get('lucid_commit')}, node {b.get('node')}")
    print(f"   cells: {len(b['cells'])}, status {sorted({c['status'] for c in b['cells']})}, "
          f"n_sessions {sorted({c['n_sessions'] for c in b['cells']})}")
    if len(b["sessions"]) != 56 or b["total_camera_frames"] != a["total_camera_frames"]:
        print("   !! session count or camera-frame total differs from old")
        rc = 1
    ra, rb = rows(a), rows(b)
    common = sorted(set(ra) & set(rb))
    diff = [k for k in common if any(ra[k].get(x) != rb[k].get(x) for x in KEYS)]
    print(f"   per-session rows: {len(common)} compared, {len(diff)} with different scores")
    for k in diff[:10]:
        print("     ", k, {x: (ra[k].get(x), rb[k].get(x)) for x in KEYS
                            if ra[k].get(x) != rb[k].get(x)})
    print("   per-cell (r = corr3d): metric old -> new, switches old -> new")
    ca = {(c["corr2d"], c["corr3d"]): c for c in a["cells"]}
    metric = b.get("metric_key") or "idf1_cross"
    for c in sorted(b["cells"], key=lambda c: c["corr3d"]):
        o = ca.get((c["corr2d"], c["corr3d"]), {})
        mk = [k for k in ("idf1_cross", "idf1_within") if k in c]
        vals = "  ".join(f"{k} {o.get(k)} -> {c.get(k)}" for k in mk)
        print(f"     r={c['corr3d']:<5} {vals}  switches {o.get('switches')} -> {c.get('switches')}"
              + ("" if all(o.get(k) == c.get(k) for k in mk + ['switches']) else "   *CHANGED*"))
sys.exit(rc)
