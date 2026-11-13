"""Sessions left out of every scored result and of Fig 5 (the "q" series).

One list, read by every re-aggregation, so each figure drops the same sessions and the
change is one auditable file. A session is named here by its folder name; `excluded()`
also matches the video timestamp some deposits use instead (folder 20250904_183921,
videos 20250904183930).
"""

#: folder name -> (video timestamp, reason)
EXCLUDE = {
    "20250904_183921": (
        "20250904183930",
        "reference 3D corrupted over frames ~42k-115k: camera 21369048's identities are "
        "swapped in the lab's SLEAP predictions and were triangulated into the 3D "
        "(preflight QC, 2026-11-12)",
    ),
}

_NAMES = set(EXCLUDE) | {ts for ts, _ in EXCLUDE.values()}


def excluded(session) -> bool:
    """True if `session` (folder name or video timestamp, possibly inside a longer id
    such as a path) is on the exclusion list."""
    s = str(session)
    return any(n in s for n in _NAMES)


def provenance() -> dict:
    """What a re-aggregated deposit records about the exclusion."""
    return {k: reason for k, (_, reason) in EXCLUDE.items()}
