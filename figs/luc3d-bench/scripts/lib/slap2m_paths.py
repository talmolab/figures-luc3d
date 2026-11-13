"""Path resolution for SLAP-2M raw predictions, calibrations, and videos.

The master_sheet TSV references per-session paths but encodes them as
`*.predictions.proofread.slp.analysis.h5` (proofread). Raw, non-proofread
predictions live next to those files as `*.predictions.slp`. Sessions use
either `_h265_` or `_h265_10bit_` naming.
"""
from __future__ import annotations

from pathlib import Path

SLAP2M_ROOT = Path("/root/talmolab-smb/eric/slap_2m")
CAMERAS = ("back", "backL", "mid", "midL", "side", "sideL", "top", "topL")
BENCH_CAMERAS = ("back", "backL", "mid", "midL", "top", "topL")  # side/sideL excluded from benchmark per user direction (2026-05-15)


def session_dir(session_path: str) -> Path:
    return SLAP2M_ROOT / session_path


def raw_pred_slp(session_path: str, camera: str) -> Path | None:
    """Locate the raw, non-proofread `.predictions.slp` for one camera.

    Returns None if the camera folder is missing or no matching SLP exists.
    """
    cam_dir = session_dir(session_path) / camera
    if not cam_dir.exists():
        return None
    candidates = []
    for p in cam_dir.glob(f"{camera}-*.predictions.slp"):
        name = p.name
        if ".proofread." in name or ".static.externals." in name or ".reprojections." in name:
            continue
        candidates.append(p)
    if not candidates:
        return None
    if len(candidates) == 1:
        return candidates[0]
    candidates.sort(key=lambda p: (".predictions.slp" == p.name[-len(".predictions.slp"):], len(p.name)))
    return candidates[0]


def proofread_analysis_h5(session_path: str, camera: str) -> Path | None:
    cam_dir = session_dir(session_path) / camera
    if not cam_dir.exists():
        return None
    matches = sorted(cam_dir.glob(f"{camera}-*.predictions.proofread.slp.analysis.h5"))
    return matches[0] if matches else None


def reprojections_slp_h5(session_path: str, camera: str) -> Path | None:
    cam_dir = session_dir(session_path) / camera
    if not cam_dir.exists():
        return None
    matches = sorted(cam_dir.glob(f"{camera}-*.reprojections.slp.h5"))
    return matches[0] if matches else None


def calibration_toml(session_path: str) -> Path:
    return session_dir(session_path) / "calibration.toml"


def num_animals_from_row(row) -> int:
    return int(row["white_mice"] + row["agouti_mice"] + row["black_mice"])
