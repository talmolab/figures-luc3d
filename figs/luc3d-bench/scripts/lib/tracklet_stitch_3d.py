"""Identity-consistent tracklet stitching using KF-extrapolated 3D positions.

Algorithm (per camera + trackIdx):
  1. Split each timeline into tracklets at frame gaps.
  2. For each adjacent pair (A ends at f_a, B starts at f_b with
     0 < f_b - f_a <= max_gap), compute:
       predicted_pos = A.last_pos3d + A.last_vel3d * (f_b - f_a)
       d2 = ||predicted_pos - B.first_pos3d||^2
       adj_3d = exp(-d2 / (2 * sigma^2)) * exp(-(f_b - f_a) * time_decay)
     If adj_3d > threshold, mark B for relabel to A.identity.
  3. Apply relabels via union-find (keeps earliest-start identity).
"""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np


@dataclass
class Tracklet:
    start_frame: int
    end_frame: int
    identity: int
    last_pos3d: np.ndarray
    last_vel3d: np.ndarray
    first_pos3d: np.ndarray
    cam: str = ""
    slot: int = 0
    members: list = field(default_factory=list)


def stitch_tracklets(tracklets, max_gap=60, sigma=50.0, time_decay=0.05,
                     threshold=0.5):
    """Single-camera-slot stitch. Returns merged list with relabeled identities.

    For each pair (a, b) sorted by start_frame where 0 < b.start - a.end <= max_gap,
    compute KF-extrapolated prediction and time-decayed spatial gate.  If the gate
    exceeds threshold, union the two tracklets (b inherits a's identity).

    Returns a list of merged Tracklet objects — one per union-find component.
    """
    if len(tracklets) <= 1:
        return list(tracklets)

    sorted_t = sorted(tracklets, key=lambda t: t.start_frame)
    n = len(sorted_t)
    two_sig2 = 2.0 * sigma * sigma

    # Union-find with path compression.
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]  # path halving
            i = parent[i]
        return i

    def union(i, j):
        ri, rj = find(i), find(j)
        if ri != rj:
            # Always attach rj to ri so that root keeps the earlier tracklet's identity.
            parent[rj] = ri

    for i in range(n - 1):
        a = sorted_t[i]
        for j in range(i + 1, n):
            b = sorted_t[j]
            dt = b.start_frame - a.end_frame
            if dt <= 0:
                continue
            if dt > max_gap:
                # Sorted by start_frame: if this b is already too far, all
                # subsequent b's start even later (their dt >= this dt), so
                # we can break early.
                break
            pred = a.last_pos3d + a.last_vel3d * dt
            d2 = float(np.sum((pred - b.first_pos3d) ** 2))
            adj = float(np.exp(-d2 / two_sig2) * np.exp(-dt * time_decay))
            if adj > threshold:
                union(i, j)

    # Collect components — each component gets the earliest tracklet's identity.
    merged: dict[int, Tracklet] = {}
    for i, t in enumerate(sorted_t):
        root = find(i)
        if root not in merged:
            # First time we see this component: use the root tracklet's identity
            # and first_pos3d; track end_frame/last_pos3d/last_vel3d as we go.
            root_t = sorted_t[root]
            merged[root] = Tracklet(
                start_frame=root_t.start_frame,
                end_frame=t.end_frame,
                identity=root_t.identity,
                last_pos3d=t.last_pos3d.copy(),
                last_vel3d=t.last_vel3d.copy(),
                first_pos3d=root_t.first_pos3d.copy(),
                cam=t.cam,
                slot=t.slot,
                members=[t],
            )
        else:
            m = merged[root]
            # Extend the merged tracklet to cover this member's range.
            m.end_frame = max(m.end_frame, t.end_frame)
            # Track the last known 3D state (latest end_frame wins).
            if t.end_frame >= m.end_frame:
                m.last_pos3d = t.last_pos3d.copy()
                m.last_vel3d = t.last_vel3d.copy()
            m.members.append(t)

    return list(merged.values())
