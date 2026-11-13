"""Fig 8d scoring: how well does LUC3D's cross-view tracker hold identity on
social-DANNCE's rig, given identity-stripped 2D?

Three metrics, deliberately separate:

  instance_acc   fraction of camera-instances assigned to the RIGHT animal, under
                 the single best global identity permutation (tracker identity ->
                 true animal) for the whole session. This is the one that answers
                 "how many incorrect labels".

  frame_acc      fraction of frames whose grouping equals the true partition,
                 label-invariantly. Fig 3d's metric.

  switches       frames where the best per-frame permutation differs from the
                 previous frame's -- i.e. the identities swapped. This is the one
                 that answers "how many switches".

A tracker that is internally consistent but globally flipped scores 0 switches and
high frame_acc but low instance_acc under the wrong permutation, which is why the
permutation is fitted GLOBALLY (best case for the tracker) and reported alongside.
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd


_SEP_CACHE = {}


def separations_for(session: str, root: Path) -> np.ndarray:
    """Per-frame inter-animal 3D separation (mm), straight from s-DANNCE's own
    predictions. Computed here rather than carried through the prepared file so a
    run that predates the field can still be stratified -- it is a property of the
    recording, not of the tracker run."""
    if session in _SEP_CACHE:
        return _SEP_CACHE[session]
    import scipy.io as sio
    f = root / session / "sdannce" / "save_data_AVG0.mat"
    if not f.exists():
        _SEP_CACHE[session] = None
        return None
    pred = np.asarray(sio.loadmat(f)["pred"])            # (T, A, 3, K)
    if pred.shape[1] != 2:
        _SEP_CACHE[session] = None
        return None
    cent = np.nanmean(pred, axis=3)                      # (T, A, 3)
    sep = np.linalg.norm(cent[:, 0] - cent[:, 1], axis=1)
    _SEP_CACHE[session] = sep
    return sep


def tag_from_name(path: Path) -> dict:
    """Recover a sweep config from the filename for runs written before the
    preparer recorded it. track_<session>_projected_<kind><value>.json"""
    stem = path.stem
    if "_projected_" not in stem:
        return {}
    tag = stem.split("_projected_", 1)[1]
    for kind, field in (("noise", "noise_px"), ("drop", "dropout"), ("occ", "occlude_views")):
        if tag.startswith(kind):
            try:
                return {"tag": tag, field: float(tag[len(kind):])}
            except ValueError:
                return {"tag": tag}
    return {"tag": tag}


def score(run: dict, sep_root: Path | None = None) -> dict:
    A = run["n_animals"]
    perms = list(itertools.permutations(range(A)))
    frames = [f for f in run["frames"] if "assignments" in f and f["assignments"]]
    if not frames:
        return {"session": run["session"], "source": run["source"], "frames": 0}

    # Identity IDs are session-global and are NOT guaranteed to be 0..A-1: a
    # fresh-per-frame run mints a new pair every frame (0,1 then 2,3 ...), and
    # even a temporal run can exceed A if it fragments. Map ids to slots instead
    # of assuming their value -- an earlier version filtered on `ident >= A` and
    # silently scored only the first frame.
    fresh = bool(run.get("fresh_per_frame"))
    per_bout = bool(run.get("per_bout"))

    def frame_counts(f):
        """A x A counts for one frame, with identities indexed by first appearance
        within that frame (fresh) or globally (temporal)."""
        truth = f["truth"]
        c = np.zeros((A, A), dtype=np.int64)
        local = {}
        over = 0
        for key, ident in f["assignments"].items():
            cam, idx = key.rsplit(":", 1)
            idx = int(idx)
            t = truth.get(cam, [])
            if idx >= len(t) or ident is None:
                continue
            if fresh:
                slot = local.setdefault(ident, len(local))
            elif per_bout:
                slot = bslot.get(f.get("bout"), {}).get(ident)
                if slot is None:
                    over += 1
                    continue
            else:
                slot = gslot.get(ident)
                if slot is None:
                    over += 1
                    continue
            if slot >= A:
                over += 1
                continue
            c[slot, int(t[idx])] += 1
        return c, over

    # Per-BOUT id -> slot. Identity ids restart with every independent bout run,
    # so a global map is wrong; but mapping per FRAME is worse -- it re-labels the
    # identities every frame by arbitrary first-appearance order, which made the
    # best permutation flip at random and reported a 50% switch rate on runs whose
    # grouping was in fact perfect. The slot must be constant across a bout, which
    # is exactly the span over which the tracker is claiming to hold identity.
    bslot = {}
    if per_bout:
        freq_b = {}
        for f in frames:
            b = f.get("bout")
            for ident in f["assignments"].values():
                if ident is None:
                    continue
                freq_b.setdefault(b, {})
                freq_b[b][ident] = freq_b[b].get(ident, 0) + 1
        for b, freq in freq_b.items():
            bslot[b] = {ident: i for i, ident in
                        enumerate(sorted(freq, key=lambda k: -freq[k])[:A])}

    # Global id -> slot for the temporal case: the A most frequent identity ids.
    gslot = {}
    if not fresh and not per_bout:
        freq = {}
        for f in frames:
            for ident in f["assignments"].values():
                if ident is not None:
                    freq[ident] = freq.get(ident, 0) + 1
        for i, ident in enumerate(sorted(freq, key=lambda k: -freq[k])[:A]):
            gslot[ident] = i

    counts = np.zeros((A, A), dtype=np.int64)
    per_frame_best = []
    n_inst = 0
    n_over = 0
    for f in frames:
        c, over = frame_counts(f)
        n_over += over
        counts += c
        n_inst += int(c.sum())
        best = max(perms, key=lambda p: sum(c[i, p[i]] for i in range(A)))
        hit = sum(c[i, best[i]] for i in range(A))
        per_frame_best.append((best, int(hit), int(c.sum())))

    gbest = max(perms, key=lambda p: sum(counts[i, p[i]] for i in range(A)))
    global_hits = sum(counts[i, gbest[i]] for i in range(A))

    # Across independent per-frame runs the identity LABELS carry no meaning, so
    # a global permutation and a switch count are undefined. Only the grouping is.
    if fresh:
        switches = float("nan")
    elif per_bout:
        # Only transitions INSIDE a bout are switches; the jump between two bouts
        # minutes apart is not one, and counting it would manufacture ~one switch
        # per episode out of nothing.
        bouts = [f.get("bout") for f in frames]
        switches = sum(1 for i in range(1, len(per_frame_best))
                       if bouts[i] == bouts[i - 1]
                       and per_frame_best[i][0] != per_frame_best[i - 1][0])
    else:
        switches = sum(1 for a, b in zip(per_frame_best, per_frame_best[1:])
                       if a[0] != b[0])
    frame_exact = sum(1 for b in per_frame_best if b[1] == b[2] and b[2] > 0)

    # Stratify by inter-animal separation: the pooled number is dominated by frames
    # where the rats are far apart and association is trivial.
    #
    # These are GROUPING accuracies (per-frame best permutation), not global
    # identity: a session that suffers one un-recovered switch still groups every
    # frame correctly and scores 1.0 here while its instance_acc is far below 1.
    # The two answer different questions and must not be quoted for each other.
    strat = {}
    seps = np.array([f.get("sep") if f.get("sep") is not None else np.nan
                     for f in frames], dtype=float)
    if not np.isfinite(seps).any() and sep_root is not None:
        table = separations_for(run["session"], sep_root)
        if table is not None:
            # Index by the SAMPLE ID when the run carries one: for the sparse
            # human-label sets the frame ordinal is 0..n while the real frames
            # are sampleIDs 500, 1500, ... -- using the ordinal would stratify
            # against a completely different part of the recording.
            def _row(f):
                k = f.get("idx")
                if k is None:
                    k = f["frame"]
                return table[k] if 0 <= k < len(table) else np.nan
            seps = np.array([_row(f) for f in frames], dtype=float)
    hits = np.array([b[1] for b in per_frame_best], dtype=float)
    tots = np.array([b[2] for b in per_frame_best], dtype=float)
    if np.isfinite(seps).any():
        for lo, hi, label in [(0, 50, "lt50mm"), (50, 100, "50_100mm"),
                              (100, 200, "100_200mm"), (200, 1e9, "gt200mm")]:
            m = np.isfinite(seps) & (seps >= lo) & (seps < hi)
            strat[f"group_acc_{label}"] = float(hits[m].sum() / tots[m].sum()) if tots[m].sum() else float("nan")
            strat[f"n_{label}"] = int(m.sum())

    return {
        "session": run["session"], "source": run["source"],
        "tag": run.get("tag", ""), "fresh_per_frame": bool(run.get("fresh_per_frame")),
        "noise_px": run.get("noise_px", 0),
        "dropout": run.get("dropout", 0), "occlude_views": run.get("occlude_views", 0),
        "frames": len(frames), "instances": n_inst,
        "n_bouts": (len({f.get("bout") for f in frames if f.get("bout") is not None})
                    if per_bout else 0),
        "instance_acc": (float("nan") if (fresh or per_bout) else
                         (float(global_hits / n_inst) if n_inst else float("nan"))),
        "unassigned_instances": int(n_over),
        "frame_acc": float(frame_exact / len(frames)),
        "switches": switches,
        "switch_rate_per_1k_frames": (float("nan") if fresh else
                                      float(1000 * switches / len(frames))),
        "ms": run.get("ms"),
        **strat,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", type=Path, nargs="+")
    ap.add_argument("--out", type=Path)
    ap.add_argument("--sep-root", type=Path,
                    default=Path("/root/vast/eric/luc3d-bench/sdannce_data"),
                    help="root of the downloaded BEDDING sessions, for separation")
    args = ap.parse_args()
    rows = []
    for p in args.runs:
        run = json.loads(p.read_text())
        for k, v in tag_from_name(p).items():
            run.setdefault(k, v) if not run.get(k) else None
            if not run.get(k):
                run[k] = v
        rows.append(score(run, args.sep_root))
    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.out, index=False)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
