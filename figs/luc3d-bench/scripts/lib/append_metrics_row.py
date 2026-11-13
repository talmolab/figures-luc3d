#!/usr/bin/env python3
"""Append (or replace) a tracker row in per_camera_session_metrics.csv.

Usage:  python scripts/lib/append_metrics_row.py <tracker_name> <eval_csv> [--from-tracker NAME]

evaluate.py emits up to 3 rows per (session, camera): "sleap", "luc3d",
"bytetrack". By default we keep the "luc3d" row (the tracker-under-test, since
evaluate.py reads its JSON for identities) and relabel it <tracker_name>. Pass
--from-tracker bytetrack to register the ByteTrack row instead.

Idempotent — removes any existing rows with the same tracker name before append.
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
MASTER_METRICS = REPO / "outputs" / "PAF_3d_kalman" / "metrics" / "per_camera_session_metrics.csv"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("tracker")
    ap.add_argument("eval_csv")
    ap.add_argument("--from-tracker", default="luc3d",
                    help="which eval-row name to keep before relabeling (default luc3d)")
    args = ap.parse_args()
    tracker = args.tracker
    src = Path(args.eval_csv)
    if not src.exists():
        print(f"ERROR: source CSV missing: {src}")
        return 1

    new = pd.read_csv(src)
    before = len(new)
    new = new[new["tracker"] == args.from_tracker].copy()
    print(f"  filtered eval CSV: {before} rows → {len(new)} (kept tracker=={args.from_tracker})")
    new["tracker"] = tracker

    if MASTER_METRICS.exists():
        existing = pd.read_csv(MASTER_METRICS)
        existing = existing[existing.tracker != tracker]
        combined = pd.concat([existing, new], ignore_index=True)
    else:
        MASTER_METRICS.parent.mkdir(parents=True, exist_ok=True)
        combined = new

    combined.to_csv(MASTER_METRICS, index=False)
    sub = combined[combined.tracker == tracker]
    print(f"appended {tracker}: n={len(sub)}  switches={int(sub['num_switches'].sum())}  "
          f"IDF1={sub['idf1'].mean():.4f}  IDP={sub['idp'].mean():.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
