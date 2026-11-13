"""Compare LP3D-detection results to SLEAP-detection results on the same sessions.

Reads:
  outputs/metrics/per_camera_session_metrics.csv      (Treatment A = SLEAP detections)
  outputs/metrics/lp3d_swap/per_camera_session_metrics.csv  (Treatment B = LP3D detections)

Produces a comparison table for the 10 sessions used in the LP3D swap experiment.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sleap-csv", type=Path, default=REPO / "outputs" / "metrics" / "per_camera_session_metrics.csv")
    ap.add_argument("--lp3d-csv", type=Path, default=REPO / "outputs" / "metrics" / "lp3d_swap" / "per_camera_session_metrics.csv")
    ap.add_argument("--out", type=Path, default=REPO / "outputs" / "metrics" / "lp3d_swap" / "comparison.csv")
    args = ap.parse_args()

    sleap = pd.read_csv(args.sleap_csv)
    lp3d = pd.read_csv(args.lp3d_csv)

    # Keep only LUC3D tracker rows
    sleap = sleap[sleap["tracker"] == "luc3d"].copy()
    lp3d = lp3d[lp3d["tracker"] == "luc3d"].copy()

    # Restrict to sessions in LP3D
    sessions = sorted(lp3d["session"].astype(str).unique())
    sleap = sleap[sleap["session"].astype(str).isin(sessions)].copy()

    # Per-session aggregate
    def agg(df, label):
        out = df.groupby("session").agg(
            **{
                f"idf1_{label}": ("idf1", "mean"),
                f"mota_{label}": ("mota", "mean"),
                f"sw_{label}": ("num_switches", "sum"),
                f"miss_{label}": ("num_misses", "sum"),
                f"fp_{label}": ("num_false_positives", "sum"),
            }
        ).reset_index()
        return out

    a_sl = agg(sleap, "sleap")
    a_lp = agg(lp3d, "lp3d")
    diff = pd.merge(a_sl, a_lp, on="session", how="outer")
    diff["d_idf1"] = diff["idf1_lp3d"] - diff["idf1_sleap"]
    diff["d_mota"] = diff["mota_lp3d"] - diff["mota_sleap"]
    diff["d_switches"] = diff["sw_lp3d"] - diff["sw_sleap"]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    diff.to_csv(args.out, index=False)
    print(f"wrote {args.out}")
    print("\nPer-session comparison (LUC3D tracker, mean over 6 cams):")
    cols = ["session", "idf1_sleap", "idf1_lp3d", "d_idf1",
            "sw_sleap", "sw_lp3d", "d_switches"]
    if all(c in diff.columns for c in cols):
        print(diff[cols].round(3).to_string(index=False))

    # Overall summary
    summary = {
        "metric": ["idf1_mean", "mota_mean", "switches_total"],
        "sleap": [a_sl["idf1_sleap"].mean(), a_sl["mota_sleap"].mean(), a_sl["sw_sleap"].sum()],
        "lp3d": [a_lp["idf1_lp3d"].mean(), a_lp["mota_lp3d"].mean(), a_lp["sw_lp3d"].sum()],
    }
    summary["delta"] = [b - a for a, b in zip(summary["sleap"], summary["lp3d"])]
    print("\nOverall summary:")
    print(pd.DataFrame(summary).to_string(index=False))


if __name__ == "__main__":
    main()
