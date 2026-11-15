"""Rename the 'luc3d' rows in an eval CSV to a tracker label of our choosing.

Used to mark Track C ReID outputs as `luc3d_paf_l1_tracklet_reid` rather than
plain `luc3d` (which evaluate.py uses regardless of the input dir).
"""
import argparse

import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_csv", required=True)
    ap.add_argument("--out", dest="out_csv", required=True)
    ap.add_argument("--from-tracker", default="luc3d")
    ap.add_argument("--to-tracker", required=True)
    ap.add_argument("--drop-others", action="store_true",
                    help="drop rows where tracker != --from-tracker")
    args = ap.parse_args()

    df = pd.read_csv(args.in_csv)
    if args.drop_others:
        df = df[df["tracker"] == args.from_tracker].copy()
    df.loc[df["tracker"] == args.from_tracker, "tracker"] = args.to_tracker
    df.to_csv(args.out_csv, index=False)
    print(f"wrote {args.out_csv} ({len(df)} rows, trackers: {df['tracker'].unique().tolist()})")
    sub = df[df["tracker"] == args.to_tracker]
    print(f"  {args.to_tracker}: switches total={int(sub['num_switches'].sum())} "
          f"IDF1={sub['idf1'].mean():.4f} MOTA={sub['mota'].mean():.4f}")


if __name__ == "__main__":
    main()
