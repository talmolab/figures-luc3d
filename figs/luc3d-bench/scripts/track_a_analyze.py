"""Aggregate Track A sweep results: pick the best config and write a report.

Reads all 12 _eval_tracklet_3d_<cfg_id>.csv files, computes per-config
totals (total switches, mean IDF1), picks the winner, writes
tracklet_3d_winner.json + track_a_report.md.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
OUT_DIR = REPO / "outputs" / "PAF_3d_kalman"

W_GEOMS = [0.5, 1.0, 2.0, 5.0]
W_VELOS = [0.0, 0.5, 1.0]

# Reference numbers from the plan (vanilla tracklet vote = (0,0) effectively).
REF_TRACKLET_SWITCHES = 1777
REF_TRACKLET_IDF1 = 0.7284
REF_TRACKLET_IDP = 0.9575
REF_NOPP_SWITCHES = 3918
REF_NOPP_IDF1 = 0.7381
REF_NOPP_IDP = 0.9429


def summarize_csv(csv_path: Path) -> dict | None:
    if not csv_path.exists():
        return None
    df = pd.read_csv(csv_path)
    # The post-processed JSONs are scored as "luc3d" tracker rows.
    sub = df[df["tracker"] == "luc3d"]
    if sub.empty:
        return None
    total_switches = int(sub["num_switches"].sum())
    mean_idf1 = float(sub["idf1"].mean())
    mean_idp = float(sub["idp"].mean())
    mean_idr = float(sub["idr"].mean())
    return {
        "total_switches": total_switches,
        "mean_idf1": mean_idf1,
        "mean_idp": mean_idp,
        "mean_idr": mean_idr,
        "n_rows": int(len(sub)),
    }


def main():
    rows = []
    for wg in W_GEOMS:
        for wv in W_VELOS:
            cfg_id = f"wg{wg}_wv{wv}"
            csv = OUT_DIR / f"_eval_tracklet_3d_{cfg_id}.csv"
            s = summarize_csv(csv)
            if s is None:
                print(f"  missing {csv.name}")
                continue
            rows.append({"w_geom": wg, "w_velo": wv, "cfg_id": cfg_id, **s})

    if not rows:
        print("no sweep results found")
        return

    df = pd.DataFrame(rows).sort_values("total_switches", ascending=True).reset_index(drop=True)
    summary_csv = OUT_DIR / "track_a_sweep_summary.csv"
    df.to_csv(summary_csv, index=False)
    print(f"wrote {summary_csv}")
    print(df.to_string(index=False))

    # Pick the winner by lowest switches; tiebreak on highest IDF1.
    df_sorted = df.sort_values(["total_switches", "mean_idf1"],
                               ascending=[True, False]).reset_index(drop=True)
    best = df_sorted.iloc[0].to_dict()

    # Also pick best IDF1.
    df_idf1 = df.sort_values(["mean_idf1", "total_switches"],
                              ascending=[False, True]).reset_index(drop=True)
    best_idf1 = df_idf1.iloc[0].to_dict()

    winner = {
        "best_by_switches": best,
        "best_by_idf1": best_idf1,
        "reference_baselines": {
            "luc3d_paf_l1_tracklet_vanilla": {
                "total_switches": REF_TRACKLET_SWITCHES,
                "mean_idf1": REF_TRACKLET_IDF1,
                "mean_idp": REF_TRACKLET_IDP,
            },
            "luc3d_paf_l1_no_postproc": {
                "total_switches": REF_NOPP_SWITCHES,
                "mean_idf1": REF_NOPP_IDF1,
                "mean_idp": REF_NOPP_IDP,
            },
        },
    }
    winner_path = OUT_DIR / "tracklet_3d_winner.json"
    winner_path.write_text(json.dumps(winner, indent=2))
    print(f"wrote {winner_path}")

    # Markdown report.
    lines = []
    lines.append("# Track A — 3D-anchored tracklet vote: sweep results")
    lines.append("")
    lines.append("**Date:** 2026-05-19")
    lines.append("**Working directory:** `/root/vast/eric/luc3d-bench`")
    lines.append("")
    lines.append("## Reference baselines (PAF L1)")
    lines.append("")
    lines.append("| Config | Total switches | IDF1 | IDP |")
    lines.append("|---|---:|---:|---:|")
    lines.append(f"| LUC3D + PAF L1 (no post-processing) | {REF_NOPP_SWITCHES} | {REF_NOPP_IDF1:.4f} | {REF_NOPP_IDP:.4f} |")
    lines.append(f"| LUC3D + PAF L1 + vanilla tracklet vote (gap=30) | {REF_TRACKLET_SWITCHES} | {REF_TRACKLET_IDF1:.4f} | {REF_TRACKLET_IDP:.4f} |")
    lines.append("")
    lines.append("## Sweep results — `tracklet_vote_3d` on PAF L1 (tracklet_gap=30, σ_geom=50, σ_velo=20)")
    lines.append("")
    lines.append("| w_geom | w_velo | Total switches | IDF1 | IDP | IDR |")
    lines.append("|---:|---:|---:|---:|---:|---:|")
    for _, r in df.iterrows():
        lines.append(
            f"| {r['w_geom']} | {r['w_velo']} | {int(r['total_switches'])} | "
            f"{r['mean_idf1']:.4f} | {r['mean_idp']:.4f} | {r['mean_idr']:.4f} |"
        )
    lines.append("")
    lines.append("## Winners")
    lines.append("")
    lines.append(f"**Best by total switches:** `w_geom={best['w_geom']}, w_velo={best['w_velo']}` "
                 f"→ switches={int(best['total_switches'])}, IDF1={best['mean_idf1']:.4f}, "
                 f"IDP={best['mean_idp']:.4f}")
    lines.append("")
    lines.append(f"**Best by IDF1:** `w_geom={best_idf1['w_geom']}, w_velo={best_idf1['w_velo']}` "
                 f"→ switches={int(best_idf1['total_switches'])}, IDF1={best_idf1['mean_idf1']:.4f}, "
                 f"IDP={best_idf1['mean_idp']:.4f}")
    lines.append("")
    sw_delta = int(best["total_switches"]) - REF_TRACKLET_SWITCHES
    idf1_delta = best["mean_idf1"] - REF_TRACKLET_IDF1
    lines.append("## Verdict")
    lines.append("")
    if sw_delta < 0:
        lines.append(f"3D anchoring **reduced switches by {-sw_delta}** vs. vanilla tracklet vote "
                     f"({int(best['total_switches'])} < {REF_TRACKLET_SWITCHES}).")
    elif sw_delta == 0:
        lines.append("3D anchoring matched the vanilla tracklet vote on total switches.")
    else:
        lines.append(f"3D anchoring did **NOT** reduce switches below the {REF_TRACKLET_SWITCHES} "
                     f"vanilla baseline. The best 3D-anchored config had {int(best['total_switches'])} "
                     f"switches ({sw_delta:+d} vs vanilla).")
    lines.append("")
    if abs(idf1_delta) >= 1e-4:
        sign = "+" if idf1_delta > 0 else ""
        lines.append(f"IDF1 changed by {sign}{idf1_delta:.4f} vs vanilla tracklet vote "
                     f"({best['mean_idf1']:.4f} vs {REF_TRACKLET_IDF1:.4f}).")
    else:
        lines.append("IDF1 was essentially unchanged vs vanilla tracklet vote.")
    lines.append("")
    lines.append("## Outputs")
    lines.append("")
    lines.append(f"- Per-config result JSONs: `outputs/PAF_3d_kalman/luc3d_l1_tracklet_3d_results/wg<wg>_wv<wv>/*.json`")
    lines.append(f"- Per-config eval CSVs:    `outputs/PAF_3d_kalman/_eval_tracklet_3d_wg<wg>_wv<wv>.csv`")
    lines.append(f"- Sweep summary CSV:       `outputs/PAF_3d_kalman/track_a_sweep_summary.csv`")
    lines.append(f"- Winner JSON:             `outputs/PAF_3d_kalman/tracklet_3d_winner.json`")
    lines.append("")

    report_path = OUT_DIR / "track_a_report.md"
    report_path.write_text("\n".join(lines))
    print(f"wrote {report_path}")


if __name__ == "__main__":
    main()
