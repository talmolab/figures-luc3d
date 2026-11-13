"""Write the detections-only benchmark report: old vs detections-only, per
tracker, plus stratified breakdowns and the headline ranking answer.

Reads outputs/PAF_3d_kalman/metrics/per_camera_session_metrics.csv and compares
each detections-only row against its "old" (input-asymmetric / data-dropping)
counterpart on the SAME (session, camera) pairs, so the comparison is paired.

Usage: python scripts/lib/write_detok_report.py --out <path.md>
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
METRICS = REPO / "outputs" / "PAF_3d_kalman" / "metrics" / "per_camera_session_metrics.csv"

# detections-only tracker -> the OLD (buggy-input) counterpart it replaces.
PAIRS = [
    ("detok_luc3d", "sleap_nn_filtered_no_paf", "LUC3D"),
    ("detok_liezl", "liezl_on_sleap_nn", "Liezl CrossView"),
    ("detok_bytetrack", "bytetrack", "ByteTrack"),
    ("detok_sleap_ref", "sleap", "SLEAP (single-view ref)"),
]

AGG_COLS = ["num_switches", "idf1", "idp", "idr", "num_fragmentations"]


def _agg(df: pd.DataFrame) -> dict:
    if len(df) == 0:
        return {}
    return {
        "n": len(df),
        "switches_total": int(df["num_switches"].sum()),
        "switches_per_sess": float(df["num_switches"].mean()),
        "idf1": float(df["idf1"].mean()),
        "idp": float(df["idp"].mean()),
        "idr": float(df["idr"].mean()),
        "frag_total": int(df["num_fragmentations"].sum()),
    }


def _fmt(a: dict) -> str:
    if not a:
        return "— | — | — | — | — | — | —"
    return (f"{a['n']} | {a['switches_total']:,} | {a['switches_per_sess']:.2f} | "
            f"{a['idf1']:.4f} | {a['idp']:.4f} | {a['idr']:.4f} | {a['frag_total']:,}")


def _paired(df, new_t, old_t):
    """Restrict both trackers to the (session,camera) pairs they share."""
    keys = ["session", "camera"]
    new = df[df.tracker == new_t]
    old = df[df.tracker == old_t]
    if len(new) == 0 or len(old) == 0:
        return new, old
    common = pd.merge(new[keys], old[keys], on=keys)
    nidx = new.set_index(keys).loc[list(map(tuple, common.values))].reset_index()
    oidx = old.set_index(keys).loc[list(map(tuple, common.values))].reset_index()
    return nidx, oidx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(REPO / "outputs" / "PAF_3d_kalman" / "detections_only_report.md"))
    ap.add_argument("--metrics", default=str(METRICS))
    a = ap.parse_args()

    df = pd.read_csv(a.metrics)
    present = set(df.tracker.unique())
    L = []
    L.append("# Detections-Only ID-Tracking Benchmark — Report\n")
    L.append(f"_Generated {pd.Timestamp.now('UTC'):%Y-%m-%d %H:%M} UTC from `{Path(a.metrics).name}`._\n")
    L.append("Every tracker consumes the **identical per-frame detection pool** "
             "(filter-only sleap-nn export, no tracking, no per-frame cap, no "
             "top-N drop). Identities are formed from scratch and scored against "
             "proofread GT with the 4-slot evaluator bottleneck removed. This "
             "fixes handoff §5 Bug 1 (top-4 track drop) and Bug 2 (input asymmetry "
             "+ index mismatch).\n")

    # ---- Headline ranking answer -----------------------------------------
    luc = df[df.tracker == "detok_luc3d"]
    lie = df[df.tracker == "detok_liezl"]
    L.append("## Headline\n")
    if len(luc) and len(lie):
        luc_sw, lie_sw = int(luc.num_switches.sum()), int(lie.num_switches.sum())
        luc_f, lie_f = luc.idf1.mean(), lie.idf1.mean()
        winner_sw = "Liezl" if lie_sw < luc_sw else "LUC3D"
        winner_f1 = "Liezl" if lie_f > luc_f else "LUC3D"
        L.append(
            f"On the identical detection pool, **LUC3D = {luc_sw:,} switches "
            f"(IDF1 {luc_f:.4f})** vs **Liezl = {lie_sw:,} switches "
            f"(IDF1 {lie_f:.4f})** across {len(luc)} cam-sessions. "
            f"Fewest switches: **{winner_sw}**; highest IDF1: **{winner_f1}**. "
            "Because both now see byte-identical inputs and are scored without the "
            "4-slot cap, this ranking reflects tracker quality, not input "
            "asymmetry.\n")
    else:
        L.append("_(detok_luc3d and/or detok_liezl rows not present yet.)_\n")

    # ---- Old vs detections-only, per tracker (paired) --------------------
    L.append("## Old (input-asymmetric) vs Detections-only — paired on shared cam-sessions\n")
    L.append("Columns: n | total switches | switches/sess | IDF1 | IDP | IDR | fragmentations\n")
    for new_t, old_t, label in PAIRS:
        if new_t not in present:
            continue
        L.append(f"\n### {label}\n")
        if old_t in present:
            new_p, old_p = _paired(df, new_t, old_t)
            L.append(f"| variant | n | switches | /sess | IDF1 | IDP | IDR | frag |")
            L.append("|---|--:|--:|--:|--:|--:|--:|--:|")
            L.append(f"| old (`{old_t}`) | {_fmt(_agg(old_p))} |")
            L.append(f"| detok (`{new_t}`) | {_fmt(_agg(new_p))} |")
            na, oa = _agg(new_p), _agg(old_p)
            if na and oa:
                d_sw = na["switches_total"] - oa["switches_total"]
                d_f1 = na["idf1"] - oa["idf1"]
                L.append(f"| Δ (detok − old) | {na['n']} | {d_sw:+,} | "
                         f"{na['switches_per_sess']-oa['switches_per_sess']:+.2f} | "
                         f"{d_f1:+.4f} | {na['idp']-oa['idp']:+.4f} | "
                         f"{na['idr']-oa['idr']:+.4f} | {na['frag_total']-oa['frag_total']:+,} |")
        else:
            L.append(f"| variant | n | switches | /sess | IDF1 | IDP | IDR | frag |")
            L.append("|---|--:|--:|--:|--:|--:|--:|--:|")
            L.append(f"| detok (`{new_t}`) | {_fmt(_agg(df[df.tracker==new_t]))} |")
            L.append(f"\n_(no old counterpart `{old_t}` present for paired Δ.)_")

    # ---- Stratified breakdowns -------------------------------------------
    detok_present = [t for t, _, _ in PAIRS if t in present and t != "detok_sleap_ref"]
    for strat in ["difficulty", "num_animals", "animals"]:
        if strat not in df.columns:
            continue
        L.append(f"\n## Stratified by `{strat}` (detections-only trackers)\n")
        L.append(f"| {strat} | tracker | n | switches | /sess | IDF1 |")
        L.append("|--:|---|--:|--:|--:|--:|")
        for val in sorted(df[strat].dropna().unique()):
            for t in detok_present:
                s = df[(df.tracker == t) & (df[strat] == val)]
                if len(s) == 0:
                    continue
                L.append(f"| {val} | {t} | {len(s)} | {int(s.num_switches.sum()):,} | "
                         f"{s.num_switches.mean():.2f} | {s.idf1.mean():.4f} |")
        break  # difficulty preferred; num_animals/animals only if difficulty absent

    # num_animals breakdown explicitly (always, if present)
    nacol = "num_animals" if "num_animals" in df.columns else ("animals" if "animals" in df.columns else None)
    if nacol:
        L.append(f"\n## Stratified by `{nacol}`\n")
        L.append(f"| {nacol} | tracker | n | switches | /sess | IDF1 |")
        L.append("|--:|---|--:|--:|--:|--:|")
        for val in sorted(df[nacol].dropna().unique()):
            for t in detok_present:
                s = df[(df.tracker == t) & (df[nacol] == val)]
                if len(s) == 0:
                    continue
                L.append(f"| {val} | {t} | {len(s)} | {int(s.num_switches.sum()):,} | "
                         f"{s.num_switches.mean():.2f} | {s.idf1.mean():.4f} |")

    # ---- Coverage / integrity note ---------------------------------------
    L.append("\n## Coverage\n")
    for t in ["detok_luc3d", "detok_liezl", "detok_bytetrack", "detok_sleap_ref"]:
        if t in present:
            s = df[df.tracker == t]
            L.append(f"- `{t}`: {len(s)} cam-sessions, "
                     f"{s.session.nunique()} sessions, {s.camera.nunique()} cameras.")
    L.append("\n_Note: the SLEAP single-view reference (`detok_sleap_ref`) is "
             "evaluated on the ORIGINAL tracked H5 (legitimate stable-slot identity), "
             "not the detections-only H5._\n")

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(L) + "\n")
    print(f"wrote {out} ({len(L)} lines)")


if __name__ == "__main__":
    main()
