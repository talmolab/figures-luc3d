#!/usr/bin/env python3
"""Probe whether `sleap-nn cli track` runs filter+retrack on an existing .slp
(with predictions) without --model_paths. If yes, Branch A (CLI shellout) is
viable; if no, retrack_one.py must use Branch B (library adapter).

Reports outcome to scripts/sleap_nn/PROBE_RESULT.txt for downstream code to
read at import time.
"""
from __future__ import annotations
import subprocess
import sys
import time
from pathlib import Path
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
MASTER = REPO / "outputs" / "predictions_master_sheet.tsv"
SLEAP_NN_PY = REPO / "sleap_nn_env" / "bin" / "python"
RESULT_PATH = REPO / "scripts" / "sleap_nn" / "PROBE_RESULT.txt"


def first_easy_raw_slp() -> Path:
    df = pd.read_csv(MASTER, sep="\t")
    row = df[df.difficulty == 1].iloc[0]
    return Path(row["back_raw_pred_slp"])


def main() -> int:
    slp = first_easy_raw_slp()
    if not slp.exists():
        msg = f"PROBE FAILED: raw .slp not found at {slp}"
        print(msg, file=sys.stderr)
        _record("does NOT work", reason=msg)
        return 2

    out = REPO / "outputs" / "_probe_sleap_nn.slp"
    out.parent.mkdir(exist_ok=True)
    cmd = [
        str(SLEAP_NN_PY), "-m", "sleap_nn.cli", "track",
        "--data_path", str(slp),
        "--filter_min_instance_score", "0.85",
        "--filter_min_mean_node_score", "0.80",
        "--filter_min_visible_nodes", "8",
        "--filter_overlapping",
        "--filter_overlapping_method", "oks",
        "--filter_overlapping_threshold", "0.50",
        "--tracking",
        "--tracking_window_size", "15",
        "--tracking_target_instance_count", "2",
        "--post_connect_single_breaks",
        "--output_path", str(out),
    ]
    print("Probe command:", " ".join(cmd))
    t0 = time.time()
    res = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    dt = time.time() - t0
    print(f"return code: {res.returncode}, wall {dt:.1f}s")
    print("STDOUT tail:", res.stdout[-1500:])
    print("STDERR tail:", res.stderr[-1500:])

    if res.returncode == 0 and out.exists() and out.stat().st_size > 0:
        _record("works", reason=f"exit=0, out={out.stat().st_size} bytes, wall={dt:.1f}s")
        print("PROBE RESULT: filter-only mode works")
        return 0

    err_summary = res.stderr.strip().splitlines()[-1] if res.stderr.strip() else "unknown"
    _record("does NOT work",
            reason=f"exit={res.returncode}, last_stderr={err_summary[:200]}")
    print("PROBE RESULT: filter-only mode does NOT work — adapter needed")
    return 1


def _record(verdict: str, reason: str) -> None:
    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    with RESULT_PATH.open("a") as f:
        f.write(f"{stamp} probe_filter_only: {verdict}  ({reason})\n")


if __name__ == "__main__":
    sys.exit(main())
