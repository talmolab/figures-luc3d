#!/bin/bash
# Run all evals + merge + plots for the PAF_3d_kalman benchmark.
# Produces outputs/PAF_3d_kalman/metrics/* and plots/*.

set -euo pipefail
cd /root/vast/eric/luc3d-bench
mkdir -p outputs/PAF_3d_kalman/metrics outputs/PAF_3d_kalman/plots

echo "=== Evaluate L1 outputs (PAF Landing 1) ==="
uv run python scripts/evaluate.py \
    --luc3d-dir outputs/PAF_3d_kalman/luc3d_l1_only_results \
    --out-csv outputs/PAF_3d_kalman/_eval_l1.csv \
    --workers 8

echo "=== Evaluate TIGHT outputs (PAF Landings 1+2+3) ==="
uv run python scripts/evaluate.py \
    --luc3d-dir outputs/PAF_3d_kalman/luc3d_paf_tight_results \
    --out-csv outputs/PAF_3d_kalman/_eval_tight.csv \
    --workers 8

echo "=== Re-evaluate baseline (canonical 3-tracker CSV) ==="
uv run python scripts/evaluate.py \
    --out-csv outputs/PAF_3d_kalman/_eval_baseline.csv \
    --workers 8

echo "=== Merge into 5-tracker CSV ==="
uv run python - <<'PYEOF'
import pandas as pd
from pathlib import Path

REPO = Path("/root/vast/eric/luc3d-bench")
base = pd.read_csv(REPO / "outputs/PAF_3d_kalman/_eval_baseline.csv")
l1   = pd.read_csv(REPO / "outputs/PAF_3d_kalman/_eval_l1.csv")
full = pd.read_csv(REPO / "outputs/PAF_3d_kalman/_eval_tight.csv")

# In each PAF CSV, the rows labeled "luc3d" are the PAF outputs. Rename them.
l1_paf   = l1[l1["tracker"] == "luc3d"].copy();   l1_paf["tracker"] = "luc3d_paf_l1"
full_paf = full[full["tracker"] == "luc3d"].copy(); full_paf["tracker"] = "luc3d_paf_full"

merged = pd.concat([base, l1_paf, full_paf], ignore_index=True)
out = REPO / "outputs/PAF_3d_kalman/metrics/per_camera_session_metrics.csv"
merged.to_csv(out, index=False)
print(f"wrote {out}: {len(merged)} rows, trackers = {sorted(merged['tracker'].unique())}")
PYEOF

echo "=== Generate plots + tables ==="
uv run python scripts/make_plots_paf.py

echo "DONE."
