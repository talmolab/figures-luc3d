#!/bin/bash
# Track A — after the sweep + all 12 evals are done:
#   1. Run track_a_analyze.py to pick the winner + write tracklet_3d_winner.json + track_a_report.md.
#   2. Copy the winning eval CSV to _eval_tracklet_3d_winner.csv for downstream merge.

set -euo pipefail
cd /root/vast/eric/luc3d-bench
source .venv/bin/activate

PROGRESS_LOG="outputs/PAF_3d_kalman/track_a_progress.log"

echo "$(date -Iseconds) ===== FINALIZE START =====" >> "${PROGRESS_LOG}"

python scripts/track_a_analyze.py

# Read the winning config from tracklet_3d_winner.json and copy its eval CSV.
python - <<'PYEOF'
import json
import shutil
from pathlib import Path

REPO = Path("/root/vast/eric/luc3d-bench")
winner = json.loads((REPO / "outputs/PAF_3d_kalman/tracklet_3d_winner.json").read_text())
best = winner["best_by_switches"]
cfg_id = best["cfg_id"]
src = REPO / f"outputs/PAF_3d_kalman/_eval_tracklet_3d_{cfg_id}.csv"
dst = REPO / "outputs/PAF_3d_kalman/_eval_tracklet_3d_winner.csv"
shutil.copy2(src, dst)
print(f"copied winner ({cfg_id}) eval CSV to {dst.name}")
PYEOF

echo "$(date -Iseconds) ===== FINALIZE DONE =====" >> "${PROGRESS_LOG}"
