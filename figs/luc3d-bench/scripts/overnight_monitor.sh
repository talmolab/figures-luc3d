#!/usr/bin/env bash
# Autonomous overnight monitor.
# 1. Waits for the active pipeline PID (in /tmp/zero_swaps_pid.txt) to exit.
# 2. If retracks are incomplete, rescues with 1800s timeout.
# 3. Re-runs downstream stages if needed.
# 4. Runs iteration controller (figure + Stage 4 sweep until baseline beaten
#    OR sweep exhausted).
#
# Idempotent within reason — runs the orchestrator pipeline up to 3 times if
# earlier rounds left holes.
set -uo pipefail

REPO=$(cd "$(dirname "$0")/.." && pwd)
cd "$REPO"
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"

LOG=logs/overnight_monitor.log
PIPELINE_LOG=logs/zero_swaps_full.log
PIPELINE_PID_FILE=/tmp/zero_swaps_pid.txt
mkdir -p logs

log()  { echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$LOG"; }

log "============================================================"
log "OVERNIGHT MONITOR STARTED"
log "============================================================"

PY=.venv/bin/python
SLEAP_NN=sleap_nn_env/bin/python

# ---------------------------------------------------------------------------
# Phase A: wait for pipeline PID
# ---------------------------------------------------------------------------
if [ -f "$PIPELINE_PID_FILE" ]; then
    PIPELINE_PID=$(cat "$PIPELINE_PID_FILE")
    log "Phase A: waiting on pipeline PID $PIPELINE_PID"
    last_ledger=0
    last_log_ts=0
    while ps -p "$PIPELINE_PID" > /dev/null 2>&1; do
        sleep 60
        now=$(date +%s)
        ledger_now=$(wc -l < outputs/sleap_nn_retracked/_retrack_ledger.jsonl 2>/dev/null || echo 0)
        if [ "$ledger_now" -ne "$last_ledger" ]; then
            log "  ledger: $ledger_now / 444 retracked"
            last_ledger=$ledger_now
            last_log_ts=$now
        elif [ $((now - last_log_ts)) -gt 300 ]; then
            tail_line=$(tail -1 "$PIPELINE_LOG" 2>/dev/null | head -c 200)
            log "  (no ledger change in 5 min) last log line: $tail_line"
            last_log_ts=$now
        fi
    done
    log "Phase A: pipeline PID $PIPELINE_PID exited"
fi

log "Pipeline tail (last 40 lines):"
tail -40 "$PIPELINE_LOG" 2>/dev/null | sed 's/^/    /' | tee -a "$LOG"

# ---------------------------------------------------------------------------
# Phase B: rescue retracks if not all 444 are in the ledger
# ---------------------------------------------------------------------------
ledger_before=$(wc -l < outputs/sleap_nn_retracked/_retrack_ledger.jsonl 2>/dev/null || echo 0)
log "Phase B: rescue retracks (ledger=$ledger_before / 444)"
attempt=0
while [ "$ledger_before" -lt 444 ] && [ $attempt -lt 3 ]; do
    attempt=$((attempt + 1))
    log "  rescue attempt $attempt"
    $SLEAP_NN scripts/sleap_nn/retrack_all.py \
        --master outputs/predictions_master_sheet.tsv \
        --out-dir outputs/sleap_nn_retracked \
        --workers 16 2>&1 | tee -a logs/zero_swaps/retrack_rescue_${attempt}.log
    ledger_after=$(wc -l < outputs/sleap_nn_retracked/_retrack_ledger.jsonl 2>/dev/null || echo 0)
    log "  ledger: $ledger_before → $ledger_after"
    if [ "$ledger_after" -eq "$ledger_before" ]; then
        log "  no progress on attempt $attempt — stopping rescue loop"
        break
    fi
    ledger_before=$ledger_after
done

# ---------------------------------------------------------------------------
# Phase C: ensure all 4 stage rows exist in per_camera_session_metrics.csv.
# If any are missing, re-run the orchestrator.
# ---------------------------------------------------------------------------
log "Phase C: verify stage rows in per_camera_session_metrics.csv"

verify_rows() {
    $PY - <<'PY'
import sys, pandas as pd
from pathlib import Path
p = Path("outputs/PAF_3d_kalman/metrics/per_camera_session_metrics.csv")
if not p.exists():
    print("missing", p); sys.exit(1)
df = pd.read_csv(p)
needed = ["sleap_nn_filtered", "luc3d_paf_l1_dedupe",
          "luc3d_paf_l1_dedupe_tracklet3d",
          "luc3d_paf_l1_dedupe_tracklet3d_stitched"]
missing = [n for n in needed if n not in df.tracker.unique()]
if missing:
    print("missing:", missing); sys.exit(1)
print("all 4 stage rows present")
sys.exit(0)
PY
}

if ! verify_rows; then
    log "  stage rows missing — re-running orchestrator (may take 1-3 hr)"
    nohup bash scripts/run_zero_swaps_pipeline.sh > "$PIPELINE_LOG.v2" 2>&1 &
    P2=$!
    disown
    log "  spawned v2 pipeline PID $P2"
    while ps -p "$P2" > /dev/null 2>&1; do
        sleep 180
        tail_line=$(tail -1 "$PIPELINE_LOG.v2" 2>/dev/null | head -c 180)
        log "  v2 alive; last: $tail_line"
    done
    log "  v2 pipeline exited"
fi

# ---------------------------------------------------------------------------
# Phase D: iteration controller
# ---------------------------------------------------------------------------
log ""
log "Phase D: iteration controller"
$PY scripts/iterate_zero_swaps.py 2>&1 | tee -a logs/zero_swaps/iterate.log

log ""
log "============================================================"
log "OVERNIGHT MONITOR COMPLETE"
log "============================================================"
echo "Final plots in outputs/PAF_3d_kalman/plots/:" | tee -a "$LOG"
ls -la outputs/PAF_3d_kalman/plots/ 2>&1 | grep -E "zero_swaps|figure_main" | tee -a "$LOG"
echo "Final metrics summary:" | tee -a "$LOG"
$PY - <<'PY' 2>&1 | tee -a "$LOG"
import pandas as pd
df = pd.read_csv("outputs/PAF_3d_kalman/metrics/per_camera_session_metrics.csv")
order = ["sleap", "luc3d", "luc3d_paf_l1_tracklet", "liezl_cross_view",
         "sleap_nn_filtered", "luc3d_paf_l1_dedupe",
         "luc3d_paf_l1_dedupe_tracklet3d",
         "luc3d_paf_l1_dedupe_tracklet3d_stitched"]
rows = []
for t in order:
    sub = df[df.tracker == t]
    if len(sub) == 0: continue
    rows.append({"tracker": t, "n": len(sub),
                 "switches": int(sub["num_switches"].sum()),
                 "/sess": round(sub["num_switches"].sum() / len(sub), 2),
                 "IDF1": round(sub["idf1"].mean(), 4),
                 "IDP": round(sub["idp"].mean(), 4)})
print(pd.DataFrame(rows).to_string(index=False))
PY
