#!/usr/bin/env bash
# Fast-path: assumes outputs/sleap_nn_retracked/ and
# outputs/sleap_nn_predictions_h5s/ are already populated. Re-runs LUC3D
# + post-process + eval for each of the 4 pipeline stages, then renders
# the comparison figure.
#
# Skips: T9 retracks (already in ledger), T9.2-7 aggregation (already done),
#         T10.1 retrack (already done).
# Runs: T10.4-7 (LUC3D + tracklet + eval + register sleap_nn_filtered),
#        T17.3-5 (Stage 2 winner on full), T19 (tracklet_vote_3d),
#        T23.3-5 (Stage 4 winner on full), then iterate_zero_swaps.py.
set -uo pipefail

REPO=$(cd "$(dirname "$0")/.." && pwd)
cd "$REPO"
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"

PY=.venv/bin/python
SLEAP_NN=sleap_nn_env/bin/python
WORKERS=${WORKERS:-16}

FULL_MASTER_NEW=outputs/sleap_nn_master_sheet.tsv
PRED_H5_NEW=outputs/sleap_nn_predictions_h5s
PARAMS_BASE=outputs/PAF_3d_kalman/paf_l1_params.json
PARAMS_DEDUPE_WINNER=outputs/PAF_3d_kalman/paf_l1_dedupe_winner.json

mkdir -p logs/zero_swaps
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
LOG="logs/zero_swaps/downstream_${STAMP}.log"

log()  { echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$LOG"; }
section(){ log ""; log "============================================================"; log "$*"; log "============================================================"; }

# Ensure full master sheet exists (rebuild from current retracks just in case):
$SLEAP_NN scripts/sleap_nn/build_master_sheet.py \
    --src outputs/predictions_master_sheet.tsv \
    --retrack-dir outputs/sleap_nn_retracked \
    --dst "$FULL_MASTER_NEW" 2>&1 | tee -a "$LOG"

# ---------- Stage 1 (T10.4-7) ----------
section "Stage 1: LUC3D PAF L1 + tracklet on retracked H5s (no dedup)"
rm -rf outputs/PAF_3d_kalman/sleap_nn_filtered_results outputs/PAF_3d_kalman/sleap_nn_filtered_tracklet_results
$PY scripts/run_luc3d_bench.py \
    --master "$FULL_MASTER_NEW" --pred-h5-dir "$PRED_H5_NEW" \
    --params "$PARAMS_BASE" \
    --out-dir outputs/PAF_3d_kalman/sleap_nn_filtered_results \
    --workers $WORKERS 2>&1 | tail -10 | tee -a "$LOG"
$PY scripts/stitch_postprocess.py \
    --in outputs/PAF_3d_kalman/sleap_nn_filtered_results \
    --out outputs/PAF_3d_kalman/sleap_nn_filtered_tracklet_results \
    --mode tracklet_vote --tracklet-gap 30 2>&1 | tail -5 | tee -a "$LOG"
$PY scripts/evaluate.py \
    --luc3d-dir outputs/PAF_3d_kalman/sleap_nn_filtered_tracklet_results \
    --pred-h5-dir "$PRED_H5_NEW" \
    --out-csv outputs/PAF_3d_kalman/_eval_sleap_nn_filtered.csv \
    --workers $WORKERS 2>&1 | tail -5 | tee -a "$LOG"
$PY scripts/lib/append_metrics_row.py sleap_nn_filtered outputs/PAF_3d_kalman/_eval_sleap_nn_filtered.csv | tee -a "$LOG"

# ---------- Stage 2 (T17.3-5, skip sweep — use prior winner or defaults) ----------
section "Stage 2: LUC3D PAF L1 + dedupe + KF gate (full bench)"
# Use defaults if winner not yet picked
if [ ! -f "$PARAMS_DEDUPE_WINNER" ]; then
    log "no winner config — using defaults from paf_l1_dedupe_params.json"
    cp outputs/PAF_3d_kalman/paf_l1_dedupe_params.json "$PARAMS_DEDUPE_WINNER"
fi
rm -rf outputs/PAF_3d_kalman/luc3d_paf_l1_dedupe_results outputs/PAF_3d_kalman/luc3d_paf_l1_dedupe_tracklet_results
$PY scripts/run_luc3d_bench.py \
    --master "$FULL_MASTER_NEW" --pred-h5-dir "$PRED_H5_NEW" \
    --params "$PARAMS_DEDUPE_WINNER" \
    --out-dir outputs/PAF_3d_kalman/luc3d_paf_l1_dedupe_results \
    --workers $WORKERS 2>&1 | tail -10 | tee -a "$LOG"
$PY scripts/stitch_postprocess.py \
    --in outputs/PAF_3d_kalman/luc3d_paf_l1_dedupe_results \
    --out outputs/PAF_3d_kalman/luc3d_paf_l1_dedupe_tracklet_results \
    --mode tracklet_vote --tracklet-gap 30 2>&1 | tail -5 | tee -a "$LOG"
$PY scripts/evaluate.py \
    --luc3d-dir outputs/PAF_3d_kalman/luc3d_paf_l1_dedupe_tracklet_results \
    --pred-h5-dir "$PRED_H5_NEW" \
    --out-csv outputs/PAF_3d_kalman/_eval_luc3d_paf_l1_dedupe.csv \
    --workers $WORKERS 2>&1 | tail -5 | tee -a "$LOG"
$PY scripts/lib/append_metrics_row.py luc3d_paf_l1_dedupe outputs/PAF_3d_kalman/_eval_luc3d_paf_l1_dedupe.csv | tee -a "$LOG"

# ---------- Stage 3 (T19) ----------
section "Stage 3: tracklet_vote_3d on Stage 2 outputs"
WG=0.5
WV=0.0
$PY scripts/stitch_postprocess.py \
    --in outputs/PAF_3d_kalman/luc3d_paf_l1_dedupe_results \
    --out outputs/PAF_3d_kalman/luc3d_paf_l1_dedupe_tracklet3d_results \
    --mode tracklet_vote_3d --tracklet-gap 30 \
    --w-geom $WG --w-velo $WV \
    --master "$FULL_MASTER_NEW" --pred-h5-dir "$PRED_H5_NEW" 2>&1 | tail -5 | tee -a "$LOG"
$PY scripts/evaluate.py \
    --luc3d-dir outputs/PAF_3d_kalman/luc3d_paf_l1_dedupe_tracklet3d_results \
    --pred-h5-dir "$PRED_H5_NEW" \
    --out-csv outputs/PAF_3d_kalman/_eval_luc3d_paf_l1_dedupe_tracklet3d.csv \
    --workers $WORKERS 2>&1 | tail -5 | tee -a "$LOG"
$PY scripts/lib/append_metrics_row.py luc3d_paf_l1_dedupe_tracklet3d outputs/PAF_3d_kalman/_eval_luc3d_paf_l1_dedupe_tracklet3d.csv | tee -a "$LOG"

# ---------- Stage 4 (T23.3-5, use defaults — sweep happens in iterate_zero_swaps) ----------
section "Stage 4: tracklet_stitch_3d (default params; iteration controller will sweep)"
$PY scripts/stitch_postprocess.py \
    --in outputs/PAF_3d_kalman/luc3d_paf_l1_dedupe_tracklet3d_results \
    --out outputs/PAF_3d_kalman/luc3d_paf_l1_dedupe_tracklet3d_stitched_results \
    --mode tracklet_stitch_3d \
    --stitch-max-gap 60 --stitch-sigma 50 --stitch-threshold 0.5 \
    --master "$FULL_MASTER_NEW" --pred-h5-dir "$PRED_H5_NEW" 2>&1 | tail -5 | tee -a "$LOG"
$PY scripts/evaluate.py \
    --luc3d-dir outputs/PAF_3d_kalman/luc3d_paf_l1_dedupe_tracklet3d_stitched_results \
    --pred-h5-dir "$PRED_H5_NEW" \
    --out-csv outputs/PAF_3d_kalman/_eval_luc3d_paf_l1_dedupe_tracklet3d_stitched.csv \
    --workers $WORKERS 2>&1 | tail -5 | tee -a "$LOG"
$PY scripts/lib/append_metrics_row.py luc3d_paf_l1_dedupe_tracklet3d_stitched outputs/PAF_3d_kalman/_eval_luc3d_paf_l1_dedupe_tracklet3d_stitched.csv | tee -a "$LOG"

# ---------- Iteration controller ----------
section "Iteration: figure + Stage 4 sweep until baseline beaten or grid exhausted"
$PY scripts/iterate_zero_swaps.py 2>&1 | tee -a "$LOG"

section "DOWNSTREAM RUN COMPLETE"
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
ls -la outputs/PAF_3d_kalman/plots/ | grep zero_swaps | tee -a "$LOG"
