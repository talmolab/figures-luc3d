#!/usr/bin/env bash
# Zero-swaps strategy pipeline orchestrator (T9 → T10 → T17 → T19 → T23 → reporting).
#
# Stages:
#   T9   sleap-nn retrack on 10-session debug slice + LUC3D PAF L1 + tracklet
#   T10  full-bench retrack + LUC3D + tracklet_vote → sleap_nn_filtered row
#   T17  Stage 2 sweep on debug, then full bench → luc3d_paf_l1_dedupe row
#   T19  tracklet_vote_3d on Stage 2 outputs → luc3d_paf_l1_dedupe_tracklet3d row
#   T23  Stage 4 sweep on debug, then full bench → luc3d_paf_l1_dedupe_tracklet3d_stitched
#   T24-T26  merge_all_variants, regenerate figure_main, write zero_swaps_report.md
#
# Resumable: each stage skips work whose outputs already exist (uses ledgers).
# Wall-clock estimate (this 64-core box, no GPU): ~4-6 hours total.
#
# Launch: nohup bash scripts/run_zero_swaps_pipeline.sh > logs/zero_swaps_full.log 2>&1 &
# Monitor: tail -f logs/zero_swaps_full.log
set -euo pipefail

REPO=$(cd "$(dirname "$0")/.." && pwd)
cd "$REPO"
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"

mkdir -p logs/zero_swaps outputs/PAF_3d_kalman
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
PIPELINE_LOG="logs/zero_swaps/pipeline_${STAMP}.log"

log()    { echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$PIPELINE_LOG"; }
fatal()  { log "FATAL: $*"; exit 1; }
section(){ log ""; log "============================================================"; log "$*"; log "============================================================"; }

PY=.venv/bin/python
SLEAP_NN=sleap_nn_env/bin/python

[ -x "$PY" ]       || fatal "missing main venv at .venv/"
[ -x "$SLEAP_NN" ] || fatal "missing sleap_nn_env (run scripts/sleap_nn/setup_env.sh)"

WORKERS=${WORKERS:-16}
DEBUG_MASTER=outputs/splits/debug_master_sheet.tsv
FULL_MASTER_NEW=outputs/sleap_nn_master_sheet.tsv
PRED_H5_NEW=outputs/sleap_nn_predictions_h5s
PARAMS_BASE=outputs/PAF_3d_kalman/paf_l1_params.json
PARAMS_DEDUPE=outputs/PAF_3d_kalman/paf_l1_dedupe_params.json
PARAMS_DEDUPE_WINNER=outputs/PAF_3d_kalman/paf_l1_dedupe_winner.json

[ -f "$DEBUG_MASTER" ] || fatal "missing $DEBUG_MASTER (run scripts/sleap_nn/build_debug_split.py or rebuild manually)"
[ -f "$PARAMS_BASE" ]  || fatal "missing $PARAMS_BASE"

# ---------------------------------------------------------------------------
# Stage 1: retrack + LUC3D + tracklet_vote on debug slice (T9)
# ---------------------------------------------------------------------------
section "T9: Stage 1 on 10-session debug slice"

log "T9.1: retrack debug 10 sessions × 6 cams (ledger-resumable)"
$SLEAP_NN scripts/sleap_nn/retrack_all.py \
    --master "$DEBUG_MASTER" \
    --out-dir outputs/sleap_nn_retracked \
    --workers $WORKERS 2>&1 | tee -a logs/zero_swaps/retrack_debug.log

log "T9.2: aggregate debug H5s (subset of 10 sessions)"
$SLEAP_NN scripts/sleap_nn/aggregate_to_h5.py \
    --master "$DEBUG_MASTER" \
    --retrack-dir outputs/sleap_nn_retracked \
    --out-dir outputs/sleap_nn_predictions_h5s_debug 2>&1 | tee -a logs/zero_swaps/aggregate_debug.log

log "T9.3: build debug master sheet pointed at retracked slps"
$SLEAP_NN scripts/sleap_nn/build_master_sheet.py \
    --src "$DEBUG_MASTER" \
    --retrack-dir outputs/sleap_nn_retracked \
    --dst outputs/sleap_nn_master_sheet_debug.tsv 2>&1 | tee -a logs/zero_swaps/build_msheet_debug.log

log "T9.4: LUC3D PAF L1 on debug detections (no dedup yet — Stage 1 alone)"
$PY scripts/run_luc3d_bench.py \
    --master outputs/sleap_nn_master_sheet_debug.tsv \
    --pred-h5-dir outputs/sleap_nn_predictions_h5s_debug \
    --params "$PARAMS_BASE" \
    --out-dir outputs/PAF_3d_kalman/sleap_nn_filtered_debug_results \
    --workers $WORKERS 2>&1 | tee -a logs/zero_swaps/luc3d_debug.log

log "T9.5: tracklet_vote post-process on debug"
$PY scripts/stitch_postprocess.py \
    --in outputs/PAF_3d_kalman/sleap_nn_filtered_debug_results \
    --out outputs/PAF_3d_kalman/sleap_nn_filtered_debug_tracklet_results \
    --mode tracklet_vote --tracklet-gap 30 2>&1 | tee -a logs/zero_swaps/stitch_debug.log

log "T9.6: evaluate Stage 1 debug results"
$PY scripts/evaluate.py \
    --luc3d-dir outputs/PAF_3d_kalman/sleap_nn_filtered_debug_tracklet_results \
    --pred-h5-dir outputs/sleap_nn_predictions_h5s_debug \
    --out-csv outputs/PAF_3d_kalman/_eval_sleap_nn_filtered_debug.csv \
    --workers $WORKERS 2>&1 | tee -a logs/zero_swaps/eval_debug.log

log "T9.7: pass-bar check vs baseline-on-debug"
$PY - <<PY 2>&1 | tee -a "$PIPELINE_LOG"
import pandas as pd
new = pd.read_csv("outputs/PAF_3d_kalman/_eval_sleap_nn_filtered_debug.csv")
full = pd.read_csv("outputs/PAF_3d_kalman/metrics/per_camera_session_metrics.csv")
debug = open("outputs/splits/debug_10_sessions.txt").read().strip().splitlines()
baseline = full[(full.tracker == "luc3d_paf_l1_tracklet") & full.session.astype(str).isin(debug)]
print(f"Stage 1 (debug):  switches={new['num_switches'].sum()}  IDF1={new['idf1'].mean():.4f}")
print(f"Baseline (debug): switches={baseline['num_switches'].sum()}  IDF1={baseline['idf1'].mean():.4f}")
ok = new['num_switches'].sum() <= baseline['num_switches'].sum() and abs(new['idf1'].mean() - baseline['idf1'].mean()) < 0.020
print("PASS-BAR:", "OK" if ok else "FAIL (continuing anyway — review metrics)")
PY

# ---------------------------------------------------------------------------
# Stage 1 full bench (T10)
# ---------------------------------------------------------------------------
section "T10: Stage 1 full 444 cam-session bench"

log "T10.1: retrack remaining sessions (ledger resumes from T9.1)"
$SLEAP_NN scripts/sleap_nn/retrack_all.py \
    --master outputs/predictions_master_sheet.tsv \
    --out-dir outputs/sleap_nn_retracked \
    --workers $WORKERS 2>&1 | tee -a logs/zero_swaps/retrack_full.log

log "T10.2: aggregate all 74 sessions to full H5s"
$SLEAP_NN scripts/sleap_nn/aggregate_to_h5.py \
    --master outputs/predictions_master_sheet.tsv \
    --retrack-dir outputs/sleap_nn_retracked \
    --out-dir "$PRED_H5_NEW" 2>&1 | tee -a logs/zero_swaps/aggregate_full.log

log "T10.3: build full new master sheet"
$SLEAP_NN scripts/sleap_nn/build_master_sheet.py \
    --src outputs/predictions_master_sheet.tsv \
    --retrack-dir outputs/sleap_nn_retracked \
    --dst "$FULL_MASTER_NEW" 2>&1 | tee -a logs/zero_swaps/build_msheet_full.log

log "T10.4: LUC3D PAF L1 on full retracked bench"
$PY scripts/run_luc3d_bench.py \
    --master "$FULL_MASTER_NEW" \
    --pred-h5-dir "$PRED_H5_NEW" \
    --params "$PARAMS_BASE" \
    --out-dir outputs/PAF_3d_kalman/sleap_nn_filtered_results \
    --workers $WORKERS 2>&1 | tee -a logs/zero_swaps/luc3d_full.log

log "T10.5: tracklet_vote on full"
$PY scripts/stitch_postprocess.py \
    --in outputs/PAF_3d_kalman/sleap_nn_filtered_results \
    --out outputs/PAF_3d_kalman/sleap_nn_filtered_tracklet_results \
    --mode tracklet_vote --tracklet-gap 30 2>&1 | tee -a logs/zero_swaps/stitch_full.log

log "T10.6: evaluate full bench"
$PY scripts/evaluate.py \
    --luc3d-dir outputs/PAF_3d_kalman/sleap_nn_filtered_tracklet_results \
    --pred-h5-dir "$PRED_H5_NEW" \
    --out-csv outputs/PAF_3d_kalman/_eval_sleap_nn_filtered.csv \
    --workers $WORKERS 2>&1 | tee -a logs/zero_swaps/eval_full.log

log "T10.7: register sleap_nn_filtered row in per_camera_session_metrics.csv"
$PY scripts/lib/append_metrics_row.py sleap_nn_filtered \
    outputs/PAF_3d_kalman/_eval_sleap_nn_filtered.csv

# ---------------------------------------------------------------------------
# Stage 2: LUC3D dedupe + KF hard gate (T17)
# ---------------------------------------------------------------------------
section "T17: Stage 2 sweep on debug + full bench"

log "T17.1: Stage 2 sweep (okeDedupThresh × kfHardGateMaha) on debug"
mkdir -p outputs/PAF_3d_kalman/_stage2_grid
for THR in 0.4 0.5 0.6; do
  for GATE in 16 25 36; do
    TAG="thr${THR}_gate${GATE}"
    [ -f "outputs/PAF_3d_kalman/_stage2_grid/_eval_${TAG}.csv" ] && { log "  skip ${TAG} (cached)"; continue; }
    cp "$PARAMS_DEDUPE" /tmp/p_${TAG}.json
    $PY -c "
import json
d = json.load(open('/tmp/p_${TAG}.json'))
d['okeDedupThresh'] = ${THR}
d['kfHardGateMaha'] = ${GATE}
json.dump(d, open('/tmp/p_${TAG}.json', 'w'))
"
    log "  ${TAG}: LUC3D + tracklet + eval"
    $PY scripts/run_luc3d_bench.py \
        --master outputs/sleap_nn_master_sheet_debug.tsv \
        --pred-h5-dir outputs/sleap_nn_predictions_h5s_debug \
        --params /tmp/p_${TAG}.json \
        --out-dir outputs/PAF_3d_kalman/_stage2_grid/${TAG} \
        --workers $WORKERS >> logs/zero_swaps/stage2_${TAG}.log 2>&1
    $PY scripts/stitch_postprocess.py \
        --in outputs/PAF_3d_kalman/_stage2_grid/${TAG} \
        --out outputs/PAF_3d_kalman/_stage2_grid/${TAG}_tracklet \
        --mode tracklet_vote --tracklet-gap 30 >> logs/zero_swaps/stage2_${TAG}.log 2>&1
    $PY scripts/evaluate.py \
        --luc3d-dir outputs/PAF_3d_kalman/_stage2_grid/${TAG}_tracklet \
        --pred-h5-dir outputs/sleap_nn_predictions_h5s_debug \
        --out-csv outputs/PAF_3d_kalman/_stage2_grid/_eval_${TAG}.csv \
        --workers $WORKERS >> logs/zero_swaps/stage2_${TAG}.log 2>&1
  done
done

log "T17.2: pick winner"
$PY - <<PY 2>&1 | tee -a "$PIPELINE_LOG"
import glob, re, json, pandas as pd
rows = []
for csv in glob.glob("outputs/PAF_3d_kalman/_stage2_grid/_eval_thr*.csv"):
    m = re.search(r"thr(\d+\.\d+)_gate(\d+)\.csv$", csv)
    if not m: continue
    df = pd.read_csv(csv)
    rows.append({"thr": float(m.group(1)), "gate": int(m.group(2)),
                 "switches": int(df["num_switches"].sum()),
                 "idf1": df["idf1"].mean(),
                 "csv": csv})
df = pd.DataFrame(rows).sort_values("switches")
print(df[["thr","gate","switches","idf1"]].to_string(index=False))
winner = df.iloc[0]
print(f"\nWinner: thr={winner['thr']} gate={winner['gate']}")
base = json.load(open("$PARAMS_BASE"))
base["okeDedupSigma"] = 20.0
base["okeDedupThresh"] = float(winner["thr"])
base["kfHardGateMaha"] = float(winner["gate"])
json.dump(base, open("$PARAMS_DEDUPE_WINNER", "w"), indent=2)
print("saved", "$PARAMS_DEDUPE_WINNER")
PY

log "T17.3: Stage 2 winner on full bench"
$PY scripts/run_luc3d_bench.py \
    --master "$FULL_MASTER_NEW" \
    --pred-h5-dir "$PRED_H5_NEW" \
    --params "$PARAMS_DEDUPE_WINNER" \
    --out-dir outputs/PAF_3d_kalman/luc3d_paf_l1_dedupe_results \
    --workers $WORKERS 2>&1 | tee -a logs/zero_swaps/stage2_full_luc3d.log

log "T17.4: tracklet_vote on Stage 2"
$PY scripts/stitch_postprocess.py \
    --in outputs/PAF_3d_kalman/luc3d_paf_l1_dedupe_results \
    --out outputs/PAF_3d_kalman/luc3d_paf_l1_dedupe_tracklet_results \
    --mode tracklet_vote --tracklet-gap 30 2>&1 | tee -a logs/zero_swaps/stage2_full_stitch.log

log "T17.5: evaluate Stage 2"
$PY scripts/evaluate.py \
    --luc3d-dir outputs/PAF_3d_kalman/luc3d_paf_l1_dedupe_tracklet_results \
    --pred-h5-dir "$PRED_H5_NEW" \
    --out-csv outputs/PAF_3d_kalman/_eval_luc3d_paf_l1_dedupe.csv \
    --workers $WORKERS 2>&1 | tee -a logs/zero_swaps/stage2_full_eval.log

$PY scripts/lib/append_metrics_row.py luc3d_paf_l1_dedupe \
    outputs/PAF_3d_kalman/_eval_luc3d_paf_l1_dedupe.csv

# ---------------------------------------------------------------------------
# Stage 3: tracklet_vote_3d on Stage 2 outputs (T19)
# ---------------------------------------------------------------------------
section "T19: tracklet_vote_3d on Stage 2 outputs"

# Read winning wg/wv from existing winner CSV (T18 selected wg=0.5, wv=0.0)
WG=0.5
WV=0.0

$PY scripts/stitch_postprocess.py \
    --in outputs/PAF_3d_kalman/luc3d_paf_l1_dedupe_results \
    --out outputs/PAF_3d_kalman/luc3d_paf_l1_dedupe_tracklet3d_results \
    --mode tracklet_vote_3d --tracklet-gap 30 \
    --w-geom $WG --w-velo $WV \
    --master "$FULL_MASTER_NEW" \
    --pred-h5-dir "$PRED_H5_NEW" 2>&1 | tee -a logs/zero_swaps/stage3.log

$PY scripts/evaluate.py \
    --luc3d-dir outputs/PAF_3d_kalman/luc3d_paf_l1_dedupe_tracklet3d_results \
    --pred-h5-dir "$PRED_H5_NEW" \
    --out-csv outputs/PAF_3d_kalman/_eval_luc3d_paf_l1_dedupe_tracklet3d.csv \
    --workers $WORKERS 2>&1 | tee -a logs/zero_swaps/stage3_eval.log

$PY scripts/lib/append_metrics_row.py luc3d_paf_l1_dedupe_tracklet3d \
    outputs/PAF_3d_kalman/_eval_luc3d_paf_l1_dedupe_tracklet3d.csv

# ---------------------------------------------------------------------------
# Stage 4: tracklet_stitch_3d sweep + full (T23)
# ---------------------------------------------------------------------------
section "T23: Stage 4 sweep + full bench"

log "T23.1: Stage 4 sweep on debug (gap × sigma × threshold)"
mkdir -p outputs/PAF_3d_kalman/_stage4_grid
# Run Stage 3 on debug first so we have an input for the stitch sweep:
$PY scripts/stitch_postprocess.py \
    --in outputs/PAF_3d_kalman/sleap_nn_filtered_debug_tracklet_results \
    --out outputs/PAF_3d_kalman/sleap_nn_filtered_debug_tracklet3d_results \
    --mode tracklet_vote_3d --tracklet-gap 30 \
    --w-geom $WG --w-velo $WV \
    --master outputs/sleap_nn_master_sheet_debug.tsv \
    --pred-h5-dir outputs/sleap_nn_predictions_h5s_debug >> logs/zero_swaps/stage4_prep.log 2>&1

for GAP in 30 60 120; do
  for SIG in 50 100 200; do
    for THR in 0.3 0.5 0.7; do
      TAG="g${GAP}_s${SIG}_t${THR}"
      [ -f "outputs/PAF_3d_kalman/_stage4_grid/_eval_${TAG}.csv" ] && { log "  skip ${TAG} (cached)"; continue; }
      log "  ${TAG}: stitch + eval"
      $PY scripts/stitch_postprocess.py \
          --in outputs/PAF_3d_kalman/sleap_nn_filtered_debug_tracklet3d_results \
          --out outputs/PAF_3d_kalman/_stage4_grid/${TAG} \
          --mode tracklet_stitch_3d \
          --stitch-max-gap $GAP --stitch-sigma $SIG --stitch-threshold $THR \
          --master outputs/sleap_nn_master_sheet_debug.tsv \
          --pred-h5-dir outputs/sleap_nn_predictions_h5s_debug >> logs/zero_swaps/stage4_${TAG}.log 2>&1
      $PY scripts/evaluate.py \
          --luc3d-dir outputs/PAF_3d_kalman/_stage4_grid/${TAG} \
          --pred-h5-dir outputs/sleap_nn_predictions_h5s_debug \
          --out-csv outputs/PAF_3d_kalman/_stage4_grid/_eval_${TAG}.csv \
          --workers $WORKERS >> logs/zero_swaps/stage4_${TAG}.log 2>&1
    done
  done
done

log "T23.2: pick Stage 4 winner"
$PY - <<PY 2>&1 | tee -a "$PIPELINE_LOG"
import glob, re, json, pandas as pd
rows = []
for csv in glob.glob("outputs/PAF_3d_kalman/_stage4_grid/_eval_g*.csv"):
    m = re.search(r"g(\d+)_s(\d+)_t(\d+\.\d+)\.csv$", csv)
    if not m: continue
    df = pd.read_csv(csv)
    rows.append({"gap": int(m.group(1)), "sig": int(m.group(2)),
                 "thr": float(m.group(3)),
                 "switches": int(df["num_switches"].sum()),
                 "idf1": df["idf1"].mean()})
df = pd.DataFrame(rows).sort_values("switches")
print(df.head(10).to_string(index=False))
w = df.iloc[0]
json.dump({"max_gap": int(w["gap"]), "sigma": int(w["sig"]),
           "threshold": float(w["thr"])},
          open("outputs/PAF_3d_kalman/stage4_winner.json", "w"), indent=2)
print(f"\nWinner: gap={w['gap']} sig={w['sig']} thr={w['thr']}")
PY

log "T23.3: Stage 4 winner on full bench"
$PY - <<'PY' > /tmp/stage4_winner.sh
import json
w = json.load(open("outputs/PAF_3d_kalman/stage4_winner.json"))
print(f"export STAGE4_GAP={w['max_gap']} STAGE4_SIG={w['sigma']} STAGE4_THR={w['threshold']}")
PY
source /tmp/stage4_winner.sh

$PY scripts/stitch_postprocess.py \
    --in outputs/PAF_3d_kalman/luc3d_paf_l1_dedupe_tracklet3d_results \
    --out outputs/PAF_3d_kalman/luc3d_paf_l1_dedupe_tracklet3d_stitched_results \
    --mode tracklet_stitch_3d \
    --stitch-max-gap $STAGE4_GAP --stitch-sigma $STAGE4_SIG --stitch-threshold $STAGE4_THR \
    --master "$FULL_MASTER_NEW" \
    --pred-h5-dir "$PRED_H5_NEW" 2>&1 | tee -a logs/zero_swaps/stage4_full.log

$PY scripts/evaluate.py \
    --luc3d-dir outputs/PAF_3d_kalman/luc3d_paf_l1_dedupe_tracklet3d_stitched_results \
    --pred-h5-dir "$PRED_H5_NEW" \
    --out-csv outputs/PAF_3d_kalman/_eval_luc3d_paf_l1_dedupe_tracklet3d_stitched.csv \
    --workers $WORKERS 2>&1 | tee -a logs/zero_swaps/stage4_full_eval.log

$PY scripts/lib/append_metrics_row.py luc3d_paf_l1_dedupe_tracklet3d_stitched \
    outputs/PAF_3d_kalman/_eval_luc3d_paf_l1_dedupe_tracklet3d_stitched.csv

# ---------------------------------------------------------------------------
# Reporting (T24, T25, T26)
# ---------------------------------------------------------------------------
section "T24-T26: reports + figure + final markdown"

log "T24: merge_all_variants"
if [ -f scripts/merge_all_variants.py ]; then
    $PY scripts/merge_all_variants.py 2>&1 | tee -a logs/zero_swaps/merge.log || \
        log "merge_all_variants failed — append_metrics_row already wrote rows directly"
fi

log "T25: regenerate figure_main"
if [ -f scripts/make_plots_paf.py ]; then
    $PY scripts/make_plots_paf.py 2>&1 | tee -a logs/zero_swaps/plot.log || \
        log "figure regen failed — check scripts/make_plots_paf.py tracker list"
fi

log "T26: write zero_swaps_report.md"
$PY scripts/lib/write_zero_swaps_report.py

log ""
log "PIPELINE COMPLETE. Headline:"
$PY - <<'PY' | tee -a "$PIPELINE_LOG"
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
                 "per_sess": round(sub["num_switches"].sum() / len(sub), 2),
                 "IDF1": round(sub["idf1"].mean(), 4),
                 "IDP": round(sub["idp"].mean(), 4)})
print(pd.DataFrame(rows).to_string(index=False))
PY
