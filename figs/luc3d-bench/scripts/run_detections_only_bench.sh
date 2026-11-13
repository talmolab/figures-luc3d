#!/usr/bin/env bash
# Detections-only ID-tracking benchmark — one-command, resumable, guarded.
# Every tracker sees the IDENTICAL per-frame detection pool; no data dropped.
#
# Launch:
#   cd /root/vast/eric/luc3d-bench
#   nohup bash scripts/run_detections_only_bench.sh > logs/detok_overnight.stdout 2>&1 &
#   tail -f logs/detok_overnight.log
set -uo pipefail
cd /root/vast/eric/luc3d-bench

# Node binary (restored under tools/) on PATH for the LUC3D driver subprocess.
export PATH="/usr/local/bin:/root/vast/eric/luc3d-bench/tools/node-v20.18.0-linux-x64/bin:$PATH"

mkdir -p logs
LOG=logs/detok_overnight.log
echo "BEGIN $(date -Is)  node=$(node --version 2>/dev/null || echo MISSING)" | tee -a "$LOG"

# Env pythons
PY_SLEAPNN="sleap_nn_env/bin/python"
PY_MAIN=".venv/bin/python"
PY_LIEZL="liezl_env/bin/python"

WORKERS="${DETOK_WORKERS:-16}"
LIEZL_WORKERS="${DETOK_LIEZL_WORKERS:-8}"
MASTER="outputs/detections_only_master_sheet.tsv"
DETOK_H5="outputs/detections_only_h5s"
PAF="outputs/PAF_3d_kalman"
PARAMS="outputs/detok_params.json"

stage () { echo "[$(date -Is)] STAGE: $*" | tee -a "$LOG"; }
guard () { "$@" || { echo "[$(date -Is)] STAGE FAILED: $*" | tee -a "$LOG"; }; }

# 1. Filter-only detection export (resumable via ledger)
stage "1 filter-only export"
guard env PYTHONPATH=. $PY_SLEAPNN scripts/sleap_nn/filter_only_all.py --workers "$WORKERS" \
    2>&1 | tee -a logs/filter_only.log

# 2. Widened detections H5 (no drop)
stage "2 aggregate widened detections H5"
guard env PYTHONPATH=. $PY_SLEAPNN scripts/sleap_nn/aggregate_detections_h5.py \
    2>&1 | tee -a logs/aggregate_detok.log

# 3. Detections master sheet
stage "3 build detections master sheet"
guard env PYTHONPATH=. $PY_MAIN scripts/sleap_nn/build_detections_master_sheet.py \
    2>&1 | tee -a "$LOG"

# 4. LUC3D (detections-only mode via params: detectionsOnly=true)
stage "4 LUC3D (detections-only)"
guard $PY_MAIN scripts/run_luc3d_bench.py --master "$MASTER" \
    --pred-h5-dir "$DETOK_H5" --out-dir "$PAF/detok_luc3d_results" \
    --params "$PARAMS" --workers "$WORKERS" \
    2>&1 | tee -a logs/detok_luc3d.log

# 5. Liezl (same detections, canonical det_pos)
stage "5 Liezl (detections-only)"
guard $PY_LIEZL scripts/liezl/run_all_sessions.py --master-sheet "$MASTER" \
    --out-dir "$PAF/detok_liezl_results" --workers "$LIEZL_WORKERS" \
    2>&1 | tee -a logs/detok_liezl.log

# 6. ByteTrack (same detections)
stage "6 ByteTrack (detections-only)"
guard $PY_MAIN scripts/run_bytetrack_bench.py --master "$MASTER" \
    --pred-h5-dir "$DETOK_H5" --out-dir "$PAF/detok_bytetrack_results" --workers "$WORKERS" \
    2>&1 | tee -a logs/detok_bytetrack.log

# 7. Eval + register. LUC3D eval also scores the shared ByteTrack H5 (one pass);
#    Liezl eval suppresses ByteTrack to avoid double-scoring.
stage "7a eval LUC3D (+ByteTrack)"
guard $PY_MAIN scripts/evaluate.py --luc3d-dir "$PAF/detok_luc3d_results" \
    --master "$MASTER" --pred-h5-dir "$DETOK_H5" --bytetrack-dir "$PAF/detok_bytetrack_results" \
    --out-csv "$PAF/_eval_detok_luc3d.csv" --no-sleap-baseline --workers "$WORKERS" \
    2>&1 | tee -a logs/detok_eval.log
guard env PYTHONPATH=. $PY_MAIN scripts/lib/append_metrics_row.py detok_luc3d "$PAF/_eval_detok_luc3d.csv" 2>&1 | tee -a logs/detok_eval.log
guard env PYTHONPATH=. $PY_MAIN scripts/lib/append_metrics_row.py detok_bytetrack "$PAF/_eval_detok_luc3d.csv" --from-tracker bytetrack 2>&1 | tee -a logs/detok_eval.log

stage "7b eval Liezl"
guard $PY_MAIN scripts/evaluate.py --luc3d-dir "$PAF/detok_liezl_results" \
    --master "$MASTER" --pred-h5-dir "$DETOK_H5" --out-csv "$PAF/_eval_detok_liezl.csv" \
    --no-sleap-baseline --no-bytetrack --workers "$WORKERS" \
    2>&1 | tee -a logs/detok_eval.log
guard env PYTHONPATH=. $PY_MAIN scripts/lib/append_metrics_row.py detok_liezl "$PAF/_eval_detok_liezl.csv" 2>&1 | tee -a logs/detok_eval.log

# 8. SLEAP single-view reference — on the OLD tracked H5 (legitimate stable-slot identity)
stage "8 SLEAP reference (old tracked H5)"
guard $PY_MAIN scripts/evaluate.py --luc3d-dir "$PAF/detok_luc3d_results" \
    --pred-h5-dir outputs/sleap_nn_predictions_h5s --no-bytetrack \
    --out-csv "$PAF/_eval_sleap_ref.csv" --workers "$WORKERS" \
    2>&1 | tee -a logs/detok_eval.log
guard env PYTHONPATH=. $PY_MAIN scripts/lib/append_metrics_row.py detok_sleap_ref "$PAF/_eval_sleap_ref.csv" --from-tracker sleap 2>&1 | tee -a logs/detok_eval.log

# 9. Figure + report
stage "9 figure + report"
guard env PYTHONPATH=. $PY_MAIN scripts/make_plots_zero_swaps.py figure_detections_only 2>&1 | tee -a "$LOG"
guard env PYTHONPATH=. $PY_MAIN scripts/lib/write_detok_report.py --out "$PAF/detections_only_report.md" 2>&1 | tee -a "$LOG"

echo "DONE $(date -Is)" | tee -a "$LOG"
