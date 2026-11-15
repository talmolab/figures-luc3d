#!/bin/bash
# Track A — 3D-anchored tracklet vote sweep.
# Generates 12 configs (w_geom x w_velo) on PAF L1 outputs, runs evaluate.py
# on each, and reports the best config by total switches and IDF1.

set -euo pipefail
cd /root/vast/eric/luc3d-bench
source .venv/bin/activate

PROGRESS_LOG="outputs/PAF_3d_kalman/track_a_progress.log"
OUT_ROOT="outputs/PAF_3d_kalman/luc3d_l1_tracklet_3d_results"
INPUT_DIR="outputs/PAF_3d_kalman/luc3d_l1_only_results"
EVAL_PREFIX="outputs/PAF_3d_kalman/_eval_tracklet_3d"

W_GEOMS=("0.5" "1.0" "2.0" "5.0")
W_VELOS=("0.0" "0.5" "1.0")

for wg in "${W_GEOMS[@]}"; do
  for wv in "${W_VELOS[@]}"; do
    cfg_id="wg${wg}_wv${wv}"
    out_dir="${OUT_ROOT}/${cfg_id}"
    eval_csv="${EVAL_PREFIX}_${cfg_id}.csv"

    if [ -d "${out_dir}" ] && [ -f "${eval_csv}" ] && \
       [ "$(ls -1 ${out_dir}/*.json 2>/dev/null | wc -l)" -eq 74 ]; then
      echo "$(date -Iseconds) SKIP ${cfg_id} (already done)" >> "${PROGRESS_LOG}"
      continue
    fi

    echo "$(date -Iseconds) ===== STARTING ${cfg_id} =====" >> "${PROGRESS_LOG}"

    python scripts/stitch_postprocess.py \
      --in "${INPUT_DIR}" --out "${out_dir}" \
      --mode tracklet_vote_3d \
      --tracklet-gap 30 \
      --w-geom "${wg}" --w-velo "${wv}" \
      --sigma-geom 50 --sigma-velo 20 \
      --pred-h5-dir outputs/predictions_h5s \
      --master-sheet outputs/predictions_master_sheet.tsv \
      --progress-log "${PROGRESS_LOG}" >> "${PROGRESS_LOG}" 2>&1

    echo "$(date -Iseconds) ===== EVAL ${cfg_id} =====" >> "${PROGRESS_LOG}"
    python scripts/evaluate.py \
      --luc3d-dir "${out_dir}" \
      --out-csv "${eval_csv}" \
      --workers 16 >> "${PROGRESS_LOG}" 2>&1
  done
done

echo "$(date -Iseconds) ===== ALL CONFIGS DONE =====" >> "${PROGRESS_LOG}"
