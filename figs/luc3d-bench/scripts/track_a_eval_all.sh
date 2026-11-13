#!/bin/bash
# Track A — eval all 12 sweep configs. Run 3 evals in parallel at 6 workers each.

set -euo pipefail
cd /root/vast/eric/luc3d-bench
source .venv/bin/activate

PROGRESS_LOG="outputs/PAF_3d_kalman/track_a_progress.log"
OUT_ROOT="outputs/PAF_3d_kalman/luc3d_l1_tracklet_3d_results"
EVAL_PREFIX="outputs/PAF_3d_kalman/_eval_tracklet_3d"

W_GEOMS=("0.5" "1.0" "2.0" "5.0")
W_VELOS=("0.0" "0.5" "1.0")

run_one() {
  local wg=$1
  local wv=$2
  local cfg_id="wg${wg}_wv${wv}"
  local out_dir="${OUT_ROOT}/${cfg_id}"
  local eval_csv="${EVAL_PREFIX}_${cfg_id}.csv"

  if [ -f "${eval_csv}" ]; then
    local n_rows
    n_rows=$(wc -l < "${eval_csv}")
    if [ "${n_rows}" -gt 100 ]; then
      echo "$(date -Iseconds) SKIP eval ${cfg_id} (csv has ${n_rows} rows)" >> "${PROGRESS_LOG}"
      return 0
    fi
  fi

  if [ ! -d "${out_dir}" ]; then
    echo "$(date -Iseconds) WARN: missing ${out_dir}, skipping ${cfg_id}" >> "${PROGRESS_LOG}"
    return 0
  fi
  local n_json
  n_json=$(ls -1 "${out_dir}"/*.json 2>/dev/null | wc -l)
  if [ "${n_json}" -lt 74 ]; then
    echo "$(date -Iseconds) WARN: ${out_dir} has only ${n_json} JSONs, skipping ${cfg_id}" >> "${PROGRESS_LOG}"
    return 0
  fi

  echo "$(date -Iseconds) eval START ${cfg_id}" >> "${PROGRESS_LOG}"
  python scripts/evaluate.py \
    --luc3d-dir "${out_dir}" \
    --out-csv "${eval_csv}" \
    --workers 12 >> "${PROGRESS_LOG}" 2>&1
  echo "$(date -Iseconds) eval DONE  ${cfg_id}" >> "${PROGRESS_LOG}"
}

# Build the full config list.
configs=()
for wg in "${W_GEOMS[@]}"; do
  for wv in "${W_VELOS[@]}"; do
    configs+=("${wg} ${wv}")
  done
done

# Run 4 evals at a time, 12 workers each (48 cores total).
i=0
total=${#configs[@]}
while [ "${i}" -lt "${total}" ]; do
  pids=()
  for k in 0 1 2 3; do
    if [ $((i+k)) -lt "${total}" ]; then
      c=(${configs[$((i+k))]})
      run_one "${c[0]}" "${c[1]}" &
      pids+=($!)
    fi
  done
  for p in "${pids[@]}"; do
    wait "${p}"
  done
  i=$((i+4))
done

echo "$(date -Iseconds) ===== ALL EVALS DONE =====" >> "${PROGRESS_LOG}"
