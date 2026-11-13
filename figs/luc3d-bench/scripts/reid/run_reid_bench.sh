#!/usr/bin/env bash
# Phase-1 orchestration: embed every session's detections with the frozen OSNet encoder
# and relabel Liezl's identities (unsupervised, Graph 1). Sessions are split across the
# two A40s. Idempotent — cached embeddings and existing result JSONs are skipped.
#
# Usage:
#   scripts/reid/run_reid_bench.sh [weights_tag] [h5_dir] [base_dir] [out_dir]
# Defaults relabel detok_liezl on the filter-only pool. For ReID on the BEST base:
#   scripts/reid/run_reid_bench.sh imagenet_keeptrack outputs/keeptrack_h5s \
#       outputs/PAF_3d_kalman/keeptrack_liezl_results \
#       outputs/PAF_3d_kalman/keeptrack_liezl_reid_results
# Logs to logs/reid_bench_gpu{0,1}.log
set -euo pipefail
cd "$(dirname "$0")/../.."

TAG="${1:-imagenet}"
H5DIR="${2:-outputs/detections_only_h5s}"
BASEDIR="${3:-outputs/PAF_3d_kalman/detok_liezl_results}"
OUTDIR="${4:-outputs/PAF_3d_kalman/detok_liezl_reid_results}"
PY=reid_env/bin/python
mkdir -p logs

# Session ids in master-sheet order, tagged with multi-animal flag. Single-animal
# sessions are NOT embedded — relabel just copies Liezl for them (no appearance needed).
mapfile -t SESSIONS < <(.venv/bin/python - <<'PY'
import pandas as pd
df = pd.read_csv("outputs/detections_only_master_sheet.tsv", sep="\t", index_col=0)
for s, n in zip(df["session"].astype(str), df["num_animals"].astype(int)):
    print(f"{s}\t{1 if n >= 2 else 0}")
PY
)

run_half () {           # $1 = gpu id, $2 = parity (0/1)
  local gpu="$1" parity="$2" i=0
  for line in "${SESSIONS[@]}"; do
    local s="${line%%$'\t'*}" multi="${line##*$'\t'}"
    if (( i % 2 == parity )); then
      if (( multi == 1 )); then
        echo "[gpu$gpu] embed $s"
        CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH=. $PY scripts/reid/embed_session.py \
          --session "$s" --weights-tag "$TAG" --h5-dir "$H5DIR" 2>&1 | grep -E '\[ok\]|\[skip\]|Error|Trace' || true
      fi
      echo "[gpu$gpu] relabel $s (multi=$multi)"
      PYTHONPATH=. $PY scripts/reid/relabel.py --session "$s" --weights-tag "$TAG" \
        --base-dir "$BASEDIR" --out-dir "$OUTDIR" 2>&1 \
        | grep -vE 'pkg_resources|UserWarning|import pkg' || true
    fi
    i=$((i+1))
  done
}

run_half 0 0 > logs/reid_bench_gpu0.log 2>&1 &
P0=$!
run_half 1 1 > logs/reid_bench_gpu1.log 2>&1 &
P1=$!
echo "launched gpu0 pid=$P0 gpu1 pid=$P1 (tag=$TAG)"
wait $P0 $P1
echo "ALL DONE (tag=$TAG)"
