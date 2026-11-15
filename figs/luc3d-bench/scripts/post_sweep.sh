#!/usr/bin/env bash
# Overnight chain: waits for the filter sweep to finish, picks the best filter
# for each tracker (LUC3D-no-PAF and Liezl), runs on full bench, then renders
# the new FINAL figure with no-PAF-L1 rows.
set -uo pipefail

REPO=$(cd "$(dirname "$0")/.." && pwd)
cd "$REPO"
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"

PY=.venv/bin/python
SLEAP_NN=sleap_nn_env/bin/python
LIEZL_PY=liezl_env/bin/python
LOG=logs/post_sweep.log
mkdir -p logs

log() { echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$LOG"; }

log "============================================================"
log "POST-SWEEP CHAIN STARTED"
log "============================================================"

# 1. Wait for the sweep PID
if [ -f /tmp/sweep_pid.txt ]; then
    SWEEP_PID=$(cat /tmp/sweep_pid.txt)
    log "Waiting on sweep PID $SWEEP_PID"
    last_results=0
    while ps -p "$SWEEP_PID" > /dev/null 2>&1; do
        sleep 300
        n=$(wc -l < outputs/PAF_3d_kalman/_filter_sweep/sweep_results.csv 2>/dev/null || echo 1)
        if [ "$n" -ne "$last_results" ]; then
            log "  sweep_results.csv: $((n-1)) configs evaluated"
            last_results=$n
        fi
    done
    log "Sweep PID $SWEEP_PID exited"
fi

# 2. Pick winning filter for Liezl-pairing (minimize switches, IDF1 >= 0.55)
#    and winning filter for LUC3D-pairing (minimize switches, IDF1 >= 0.74)
log "Picking winning filter configs from sweep_results.csv"
$PY - <<'PY' 2>&1 | tee -a "$LOG"
import pandas as pd
df = pd.read_csv("outputs/PAF_3d_kalman/_filter_sweep/sweep_results.csv")
df = df[df["stage"] == "ok"].copy()
print(f"sweep has {len(df)} successful configs")
print("\nAll configs sorted by Liezl switches:")
cols = ["tag", "min_mean_node_score", "overlapping_threshold", "min_visible_nodes",
        "luc3d_no_paf_switches", "luc3d_no_paf_idf1",
        "liezl_switches", "liezl_idf1"]
avail = [c for c in cols if c in df.columns]
print(df.sort_values("liezl_switches")[avail].to_string(index=False))

# Picking strategy:
# - For Liezl: minimize switches subject to liezl_idf1 >= 0.55
#   (Liezl IDF1 was 0.6421 at baseline filter; allow slight regression for big switch win)
# - For LUC3D: minimize switches subject to luc3d_no_paf_idf1 >= 0.74
#   (close to baseline IDF1)

def pick(df, sw_col, idf1_col, idf1_min):
    ok = df[df[idf1_col] >= idf1_min]
    if len(ok) == 0:
        ok = df  # fall back to all
    return ok.sort_values(sw_col).iloc[0]

best_l = pick(df, "liezl_switches", "liezl_idf1", 0.55)
best_u = pick(df, "luc3d_no_paf_switches", "luc3d_no_paf_idf1", 0.74)

print(f"\nWinner for LIEZL:  tag={best_l['tag']}  sw={best_l['liezl_switches']}  IDF1={best_l['liezl_idf1']:.4f}")
print(f"Winner for LUC3D:  tag={best_u['tag']}  sw={best_u['luc3d_no_paf_switches']}  IDF1={best_u['luc3d_no_paf_idf1']:.4f}")

import json
winners = {
    "liezl": {
        "min_instance_score": float(best_l["min_instance_score"]),
        "min_mean_node_score": float(best_l["min_mean_node_score"]),
        "min_visible_nodes": int(best_l["min_visible_nodes"]),
        "overlapping_threshold": float(best_l["overlapping_threshold"]),
        "window_size": int(best_l["window_size"]),
    },
    "luc3d_no_paf": {
        "min_instance_score": float(best_u["min_instance_score"]),
        "min_mean_node_score": float(best_u["min_mean_node_score"]),
        "min_visible_nodes": int(best_u["min_visible_nodes"]),
        "overlapping_threshold": float(best_u["overlapping_threshold"]),
        "window_size": int(best_u["window_size"]),
    },
}
json.dump(winners, open("outputs/PAF_3d_kalman/_filter_sweep/winners.json", "w"), indent=2)
print("\nsaved outputs/PAF_3d_kalman/_filter_sweep/winners.json")
PY

# 3. Determine which winner needs a fresh full-bench retrack
$PY - <<'PY' 2>&1 | tee -a "$LOG"
import json, hashlib
winners = json.load(open("outputs/PAF_3d_kalman/_filter_sweep/winners.json"))
current = dict(min_instance_score=0.85, min_mean_node_score=0.55,
               min_visible_nodes=8, overlapping_threshold=0.50, window_size=15)
needs = []
for kind, w in winners.items():
    if w == current:
        print(f"{kind}: winner == current baseline (no re-retrack needed)")
    else:
        print(f"{kind}: winner differs from baseline — needs full-bench retrack")
        needs.append(kind)
open("/tmp/post_sweep_needs.txt", "w").write("\n".join(needs) + "\n" if needs else "")
PY

NEEDS=$(cat /tmp/post_sweep_needs.txt 2>/dev/null | head -5)
log "Configs needing fresh full-bench retracks: $NEEDS"

# 4. For each unique winning config, do full-bench retrack + downstream
process_config() {
    local KIND=$1
    log "===== Processing winner for $KIND ====="
    local W_JSON=outputs/PAF_3d_kalman/_filter_sweep/winners.json
    local MN=$($PY -c "import json; print(json.load(open('$W_JSON'))['$KIND']['min_mean_node_score'])")
    local IN=$($PY -c "import json; print(json.load(open('$W_JSON'))['$KIND']['min_instance_score'])")
    local VN=$($PY -c "import json; print(json.load(open('$W_JSON'))['$KIND']['min_visible_nodes'])")
    local OV=$($PY -c "import json; print(json.load(open('$W_JSON'))['$KIND']['overlapping_threshold'])")
    local WS=$($PY -c "import json; print(json.load(open('$W_JSON'))['$KIND']['window_size'])")
    local TAG="mn${MN}_in${IN}_vn${VN}_ov${OV}_ws${WS}"
    log "  filter: $TAG"
    local RETRACK_DIR=outputs/sleap_nn_retracked_${KIND}
    local H5_DIR=outputs/sleap_nn_predictions_h5s_${KIND}
    local MSHEET=outputs/sleap_nn_master_sheet_${KIND}.tsv

    # Full-bench retrack with this filter
    log "  retracking 444 cam-sessions for $KIND"
    $SLEAP_NN scripts/sleap_nn/retrack_all.py \
        --master outputs/predictions_master_sheet.tsv \
        --out-dir "$RETRACK_DIR" \
        --workers 16 \
        --min-instance-score $IN --min-mean-node-score $MN \
        --min-visible-nodes $VN --overlapping-threshold $OV --window-size $WS \
        2>&1 | tee -a logs/post_sweep_${KIND}_retrack.log | tail -5

    log "  aggregating H5s for $KIND"
    $SLEAP_NN scripts/sleap_nn/aggregate_to_h5.py \
        --master outputs/predictions_master_sheet.tsv \
        --retrack-dir "$RETRACK_DIR" \
        --out-dir "$H5_DIR" 2>&1 | tail -10

    log "  building master sheet for $KIND"
    $SLEAP_NN scripts/sleap_nn/build_master_sheet.py \
        --src outputs/predictions_master_sheet.tsv \
        --retrack-dir "$RETRACK_DIR" \
        --dst "$MSHEET" 2>&1 | tail -3

    if [ "$KIND" = "liezl" ]; then
        log "  running Liezl on best-filter retracks"
        local OUT_DIR=outputs/PAF_3d_kalman/liezl_on_sleap_nn_best_results
        $LIEZL_PY scripts/liezl/run_all_sessions.py \
            --master-sheet "$MSHEET" \
            --out-dir "$OUT_DIR" \
            --workers 8 --skip-existing 2>&1 | tee logs/post_sweep_liezl_full.log | tail -5
        $PY scripts/evaluate.py \
            --luc3d-dir "$OUT_DIR" \
            --pred-h5-dir "$H5_DIR" \
            --out-csv outputs/PAF_3d_kalman/_eval_liezl_on_sleap_nn_best.csv \
            --workers 16 2>&1 | tail -3
        $PY scripts/lib/append_metrics_row.py liezl_on_sleap_nn_best \
            outputs/PAF_3d_kalman/_eval_liezl_on_sleap_nn_best.csv | tee -a "$LOG"

        # ALSO apply tracklet_vote to Liezl output:
        log "  tracklet_vote post-process on Liezl output"
        local OUT_TV=outputs/PAF_3d_kalman/liezl_on_sleap_nn_best_tracklet_results
        $PY scripts/stitch_postprocess.py \
            --in "$OUT_DIR" --out "$OUT_TV" \
            --mode tracklet_vote --tracklet-gap 30 2>&1 | tail -3
        $PY scripts/evaluate.py \
            --luc3d-dir "$OUT_TV" \
            --pred-h5-dir "$H5_DIR" \
            --out-csv outputs/PAF_3d_kalman/_eval_liezl_on_sleap_nn_best_tracklet.csv \
            --workers 16 2>&1 | tail -3
        $PY scripts/lib/append_metrics_row.py liezl_on_sleap_nn_best_tracklet \
            outputs/PAF_3d_kalman/_eval_liezl_on_sleap_nn_best_tracklet.csv | tee -a "$LOG"
    fi

    if [ "$KIND" = "luc3d_no_paf" ]; then
        log "  running LUC3D no-PAF on best-filter retracks"
        local OUT_DIR=outputs/PAF_3d_kalman/sleap_nn_filtered_best_no_paf_results
        local OUT_TV=outputs/PAF_3d_kalman/sleap_nn_filtered_best_no_paf_tracklet_results
        $PY scripts/run_luc3d_bench.py \
            --master "$MSHEET" --pred-h5-dir "$H5_DIR" \
            --params outputs/PAF_3d_kalman/no_paf_params.json \
            --out-dir "$OUT_DIR" --workers 16 2>&1 | tail -5
        $PY scripts/stitch_postprocess.py \
            --in "$OUT_DIR" --out "$OUT_TV" \
            --mode tracklet_vote --tracklet-gap 30 2>&1 | tail -3
        $PY scripts/evaluate.py \
            --luc3d-dir "$OUT_TV" \
            --pred-h5-dir "$H5_DIR" \
            --out-csv outputs/PAF_3d_kalman/_eval_sleap_nn_filtered_best_no_paf.csv \
            --workers 16 2>&1 | tail -3
        $PY scripts/lib/append_metrics_row.py sleap_nn_filtered_best_no_paf \
            outputs/PAF_3d_kalman/_eval_sleap_nn_filtered_best_no_paf.csv | tee -a "$LOG"
    fi
}

# Also always apply tracklet_vote to the existing liezl_on_sleap_nn (default filter)
log "===== Applying tracklet_vote to existing liezl_on_sleap_nn (baseline filter) ====="
$PY scripts/stitch_postprocess.py \
    --in outputs/PAF_3d_kalman/liezl_on_sleap_nn_results \
    --out outputs/PAF_3d_kalman/liezl_on_sleap_nn_tracklet_results \
    --mode tracklet_vote --tracklet-gap 30 2>&1 | tail -3 | tee -a "$LOG"
$PY scripts/evaluate.py \
    --luc3d-dir outputs/PAF_3d_kalman/liezl_on_sleap_nn_tracklet_results \
    --pred-h5-dir outputs/sleap_nn_predictions_h5s \
    --out-csv outputs/PAF_3d_kalman/_eval_liezl_on_sleap_nn_tracklet.csv \
    --workers 16 2>&1 | tail -3
$PY scripts/lib/append_metrics_row.py liezl_on_sleap_nn_tracklet \
    outputs/PAF_3d_kalman/_eval_liezl_on_sleap_nn_tracklet.csv | tee -a "$LOG"

# Process unique winners (avoid duplicate retrack if same config wins both)
if [ -s /tmp/post_sweep_needs.txt ]; then
    declare -A done_filters
    while IFS= read -r KIND; do
        [ -z "$KIND" ] && continue
        FILT_KEY=$($PY -c "import json; w=json.load(open('outputs/PAF_3d_kalman/_filter_sweep/winners.json'))['$KIND']; print('-'.join(f'{k}={v}' for k,v in sorted(w.items())))")
        if [ -n "${done_filters[$FILT_KEY]:-}" ]; then
            log "$KIND uses same filter as ${done_filters[$FILT_KEY]} — reusing retracks"
            # Symlink retrack/h5/msheet to existing
            OTHER=${done_filters[$FILT_KEY]}
            ln -sfn "$(realpath outputs/sleap_nn_retracked_${OTHER})" outputs/sleap_nn_retracked_${KIND}
            ln -sfn "$(realpath outputs/sleap_nn_predictions_h5s_${OTHER})" outputs/sleap_nn_predictions_h5s_${KIND}
            ln -sfn "$(realpath outputs/sleap_nn_master_sheet_${OTHER}.tsv)" outputs/sleap_nn_master_sheet_${KIND}.tsv
        fi
        process_config "$KIND"
        done_filters[$FILT_KEY]="$KIND"
    done < /tmp/post_sweep_needs.txt
fi

# 5. Render the FINAL figure with NO PAF L1 rows
log "===== Rendering final figure (no PAF L1) ====="
$PY scripts/make_plots_zero_swaps.py figure_main_zero_swaps_FINAL 2>&1 | tail -20 | tee -a "$LOG"
log "FINAL figure: outputs/PAF_3d_kalman/plots/figure_main_zero_swaps_FINAL.png"

# 6. Print the headline table
log "===== HEADLINE TABLE ====="
$PY - <<'PY' 2>&1 | tee -a "$LOG"
import pandas as pd
df = pd.read_csv("outputs/PAF_3d_kalman/metrics/per_camera_session_metrics.csv")
order = ["sleap","bytetrack","luc3d",
         "liezl_cross_view",
         "sleap_nn_filtered_no_paf",
         "liezl_on_sleap_nn",
         "liezl_on_sleap_nn_tracklet",
         "sleap_nn_filtered_best_no_paf",
         "liezl_on_sleap_nn_best",
         "liezl_on_sleap_nn_best_tracklet"]
rows=[]
for t in order:
    s = df[df.tracker==t]
    if len(s)==0: continue
    rows.append({"tracker":t,"n":len(s),"sw":int(s["num_switches"].sum()),
                 "/sess":round(s["num_switches"].sum()/len(s),2),
                 "IDF1":round(s["idf1"].mean(),4),
                 "IDP":round(s["idp"].mean(),4)})
print(pd.DataFrame(rows).to_string(index=False))
PY

log "============================================================"
log "POST-SWEEP CHAIN COMPLETE"
log "============================================================"
