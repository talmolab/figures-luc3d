#!/usr/bin/env bash
# Wait until the detection H5s are built (by the orchestrator's build_inputs stage),
# then run 3D-MuPPET on all sessions so it's ready before the aggregate eval.
set -uo pipefail
cd /root/vast/eric/luc3d-bench
BM=/root/vast/eric/luc3d-bench/outputs/bmimica
LOG=$BM/muppet_run.log
echo "[muppet-watcher $(date +%H:%M:%S)] waiting for detection H5s..." >> $LOG
# wait until build_inputs reports done (or det_h5 has 250 files)
until grep -q "^DONE build_inputs" $BM/build_inputs.log 2>/dev/null || \
      [ "$(find $BM/det_h5 -name '*_predictions.h5' 2>/dev/null | wc -l)" -ge 250 ]; do
  sleep 60
done
echo "[muppet-watcher $(date +%H:%M:%S)] det H5s ready, running 3D-MuPPET" >> $LOG
/root/vast/eric/muppet_env/bin/python scripts/bartul/bmimica_muppet.py --workers 12 >> $LOG 2>&1
echo "[muppet-watcher $(date +%H:%M:%S)] 3D-MuPPET DONE: $(tail -1 $LOG)" >> $LOG
