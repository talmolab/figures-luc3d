#!/usr/bin/env bash
# Wait for UDMT master to finish, then score UDMT vs GT and regenerate the figure.
cd /root/vast/eric/luc3d-bench
MASTER=outputs/bartul_smoke/results/udmt_master.log
# wait up to 8h for completion line
for i in $(seq 1 2880); do
  if grep -q "ALL UDMT CAMERAS DONE" "$MASTER" 2>/dev/null; then break; fi
  sleep 10
done
echo "[watcher] UDMT master finished at $(date +%T)" >> outputs/bartul_smoke/results/udmt_watcher.log
.venv/bin/python scripts/bartul/score_udmt_bartul.py >> outputs/bartul_smoke/results/udmt_watcher.log 2>&1
lp3d_env/bin/python scripts/bartul/make_plots_bartul.py >> outputs/bartul_smoke/results/udmt_watcher.log 2>&1
echo "[watcher] scored + figure regenerated at $(date +%T)" >> outputs/bartul_smoke/results/udmt_watcher.log
