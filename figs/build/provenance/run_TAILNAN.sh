#!/usr/bin/env bash
# B3 (2026-10-15): LUC3D with the tail nulled in the tracker INPUT vs the v1.0.5 reference outputs.
set -uo pipefail
cd /root/vast/eric/sleap-3d-gui/scratch/repos/lucid
export PATH=/root/vast/eric/node26_env/bin:$PATH LUCID_ROOT=/root/vast/eric/lucid-main-md56
echo "[$(date -u +%T) $(hostname -s)] start (load $(cut -d' ' -f1 /proc/loadavg))"
/root/vast/eric/luc3d-bench/liezl_env/bin/python figs/tailnan_check.py --workers 24 > figs/out/md56_logs/TAILNAN.log 2>&1
rc=$?
echo "[$(date -u +%T)] tailnan exit=$rc"; tail -3 figs/out/md56_logs/TAILNAN.log
[ $rc -eq 0 ] && echo "TAILNAN RUN DONE" || echo "TAILNAN RUN FAILED"
exit $rc
