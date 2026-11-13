#!/usr/bin/env bash
# md56 F4main (2026-10-14): Fig 4D re-tracked on origin/main's real tracker, digest-compared.
set -uo pipefail
cd /root/vast/eric/sleap-3d-gui/scratch/repos/lucid
export PATH=/root/vast/eric/node26_env/bin:$PATH LUCID_ROOT=/root/vast/eric/lucid-main-md56
/root/vast/eric/luc3d-bench/liezl_env/bin/python figs/fig4_main_check.py --workers 24 > figs/out/md56_logs/F4main.log 2>&1
echo "[$(date +%T)] F4main exit=$? $(tail -n 1 figs/out/md56_logs/F4main.log)"
