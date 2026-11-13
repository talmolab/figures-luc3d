#!/usr/bin/env bash
# After the 3c speed test: reassemble Fig 3b without its footer (3c is rebuilt by the test).
cd /root/vast/eric/sleap-3d-gui/scratch/repos/lucid
until grep -q "RETIME DONE" figs/out/bvis5_logs/run_retime3c.out; do sleep 30; done
FIG_TAG=b FIG_VARIANT=bvis4 figs/.venv/bin/python figs/assemble.py 3 | grep assembled
echo "[$(date +%T)] 3b reassembled without footer"
