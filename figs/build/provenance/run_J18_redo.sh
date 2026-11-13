#!/usr/bin/env bash
# J18 redo (2026-10-21): re-time ONLY the calls that overlapped a watcher sample over 0.5 core
# (dropped from calls.jsonl, list in J18_redo<N>_calls.json). Same script, frames, order rule;
# T_RESUME=1 keeps every other call. Watched end to end. Arg 1 = redo number.
set -uo pipefail
N=$1
cd /root/vast/eric/sleap-3d-gui/scratch/repos/lucid
L=figs/out/md56_logs
export PATH=/root/vast/eric/node26_env/bin:$PATH LUCID_ROOT=/root/vast/eric/lucid-main-md56 \
       FIG3_VARIANT=b3 SCORE_NOTAIL=1 T_TAG=_j18 T_RESUME=1 T_MAX_OTHER_CORES=0.5
export NUMBA_NUM_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 UV_THREADPOOL_SIZE=1
[ "$(git -C $LUCID_ROOT rev-parse HEAD)" = 2dc001fe527c4dbaad17fa171da132cfdfe010e6 ] || { echo "not 2dc001fe"; exit 1; }
( /root/vast/eric/aniposelib413_env/bin/python figs/fig3f_timing_clean.py > $L/J18_redo${N}_timing.log 2>&1 ) &
SP=$!
python3 figs/timing_watch.py watch --root $SP --out $L/J18_redo${N}_watch.jsonl --interval 5 &
WP=$!
wait $SP; rc=$?; wait $WP
echo "J18 redo $N exit $rc $(tail -1 $L/J18_redo${N}_watch.jsonl)" > $L/J18_redo${N}.done
