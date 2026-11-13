#!/usr/bin/env bash
# Creates sleap_nn_env/ alongside lp3d_env/ and liezl_env/.
# Pinned to sleap-nn 0.2.0 (latest as of 2026-05-20).
set -euo pipefail
cd "$(dirname "$0")/../.."
if [ -d sleap_nn_env ]; then
    echo "sleap_nn_env already exists; skipping create. Re-run after rm -rf to refresh."
    exit 0
fi
python3.11 -m venv sleap_nn_env
sleap_nn_env/bin/pip install --quiet --upgrade pip
sleap_nn_env/bin/pip install --quiet sleap-nn==0.2.0
sleap_nn_env/bin/python -c "import sleap_nn; print('sleap_nn', sleap_nn.__version__)"
