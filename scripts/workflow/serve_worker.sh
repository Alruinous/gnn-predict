#!/bin/sh
# Crater DDP worker: join the master's Ray Head and block. One accelerator card per worker Pod.
# MASTER_ADDR is injected by Volcano; RAY_PORT must match serve_master.sh.
set -e

sleep 5

export PROJECT_DIR=/home/wangjh/gnn_predict
export VENV_DIR="$PROJECT_DIR/.venv"
export RAY_PORT="${RAY_PORT:-6667}"
export MASTER_ADDR="${MASTER_ADDR:-localhost}"

cd "$PROJECT_DIR"
. "$VENV_DIR/bin/activate"

exec ray start --address="${MASTER_ADDR}:${RAY_PORT}" --disable-usage-stats --block
