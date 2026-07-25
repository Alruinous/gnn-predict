#!/bin/sh
# Resident Ray worker: join a dev-container head and block. One accelerator card per Pod;
# stays up across many experiments. RAY_HEAD_ADDR must point at the dev-container head IP
# and RAY_PORT at that head's port. Precedence RAY_HEAD_ADDR > MASTER_ADDR > localhost:
# RAY_HEAD_ADDR sidesteps the Volcano-injected MASTER_ADDR (which targets the DDP job's own
# master role, not the resident head). Set it inline as a single prefix, e.g.
#   RAY_HEAD_ADDR=10.244.18.19 RAY_PORT=6661 sh scripts/workflow/serve_worker.sh
set -e

sleep 5

export PROJECT_DIR=/home/wangjh/gnn_predict
export VENV_DIR="$PROJECT_DIR/.venv"
export RAY_PORT="${RAY_PORT:-6667}"
export RAY_HEAD_ADDR="${RAY_HEAD_ADDR:-10.244.18.20}"

cd "$PROJECT_DIR"
. "$VENV_DIR/bin/activate"

exec ray start \
    --address="${RAY_HEAD_ADDR}:${RAY_PORT}" \
    --disable-usage-stats \
    --block
