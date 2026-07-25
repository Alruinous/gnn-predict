#!/bin/sh
# Resident Ray head in the dev container: one cluster per RAY_PORT, GPU-less so the dev card
# is never recruited, with isolated temp-dir + ancillary ports so several heads coexist on
# this one box. Workers join from the job platform via serve_worker.sh; experiments attach
# via serve_submit.sh. `ray start --head` daemonizes and returns, leaving the head resident.
set -e

export PROJECT_DIR="${PROJECT_DIR:-/home/wangjh/gnn_predict}"
export VENV_DIR="${VENV_DIR:-$PROJECT_DIR/.venv}"
export RAY_PORT="${RAY_PORT:?set RAY_PORT, e.g. 6661}"
HEAD_IP="${HEAD_IP:-$(hostname -i | awk '{print $1}')}"
export PYTHONPATH="$PROJECT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"

cd "$PROJECT_DIR"
. "$VENV_DIR/bin/activate"

# timeout keeps the probe fast: `ray status` on a dead port blocks ~35s before failing.
if timeout 6 ray status --address "127.0.0.1:$RAY_PORT" >/dev/null 2>&1; then
  echo "Ray head already up on :$RAY_PORT — leaving it running"
  exit 0
fi

# Ancillary ports derive from RAY_PORT so co-located heads never collide; the dashboard
# agent + metrics ports run per node even with the web dashboard disabled. No Ray Client
# server (we attach via GCS address, not ray://, and ray[client] is not installed).
exec ray start --head \
  --node-ip-address="$HEAD_IP" \
  --port="$RAY_PORT" \
  --temp-dir="/tmp/ray_${RAY_PORT}" \
  --num-gpus=0 \
  --num-cpus=0 \
  --include-dashboard=false \
  --metrics-export-port="$((RAY_PORT + 300))" \
  --dashboard-agent-listen-port="$((RAY_PORT + 400))" \
  --dashboard-agent-grpc-port="$((RAY_PORT + 500))" \
  --disable-usage-stats \
  --block