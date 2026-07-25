#!/bin/sh
# Submit ONE experiment to a resident Ray head (started by serve_head.sh) and exit. The
# fleet evicts its replicas + kills its own scheduler/trace actors on shutdown but never
# stops the head or worker raylets (master attaches with ray.init(address=...), _owns_ray
# stays False), so the next serve_submit against the same RAY_PORT reuses the live cluster.
# Run in the dev container. One cluster runs one experiment at a time (each run evict_all's
# the shared GPU pool) — chain runs with `;` for a serial stream, use a second head/RAY_PORT
# for parallelism.
#
# Env interface is identical to serve_master.sh: RUN_ID / SCHED_CONFIG / EXPERIMENT_CONFIG /
# WORKFLOW_FILES / OUTPUT_DIR / MIN_GPUS / GPU_MEM / PREDICTIONS / PRIORITY_WEIGHT /
# VLLM_PYTHON. RAY_PORT selects which resident head to attach (default 6667).
set -e

umask 000

export PROJECT_DIR=/home/wangjh/gnn_predict
export VENV_DIR="$PROJECT_DIR/.venv"
export RAY_PORT="${RAY_PORT:-6667}"
export RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
export VLLM_PYTHON="${VLLM_PYTHON:-/home/wangjh/gnn_predict/envs/vllm-v100/.venv/bin/python}"
export MIN_GPUS="${MIN_GPUS:-2}"
export GPU_MEM="${GPU_MEM:-v100=32768,a100=81920}"
export PREDICTIONS="${PREDICTIONS:-/home/wangjh/gnn_predict/cache/profile_v2/predictions.yaml}"
export SCHED_CONFIG="${SCHED_CONFIG:-config/workflow/serve/scheduler_cache.yaml}"
export PRIORITY_WEIGHT="${PRIORITY_WEIGHT:-}"
export EXPERIMENT_CONFIG="${EXPERIMENT_CONFIG:-config/workflow/serve/replay_4w.yaml}"
export WORKFLOW_FILES="${WORKFLOW_FILES:-config/workflow/serve/qmsum1.yaml,config/workflow/serve/mbpp1.yaml,config/workflow/serve/qmsum2.yaml,config/workflow/serve/mbpp2.yaml}"
export OUTPUT_DIR="${OUTPUT_DIR:-output/serve_4workflow}"
export PYTHONPATH="$PROJECT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"

PRIORITY_ARG=""
if [ -n "$PRIORITY_WEIGHT" ]; then
  PRIORITY_ARG="--priority-weight $PRIORITY_WEIGHT"
fi

cd "$PROJECT_DIR"
. "$VENV_DIR/bin/activate"

exec python -m workflow.master \
  --workflow-files "$WORKFLOW_FILES" \
  --functions experiment.workflow.scenario_functions:build_registry \
  --experiment experiment.workflow.experiments.dataset_replay:run \
  --experiment-config "$EXPERIMENT_CONFIG" \
  --scheduler-config "$SCHED_CONFIG" \
  --vllm-python "$VLLM_PYTHON" \
  --predictions "$PREDICTIONS" \
  --gpu-mem "$GPU_MEM" \
  --min-gpus "$MIN_GPUS" \
  --output-dir "$OUTPUT_DIR" \
  --run-id "$RUN_ID" \
  --ray-address "127.0.0.1:$RAY_PORT" \
  $PRIORITY_ARG
