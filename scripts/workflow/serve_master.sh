#!/bin/sh
# Crater DDP master: start the Ray Head and run the experiment-driven master.
# Replace the placeholders below, expose no port (results are files under output/serve).
#
#   PROJECT_DIR / VENV_DIR  : identical across master and workers
#   VLLM_PYTHON             : python inside the vLLM serving venv
#   MIN_GPUS                : total accelerator cards across all worker Pods
#   GPU_MEM                 : gpu_kind=total_mem_mb pairs for accelerator discovery
#   PREDICTIONS             : resource contract cache covering every model on every gpu_kind
#
# Policy sweep: set SCHED_CONFIG=config/workflow/serve/scheduler_{fifo,history,cache,kairos}.yaml
# and a fresh RUN_ID per run; each writes output/serve/<RUN_ID>/workflow_trace.jsonl.
# Weighted fairness: set PRIORITY_WEIGHT=qmsum=2,mbpp=1 (empty = equal rotation).
# Topology: default = 4wf (qmsum1/2+mbpp1/2 -> output/serve_4workflow). For 2wf set
# WORKFLOW_FILES=config/workflow/serve/qmsum1.yaml,config/workflow/serve/mbpp1.yaml + OUTPUT_DIR.
# Arrival: EXPERIMENT_CONFIG picks burst (replay_{2w,4w}.yaml) or poisson (replay_*_poisson_r0*.yaml).
set -e

export PROJECT_DIR=/home/wangjh/gnn_predict
export VENV_DIR="$PROJECT_DIR/.venv"
export RAY_PORT="${RAY_PORT:-6667}"
export RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
export VLLM_PYTHON="${VLLM_PYTHON:-/home/wangjh/gnn_predict/envs/vllm-v100/.venv/bin/python}"
export MIN_GPUS="${MIN_GPUS:-2}"
export GPU_MEM="${GPU_MEM:-v100=32768,a100=81920}"
export PREDICTIONS="${PREDICTIONS:-/home/wangjh/gnn_predict/cache/profile/predictions.yaml}"
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

ray start --head --port="$RAY_PORT" --disable-usage-stats

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
  --ray-address auto \
  $PRIORITY_ARG
