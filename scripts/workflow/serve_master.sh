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
# Policy comparison: change `policy` in config/workflow/serve/scheduler.yaml (fifo|history|cache)
# and resubmit with a fresh RUN_ID; each run writes output/serve/<RUN_ID>/workflow_trace.jsonl.
set -e

export PROJECT_DIR=/home/wangjh/gnn_predict
export VENV_DIR="$PROJECT_DIR/.venv"
export RAY_PORT="${RAY_PORT:-6667}"
export RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
export VLLM_PYTHON="${VLLM_PYTHON:-/home/wangjh/gnn_predict/envs/vllm-v100/.venv/bin/python}"
export MIN_GPUS="${MIN_GPUS:-2}"
export GPU_MEM="${GPU_MEM:-v100=32768,a100=81920}"
export PREDICTIONS="${PREDICTIONS:-output/serve/predictions.json}"

cd "$PROJECT_DIR"
. "$VENV_DIR/bin/activate"

ray start --head --port="$RAY_PORT" --disable-usage-stats

exec env PYTHONPATH=src python -m workflow.master \
  --workflow-files config/workflow/serve/qmsum.yaml,config/workflow/serve/mbpp.yaml \
  --functions experiment.workflow.scenario_functions:build_registry \
  --experiment experiment.workflow.experiments.dataset_replay:run \
  --experiment-config config/workflow/serve/replay.yaml \
  --scheduler-config config/workflow/serve/scheduler.yaml \
  --vllm-python "$VLLM_PYTHON" \
  --predictions "$PREDICTIONS" \
  --gpu-mem "$GPU_MEM" \
  --min-gpus "$MIN_GPUS" \
  --output-dir output/serve \
  --run-id "$RUN_ID" \
  --ray-address auto
