#!/bin/sh
# Run one evaluation configuration against a resident Ray head and exit.
#
# Must run from the main checkout: the resident cluster's _SchedulerActor and NodeWorker
# actors carry no runtime_env, so they always import /home/wangjh/gnn_predict/src. A
# master started from a worktree ships a SchedulerConfig whose field set does not match
# the actors' class and dies mid-tick.
#
# Usage:
#   RAY_PORT=6661 RUN_ID=parrot_r1 SCHEDULER=parrot PREDICTIONS=cache/profile/predictions.yaml \
#     sh scripts/workflow/run_eval_matrix.sh
#   RAY_PORT=6662 RUN_ID=sys_gnn_v2_r1 SCHEDULER=sagepilot \
#     PREDICTIONS=cache/gnn_v2/predictions.yaml FUSE_NODES=1 \
#     sh scripts/workflow/run_eval_matrix.sh
set -e

PROJECT_DIR="${PROJECT_DIR:-/home/wangjh/gnn_predict}"
RAY_PORT="${RAY_PORT:-6661}"
CFG="config/workflow/serve/motivation_20260727"
OUTPUT_DIR="${OUTPUT_DIR:-output/eval_20260727}"
EXPERIMENT_CONFIG="${EXPERIMENT_CONFIG:-$CFG/replay_eval.yaml}"
SCHEDULER="${SCHEDULER:-sagepilot}"
PREDICTIONS="${PREDICTIONS:-cache/gnn_v2/predictions.yaml}"
FUSE_NODES="${FUSE_NODES:-}"
RUN_ID="${RUN_ID:?RUN_ID is required}"
LOG_DIR="${LOG_DIR:-$OUTPUT_DIR/logs}"

cd "$PROJECT_DIR"
. .venv/bin/activate
mkdir -p "$LOG_DIR"

FUSE_ARG=""
if [ -n "$FUSE_NODES" ]; then
  FUSE_ARG="--fuse-nodes"
fi

echo "=== $RUN_ID (scheduler=$SCHEDULER, predictions=$PREDICTIONS, fuse=${FUSE_NODES:-off}) $(date -Iseconds) ==="
PYTHONPATH=src PYTHONHASHSEED=0 python -m workflow.master \
  --workflow-files "$CFG/h5_moa_gsm8k.yaml,$CFG/h5_repair_mbpp.yaml,$CFG/h5_chain_qmsum.yaml" \
  --functions experiment.workflow.motivation_workflows:build_registry \
  --experiment experiment.workflow.experiments.dataset_replay:run \
  --experiment-config "$EXPERIMENT_CONFIG" \
  --scheduler-config "$CFG/scheduler_${SCHEDULER}.yaml" \
  --vllm-python "$PROJECT_DIR/envs/vllm-v100/.venv/bin/python" \
  --predictions "$PREDICTIONS" \
  --gpu-mem v100=32768,a100=81920 \
  --min-gpus 3 \
  --output-dir "$OUTPUT_DIR" \
  --run-id "$RUN_ID" \
  --ray-address "127.0.0.1:$RAY_PORT" \
  $FUSE_ARG \
  > "$LOG_DIR/$RUN_ID.log" 2>&1
echo "=== $RUN_ID done $(date -Iseconds) ==="
