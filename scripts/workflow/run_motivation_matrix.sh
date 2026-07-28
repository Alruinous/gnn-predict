#!/bin/sh
# Run the model-residency motivation matrix serially on one resident Ray head.
#
# Must run from the main checkout: the resident cluster's _SchedulerActor and NodeWorker
# actors carry no runtime_env, so they always import /home/wangjh/gnn_predict/src. A
# master started from a worktree ships a SchedulerConfig whose field set does not match
# the actors' class and dies mid-tick.
#
# Launch detached (setsid nohup) so the run outlives the shell that started it.
set -e

PROJECT_DIR="${PROJECT_DIR:-/home/wangjh/gnn_predict}"
RAY_PORT="${RAY_PORT:-6663}"
CFG="config/workflow/serve/motivation_20260727"
OUTPUT_DIR="${OUTPUT_DIR:-output/motivation_20260727_heterogeneity_residency}"
EXPERIMENT_CONFIG="${EXPERIMENT_CONFIG:-$CFG/replay_main.yaml}"
LOG_DIR="${LOG_DIR:-$OUTPUT_DIR/logs}"

cd "$PROJECT_DIR"
. .venv/bin/activate
mkdir -p "$LOG_DIR"

run_arm() {
  run_id="$1"
  heterogeneity="$2"
  scheduler="$3"
  echo "=== $run_id (H=$heterogeneity, $scheduler) $(date -Iseconds) ==="
  PYTHONPATH=src PYTHONHASHSEED=0 python -m workflow.master \
    --workflow-files "$CFG/h${heterogeneity}_moa_gsm8k.yaml,$CFG/h${heterogeneity}_repair_mbpp.yaml,$CFG/h${heterogeneity}_mapreduce_qmsum.yaml" \
    --functions experiment.workflow.motivation_workflows:build_registry \
    --experiment experiment.workflow.experiments.dataset_replay:run \
    --experiment-config "$EXPERIMENT_CONFIG" \
    --scheduler-config "$CFG/scheduler_${scheduler}.yaml" \
    --vllm-python "$PROJECT_DIR/envs/vllm-v100/.venv/bin/python" \
    --predictions cache/profile_v2/predictions.yaml \
    --gpu-mem v100=32768,a100=81920 \
    --min-gpus 3 \
    --output-dir "$OUTPUT_DIR" \
    --run-id "$run_id" \
    --ray-address "127.0.0.1:$RAY_PORT" \
    > "$LOG_DIR/$run_id.log" 2>&1
  echo "=== $run_id done $(date -Iseconds) ==="
}

run_arm h5_parrot 5 parrot
run_arm h5_kairos 5 kairos
run_arm h1_parrot 1 parrot
echo "=== matrix complete $(date -Iseconds) ==="
