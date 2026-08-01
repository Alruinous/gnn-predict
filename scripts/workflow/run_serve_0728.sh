#!/bin/sh
# Run ONE cell of the serve_0728 matrix against a resident Ray head and exit.
#
# Must run from the main checkout: the resident cluster's _SchedulerActor and NodeWorker
# actors carry no runtime_env, so they always import /home/wangjh/gnn_predict/src. A
# master started from a worktree ships a SchedulerConfig whose field set does not match
# the actors' class and dies mid-tick. After changing SchedulerConfig, restart the
# resident actors on every head before submitting.
#
# Factors: ARM x ARRIVAL x RAY_PORT (hardware) x REPEAT. One cluster runs one experiment
# at a time (each run evict_all's the shared GPU pool); parallelism comes from the three
# heads, which are the three hardware cells.
#
# Usage (one line = one run, no loops):
#   ARM=sagepilot ARRIVAL=burst RAY_PORT=6661 REPEAT=1 sh scripts/workflow/run_serve_0728.sh
#   ARM=noprefetch ARRIVAL=poisson_r025 RAY_PORT=6663 REPEAT=2 sh scripts/workflow/run_serve_0728.sh
#
# Output: output/serve_0728/<hw>__<arrival>__<arm>/r<REPEAT>/
set -e

PROJECT_DIR="${PROJECT_DIR:-/home/wangjh/gnn_predict}"
CFG="config/workflow/serve/serve_0728"
WORKFLOW_CFG="config/workflow/serve/motivation_20260727"
OUTPUT_ROOT="${OUTPUT_ROOT:-output/serve_0728}"

ARM="${ARM:?ARM is required}"
ARRIVAL="${ARRIVAL:?ARRIVAL is required}"
RAY_PORT="${RAY_PORT:?RAY_PORT is required}"
REPEAT="${REPEAT:?REPEAT is required}"

# An unknown factor value must fail here. Falling back to a default would produce a run
# that looks valid and only reveals itself as the wrong cell at report time.
case "$ARM" in
  sagepilot)  SCHED=scheduler_sagepilot;            PREDICTIONS=cache/profile_v2/predictions.yaml;  FUSE=1 ;;
  analytical) SCHED=scheduler_sagepilot;            PREDICTIONS=cache/static/predictions.yaml;  FUSE=1 ;;
  gbdt)       SCHED=scheduler_sagepilot;            PREDICTIONS=cache/tabular/predictions.yaml; FUSE=1 ;;
  # The three ablations share the reference cell's cache on purpose: each must differ
  # from sagepilot in exactly one mechanism, or its effect cannot be attributed.
  nofuse)     SCHED=scheduler_sagepilot;            PREDICTIONS=cache/profile_v2/predictions.yaml;  FUSE=  ;;
  noprefetch) SCHED=scheduler_sagepilot_noprefetch; PREDICTIONS=cache/profile_v2/predictions.yaml;  FUSE=1 ;;
  noxwf)      SCHED=scheduler_sagepilot_noxwf;      PREDICTIONS=cache/profile_v2/predictions.yaml;  FUSE=1 ;;
  # The two published orderings keep cache/profile (raw measurements): profile_v2 and
  # gnn_v2 carry this project's own load-time calibration, which is not theirs to get.
  parrot)     SCHED=scheduler_parrot;               PREDICTIONS=cache/profile/predictions.yaml; FUSE=  ;;
  kairos)     SCHED=scheduler_kairos;               PREDICTIONS=cache/profile/predictions.yaml; FUSE=  ;;
  *) echo "unknown ARM: $ARM" >&2; exit 2 ;;
esac

case "$ARRIVAL" in
  burst|poisson_r0125|poisson_r025|poisson_r050) ;;
  *) echo "unknown ARRIVAL: $ARRIVAL" >&2; exit 2 ;;
esac

case "$RAY_PORT" in
  6661) HW=a1v2; MIN_GPUS=3 ;;
  6662) HW=a1v3; MIN_GPUS=4 ;;
  6663) HW=a1v4; MIN_GPUS=5 ;;
  *) echo "unknown RAY_PORT: $RAY_PORT" >&2; exit 2 ;;
esac

CELL="${HW}__${ARRIVAL}__${ARM}"
OUTPUT_DIR="$OUTPUT_ROOT/$CELL"
RUN_ID="r$REPEAT"
LOG_DIR="${LOG_DIR:-$OUTPUT_ROOT/logs}"

cd "$PROJECT_DIR"
. .venv/bin/activate
mkdir -p "$LOG_DIR"

FUSE_ARG=""
if [ -n "$FUSE" ]; then
  FUSE_ARG="--fuse-nodes"
fi

echo "=== $CELL/$RUN_ID (sched=$SCHED, predictions=$PREDICTIONS, fuse=${FUSE:-off}, gpus=$MIN_GPUS) $(date -Iseconds) ==="
PYTHONPATH=src PYTHONHASHSEED=0 python -m workflow.master \
  --workflow-files "$WORKFLOW_CFG/h5_moa_gsm8k.yaml,$WORKFLOW_CFG/h5_repair_mbpp.yaml,$WORKFLOW_CFG/h5_chain_qmsum.yaml" \
  --functions experiment.workflow.motivation_workflows:build_registry \
  --experiment experiment.workflow.experiments.dataset_replay:run \
  --experiment-config "$CFG/replay_${ARRIVAL}.yaml" \
  --scheduler-config "$CFG/$SCHED.yaml" \
  --vllm-python "$PROJECT_DIR/envs/vllm-v100/.venv/bin/python" \
  --predictions "$PREDICTIONS" \
  --gpu-mem v100=32768,a100=81920 \
  --min-gpus "$MIN_GPUS" \
  --output-dir "$OUTPUT_DIR" \
  --run-id "$RUN_ID" \
  --ray-address "127.0.0.1:$RAY_PORT" \
  $FUSE_ARG \
  > "$LOG_DIR/${CELL}__${RUN_ID}.log" 2>&1
echo "=== $CELL/$RUN_ID done $(date -Iseconds) ==="
