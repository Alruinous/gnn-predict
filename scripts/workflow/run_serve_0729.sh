#!/bin/sh
# Run ONE cell of the serve_0729 GPU-time-distribution matrix and exit.
#
# serve_0729 differs from serve_0728 in what "GBDT" and "Formula" mean. There they were
# SagePilot with its predictor swapped, i.e. predictor ablations. Here the four baseline
# arms use none of SagePilot's scheduling mechanisms and keep only their own prediction
# cache, so parrot / formula_baseline / gbdt_baseline differ from one another in exactly
# one factor: the cache. The *_with_sagepilot arms reproduce the serve_0728 definition on
# the same cards for direct comparison.
#
# Must run from the main checkout: the resident cluster's _SchedulerActor and NodeWorker
# actors carry no runtime_env, so they always import /home/wangjh/gnn_predict/src. A
# master started from a worktree ships a SchedulerConfig whose field set does not match
# the actors' class and dies mid-tick.
#
# One cluster runs one experiment at a time (each run evict_all's the shared GPU pool).
#
# Usage (one line = one run, no loops):
#   ARM=parrot ARRIVAL=burst RAY_PORT=6663 REPEAT=1 sh scripts/workflow/run_serve_0729.sh
#
# Output: output/serve_0729/<hw>__<arrival>__<arm>/r<REPEAT>/
set -e

PROJECT_DIR="${PROJECT_DIR:-/home/wangjh/gnn_predict}"
# The arrival and workflow definitions are read from serve_0728's directory rather than
# copied: run_manifest.json records each file's sha256, so reusing the originals proves
# the two datasets were offered the identical session sequence.
CFG="config/workflow/serve/serve_0729"
ARRIVAL_CFG="config/workflow/serve/serve_0728"
WORKFLOW_CFG="config/workflow/serve/motivation_20260727"
OUTPUT_ROOT="${OUTPUT_ROOT:-output/serve_0729}"

ARM="${ARM:?ARM is required}"
ARRIVAL="${ARRIVAL:?ARRIVAL is required}"
RAY_PORT="${RAY_PORT:?RAY_PORT is required}"
REPEAT="${REPEAT:?REPEAT is required}"

# An unknown factor value must fail here. Falling back to a default would produce a run
# that looks valid and only reveals itself as the wrong cell at report time.
case "$ARM" in
  # Four baselines. No prefetch, no cost-aware eviction, no load-yield ordering, no
  # elastic replicas, no anti-flap, no node fusion — see scheduler_baseline_fifo.yaml.
  parrot)           SCHED=scheduler_baseline_fifo; PREDICTIONS=cache/profile/predictions.yaml;    FUSE=  ;;
  kairos)           SCHED=scheduler_kairos;        PREDICTIONS=cache/profile/predictions.yaml;    FUSE=  ;;
  formula_baseline) SCHED=scheduler_baseline_fifo; PREDICTIONS=cache/static/predictions.yaml;     FUSE=  ;;
  gbdt_baseline)    SCHED=scheduler_baseline_fifo; PREDICTIONS=cache/tabular/predictions.yaml;    FUSE=  ;;
  # The full system.
  sagepilot)        SCHED=scheduler_sagepilot;     PREDICTIONS=cache/profile_v2/predictions.yaml; FUSE=1 ;;
  # The serve_0728 reading of the same two predictors: full SagePilot scheduler, cache
  # swapped. Run second, so the figure can separate "changed the predictor" from
  # "changed the whole scheduler" on one set of cards.
  formula_with_sagepilot) SCHED=scheduler_sagepilot; PREDICTIONS=cache/static/predictions.yaml;  FUSE=1 ;;
  gbdt_with_sagepilot)    SCHED=scheduler_sagepilot; PREDICTIONS=cache/tabular/predictions.yaml; FUSE=1 ;;
  *) echo "unknown ARM: $ARM" >&2; exit 2 ;;
esac

case "$ARRIVAL" in
  burst|poisson_r0125|poisson_r025|poisson_r050) ;;
  *) echo "unknown ARRIVAL: $ARRIVAL" >&2; exit 2 ;;
esac

case "$RAY_PORT" in
  6661|6662|6663) ;;
  *) echo "unknown RAY_PORT: $RAY_PORT" >&2; exit 2 ;;
esac

cd "$PROJECT_DIR"
. .venv/bin/activate

# Read the card mix from the head instead of assuming it from the port, so the
# directory's hardware segment always matches the accelerator list run_manifest.json
# records. Only `ray status -v` reports accelerator_type, and its cluster totals come
# before the first per-node block, so stop there rather than counting each node twice.
HW_SPEC=$(ray status --address "127.0.0.1:$RAY_PORT" -v | awk '
  /^Node:/ { exit }
  $2 == "GPU" { split($1, f, "/"); total = f[2] + 0 }
  /accelerator_type:A100/ { split($1, f, "/"); a = f[2] + 0 }
  /accelerator_type:V100/ { split($1, f, "/"); v = f[2] + 0 }
  END { if (a > 0 && v > 0 && a + v == total) printf "a%dv%d %d", a, v, a + v }')
# Qwen3-14B at 8192 context fits only on an A100, so a pool without one cannot run this
# workload at all. A total that A100 + V100 does not account for means a third card kind
# joined, which --gpu-mem has no entry for; neither is something to guess past.
if [ -z "$HW_SPEC" ]; then
  echo "head 127.0.0.1:$RAY_PORT has no readable A100 + V100 card mix" >&2
  exit 2
fi
HW="${HW_SPEC% *}"
MIN_GPUS="${HW_SPEC#* }"

CELL="${HW}__${ARRIVAL}__${ARM}"
OUTPUT_DIR="$OUTPUT_ROOT/$CELL"
RUN_ID="r$REPEAT"
LOG_DIR="${LOG_DIR:-$OUTPUT_ROOT/logs}"

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
  --experiment-config "$ARRIVAL_CFG/replay_${ARRIVAL}.yaml" \
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
