from __future__ import annotations

import math

from workflow.artifacts import (
    AcceleratorConfig,
    GpuKind,
    PredictionCache,
    PredictionEntry,
)
from workflow.policy import (
    EvictionCandidate,
    select_eviction_victim,
    select_token_budget,
)
from workflow.schema import AgentNodeConfig
from workflow.types import (
    ModelReplicaState,
    TokenBudgetAction,
    WorkflowModelFeatureKey,
)


def agent_node(
    *,
    min_tokens: int = 128,
    default_tokens: int = 512,
    max_tokens: int = 1024,
) -> AgentNodeConfig:
    return AgentNodeConfig.model_validate(
        {
            "name": "agent-a",
            "type": "agent",
            "model": {"name": "test-model"},
            "execution": {"model_path": "/models/test-model", "dtype": "float16"},
            "token_budget": {
                "min_max_new_tokens": min_tokens,
                "default_max_new_tokens": default_tokens,
                "max_max_new_tokens": max_tokens,
            },
            "prompt_template": "{content}",
        }
    )


def accelerator(
    gpu_kind: GpuKind,
    total_mem_mb: int,
    *,
    hostname: str | None = None,
    local_index: int = 0,
) -> AcceleratorConfig:
    return AcceleratorConfig(
        hostname=hostname or f"{gpu_kind}-node",
        gpu_kind=gpu_kind,
        local_index=local_index,
        total_mem_mb=total_mem_mb,
    )


def entry(
    gpu_kind: GpuKind,
    sequence_length: int,
    output_length: int,
    peak_vram_mb: float,
    run_sec: float,
) -> PredictionEntry:
    return PredictionEntry(
        key=WorkflowModelFeatureKey(
            model_name="test-model",
            phase="decode",
            gpu_name=gpu_kind,
            batch_size=1,
            sequence_length=sequence_length,
            decode_output_length=output_length,
        ),
        predicted_run_sec=run_sec,
        predicted_peak_vram_mb=peak_vram_mb,
    )


def prediction_cache() -> PredictionCache:
    entries = (
        entry("v100", 1024, 512, 9_000, 2.0),
        entry("v100", 2048, 128, 11_000, 3.0),
        entry("v100", 2048, 512, 15_000, 4.0),
        entry("v100", 2048, 1024, 17_000, 5.0),
        entry("a100", 2048, 128, 11_000, 2.5),
        entry("a100", 2048, 512, 15_000, 3.0),
        entry("a100", 2048, 1024, 25_000, 3.5),
    )
    return PredictionCache(version=1, entries=entries)


def test_token_policy_chooses_largest_feasible_cached_budget() -> None:
    decision = select_token_budget(
        node=agent_node(),
        input_tokens=1100,
        accelerators=[accelerator("v100", 16_000), accelerator("a100", 40_000)],
        predictions=prediction_cache(),
        history={},
        oom_penalties={},
        eps_mem_mb=512,
    )

    assert decision.gpu_kind == "v100"
    assert decision.sequence_length == 2048
    assert decision.granted_max_new_tokens == 512
    assert decision.action == TokenBudgetAction.FIXED


def test_token_policy_prefers_quality_headroom_after_repeated_limit_hits() -> None:
    decision = select_token_budget(
        node=agent_node(),
        input_tokens=1100,
        accelerators=[accelerator("v100", 16_000), accelerator("a100", 40_000)],
        predictions=prediction_cache(),
        history={"v100": 0.2},
        oom_penalties={},
        eps_mem_mb=512,
    )

    assert decision.gpu_kind == "a100"
    assert decision.granted_max_new_tokens == 1024
    assert decision.action == TokenBudgetAction.UPSCALED


def test_token_policy_falls_back_to_largest_budget_when_small_gpu_hits_limit() -> None:
    cache = PredictionCache(
        version=1,
        entries=(
            entry("v100", 2048, 1024, 12_000, 4.0),
            entry("a100", 2048, 512, 12_000, 3.0),
        ),
    )

    decision = select_token_budget(
        node=agent_node(),
        input_tokens=1100,
        accelerators=[accelerator("v100", 16_000), accelerator("a100", 40_000)],
        predictions=cache,
        history={"v100": 0.2},
        oom_penalties={},
        eps_mem_mb=512,
    )

    assert decision.gpu_kind == "v100"
    assert decision.granted_max_new_tokens == 1024


def test_token_policy_applies_the_exact_bucket_oom_penalty() -> None:
    cache = prediction_cache()
    penalized_key = WorkflowModelFeatureKey(
        model_name="test-model",
        phase="decode",
        gpu_name="v100",
        batch_size=1,
        sequence_length=2048,
        decode_output_length=512,
    )

    decision = select_token_budget(
        node=agent_node(),
        input_tokens=1100,
        accelerators=[accelerator("v100", 16_000)],
        predictions=cache,
        history={},
        oom_penalties={penalized_key: 1_000},
        eps_mem_mb=512,
    )

    assert decision.granted_max_new_tokens == 128
    assert decision.effective_vram_mb == 11_512
    assert decision.action == TokenBudgetAction.DOWNSCALED


def test_token_policy_does_not_clamp_an_uncovered_input() -> None:
    decision = select_token_budget(
        node=agent_node(),
        input_tokens=8192,
        accelerators=[accelerator("v100", 16_000), accelerator("a100", 40_000)],
        predictions=prediction_cache(),
        history={},
        oom_penalties={},
        eps_mem_mb=512,
    )

    assert decision.action == TokenBudgetAction.INFEASIBLE
    assert decision.reason == "input_bucket_missing"
    assert decision.granted_max_new_tokens is None


def test_token_policy_uses_cached_decode_buckets_inside_the_configured_range() -> None:
    decision = select_token_budget(
        node=agent_node(min_tokens=400, default_tokens=600, max_tokens=900),
        input_tokens=1100,
        accelerators=[accelerator("a100", 40_000)],
        predictions=prediction_cache(),
        history={},
        oom_penalties={},
        eps_mem_mb=512,
    )

    assert decision.granted_max_new_tokens == 512
    assert decision.action == TokenBudgetAction.DOWNSCALED


def test_token_policy_uses_runtime_then_id_for_stable_ties() -> None:
    cache = PredictionCache(
        version=1,
        entries=(
            entry("v100", 2048, 1024, 12_000, 4.0),
            entry("a100", 2048, 1024, 12_000, 3.0),
        ),
    )
    decision = select_token_budget(
        node=agent_node(),
        input_tokens=1100,
        accelerators=[
            accelerator("a100", 40_000, hostname="z-node"),
            accelerator("a100", 40_000, hostname="a-node"),
            accelerator("v100", 16_000),
        ],
        predictions=cache,
        history={"v100": 0.2},
        oom_penalties={},
        eps_mem_mb=512,
    )

    assert decision.accelerator_id == "a-node/a100:0"


def eviction_candidate(
    replica_id: str,
    state: ModelReplicaState,
    *,
    idle_since: float,
    reuse_distance_sec: float | None,
    reload_cost_sec: float | None,
) -> EvictionCandidate:
    return EvictionCandidate(
        replica_id=replica_id,
        state=state,
        idle_since=idle_since,
        reuse_distance_sec=reuse_distance_sec,
        reload_cost_sec=reload_cost_sec,
    )


def test_eviction_prefers_no_future_demand_and_excludes_busy_replicas() -> None:
    victim = select_eviction_victim(
        [
            eviction_candidate(
                "busy",
                ModelReplicaState.BUSY,
                idle_since=1,
                reuse_distance_sec=math.inf,
                reload_cost_sec=1,
            ),
            eviction_candidate(
                "later",
                ModelReplicaState.IDLE,
                idle_since=2,
                reuse_distance_sec=50,
                reload_cost_sec=10,
            ),
            eviction_candidate(
                "never",
                ModelReplicaState.SUSPECT,
                idle_since=3,
                reuse_distance_sec=math.inf,
                reload_cost_sec=100,
            ),
        ]
    )

    assert victim is not None
    assert victim.replica_id == "never"


def test_eviction_uses_a_computable_score_before_longest_idle_fallback() -> None:
    victim = select_eviction_victim(
        [
            eviction_candidate(
                "unknown-old",
                ModelReplicaState.IDLE,
                idle_since=1,
                reuse_distance_sec=None,
                reload_cost_sec=1,
            ),
            eviction_candidate(
                "known-new",
                ModelReplicaState.IDLE,
                idle_since=20,
                reuse_distance_sec=10,
                reload_cost_sec=3,
            ),
        ]
    )

    assert victim is not None
    assert victim.replica_id == "known-new"


def test_eviction_falls_back_to_longest_idle_only_when_all_scores_are_unknown() -> None:
    victim = select_eviction_victim(
        [
            eviction_candidate(
                "newer",
                ModelReplicaState.IDLE,
                idle_since=20,
                reuse_distance_sec=None,
                reload_cost_sec=2,
            ),
            eviction_candidate(
                "older",
                ModelReplicaState.SUSPECT,
                idle_since=5,
                reuse_distance_sec=30,
                reload_cost_sec=None,
            ),
        ]
    )

    assert victim is not None
    assert victim.replica_id == "older"


def test_eviction_returns_none_without_a_legal_victim() -> None:
    victim = select_eviction_victim(
        [
            eviction_candidate(
                "loading",
                ModelReplicaState.LOADING,
                idle_since=0,
                reuse_distance_sec=math.inf,
                reload_cost_sec=1,
            )
        ]
    )

    assert victim is None
