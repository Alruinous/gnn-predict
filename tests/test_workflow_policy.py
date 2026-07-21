from __future__ import annotations

import math

from workflow.artifacts import (
    AcceleratorConfig,
    GpuKind,
    ResourceContract,
    ResourceContractCache,
    ResourceContractSource,
    ResourceEvidence,
)
from workflow.policy import (
    EvictionCandidate,
    select_eviction_victim,
    select_placement,
)
from workflow.schema import AgentNodeConfig
from workflow.types import (
    ModelReplicaState,
    WorkflowModelFeatureKey,
)


def agent_node(
    *,
    max_new_tokens: int = 512,
) -> AgentNodeConfig:
    return AgentNodeConfig.model_validate(
        {
            "name": "agent-a",
            "type": "agent",
            "model": {"name": "test-model"},
            "execution": {
                "model_path": "/models/test-model",
                "max_new_tokens": max_new_tokens,
                "dtype": "float16",
                "serving": {
                    "max_model_len": 4096,
                    "max_num_seqs": 1,
                    "max_num_batched_tokens": 4096,
                },
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
    *,
    peak_vram_mb_upper_bound: float | None = None,
) -> ResourceContract:
    return ResourceContract(
        key=WorkflowModelFeatureKey(
            model_name="test-model",
            phase="decode",
            gpu_name=gpu_kind,
            batch_size=1,
            sequence_length=sequence_length,
            decode_output_length=output_length,
        ),
        source=ResourceContractSource.SYNTHETIC_FIXTURE,
        predicted_load_sec=5.0,
        predicted_run_sec=run_sec,
        predicted_peak_vram_mb=peak_vram_mb,
        peak_vram_mb_upper_bound=(
            peak_vram_mb if peak_vram_mb_upper_bound is None else peak_vram_mb_upper_bound
        ),
        peak_vram_mb_evidence=ResourceEvidence(
            method="point_estimate_only", sample_count=1
        ),
    )


def prediction_cache() -> ResourceContractCache:
    entries = (
        entry("v100", 1024, 512, 9_000, 2.0),
        entry("v100", 2048, 128, 11_000, 3.0),
        entry("v100", 2048, 512, 15_000, 4.0),
        entry("v100", 2048, 1024, 17_000, 5.0),
        entry("a100", 2048, 128, 11_000, 2.5),
        entry("a100", 2048, 512, 15_000, 3.0),
        entry("a100", 2048, 1024, 25_000, 3.5),
    )
    return ResourceContractCache(version=1, entries=entries)


def test_placement_uses_the_smallest_bucket_covering_the_fixed_limit() -> None:
    decision = select_placement(
        node=agent_node(),
        input_tokens=1100,
        accelerators=[accelerator("v100", 16_000), accelerator("a100", 40_000)],
        predictions=prediction_cache(),
        oom_penalties={},
        eps_mem_mb=512,
    )

    assert decision.gpu_kind == "v100"
    assert decision.sequence_length == 2048
    assert decision.predicted_load_sec == 5.0
    assert decision.max_new_tokens == 512
    assert decision.prediction_key is not None
    assert decision.prediction_key.decode_output_length == 512
    assert decision.feasible is True


def test_placement_uses_a_larger_bucket_without_changing_the_fixed_limit() -> None:
    decision = select_placement(
        node=agent_node(max_new_tokens=600),
        input_tokens=1100,
        accelerators=[accelerator("a100", 40_000)],
        predictions=prediction_cache(),
        oom_penalties={},
        eps_mem_mb=512,
    )

    assert decision.max_new_tokens == 600
    assert decision.prediction_key is not None
    assert decision.prediction_key.decode_output_length == 1024
    assert decision.gpu_kind == "a100"


def test_placement_oom_penalty_never_downscales_the_fixed_limit() -> None:
    cache = prediction_cache()
    penalized_key = WorkflowModelFeatureKey(
        model_name="test-model",
        phase="decode",
        gpu_name="v100",
        batch_size=1,
        sequence_length=2048,
        decode_output_length=512,
    )

    decision = select_placement(
        node=agent_node(),
        input_tokens=1100,
        accelerators=[accelerator("v100", 16_000)],
        predictions=cache,
        oom_penalties={penalized_key: 1_000},
        eps_mem_mb=512,
    )

    assert decision.feasible is False
    assert decision.reason == "request_infeasible"


def test_placement_feasibility_uses_upper_bound_not_point_estimate() -> None:
    # ŷ=10_000 alone would fit a 10_600MB accelerator, but the calibrated
    # upper bound U=15_000 must be the hard OOM safety gate.
    cache = ResourceContractCache(
        version=1,
        entries=(
            entry(
                "v100", 2048, 512, 10_000, 4.0, peak_vram_mb_upper_bound=15_000
            ),
        ),
    )

    decision = select_placement(
        node=agent_node(),
        input_tokens=1100,
        accelerators=[accelerator("v100", 10_600)],
        predictions=cache,
        oom_penalties={},
        eps_mem_mb=512,
    )

    assert decision.feasible is False
    assert decision.reason == "request_infeasible"


def test_placement_feasibility_accepts_once_the_upper_bound_fits() -> None:
    cache = ResourceContractCache(
        version=1,
        entries=(
            entry(
                "v100", 2048, 512, 10_000, 4.0, peak_vram_mb_upper_bound=15_000
            ),
        ),
    )

    decision = select_placement(
        node=agent_node(),
        input_tokens=1100,
        accelerators=[accelerator("v100", 15_600)],
        predictions=cache,
        oom_penalties={},
        eps_mem_mb=512,
    )

    assert decision.feasible is True
    assert decision.effective_vram_mb == 15_512


def test_placement_does_not_clamp_an_uncovered_input() -> None:
    decision = select_placement(
        node=agent_node(),
        input_tokens=3000,
        accelerators=[accelerator("v100", 16_000), accelerator("a100", 40_000)],
        predictions=prediction_cache(),
        oom_penalties={},
        eps_mem_mb=512,
    )

    assert decision.feasible is False
    assert decision.reason == "input_bucket_missing"
    assert decision.max_new_tokens is None


def test_placement_rejects_a_request_outside_the_serving_context() -> None:
    decision = select_placement(
        node=agent_node(),
        input_tokens=4090,
        accelerators=[accelerator("v100", 16_000)],
        predictions=prediction_cache(),
        oom_penalties={},
        eps_mem_mb=512,
    )

    assert decision.feasible is False
    assert decision.reason == "request_infeasible"


def test_placement_uses_runtime_then_id_for_stable_ties() -> None:
    cache = ResourceContractCache(
        version=1,
        entries=(
            entry("v100", 2048, 1024, 12_000, 4.0),
            entry("a100", 2048, 1024, 12_000, 3.0),
        ),
    )
    decision = select_placement(
        node=agent_node(),
        input_tokens=1100,
        accelerators=[
            accelerator("a100", 40_000, hostname="z-node"),
            accelerator("a100", 40_000, hostname="a-node"),
        ],
        predictions=cache,
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
