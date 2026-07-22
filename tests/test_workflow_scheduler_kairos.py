from __future__ import annotations

from typing import Any

from workflow.artifacts import (
    AcceleratorConfig,
    ResourceContract,
    ResourceContractCache,
    ResourceContractSource,
    ResourceEvidence,
    SchedulerConfig,
    SchedulerPolicy,
)
from workflow.policy import critical_path_remaining_latency
from workflow.replica import ReplicaLoadResult
from workflow.scheduler import GrantInfo, LoadReplicaAction, SchedulerCore
from workflow.schema import Workflow, WorkflowGraph
from workflow.types import WorkflowModelFeatureKey

V100_MEM_MB = 16_000


def agent_node(name: str, model_name: str) -> dict[str, Any]:
    return {
        "name": name,
        "type": "agent",
        "model": {"name": model_name},
        "execution": {
            "model_path": f"/models/{model_name}",
            "max_new_tokens": 512,
            "dtype": "float16",
            "serving": {
                "max_model_len": 4096,
                "max_num_seqs": 1,
                "max_num_batched_tokens": 4096,
            },
        },
        "prompt_template": "{content}",
    }


def single_agent_workflow(workflow_name: str, node_name: str, model_name: str) -> Workflow:
    return Workflow.model_validate(
        {
            "workflow_name": workflow_name,
            "nodes": [agent_node(node_name, model_name)],
            "edges": [],
        }
    )


def chain_workflow(
    workflow_name: str, upstream: str, agent: str, model_name: str
) -> Workflow:
    return Workflow.model_validate(
        {
            "workflow_name": workflow_name,
            "nodes": [
                {"name": upstream, "type": "function", "function": "upstream"},
                agent_node(agent, model_name),
            ],
            "edges": [{"source": upstream, "target": agent}],
        }
    )


def prediction(model_name: str, *, run_sec: float) -> ResourceContract:
    return ResourceContract(
        key=WorkflowModelFeatureKey(
            model_name=model_name,
            phase="decode",
            gpu_name="v100",
            batch_size=1,
            sequence_length=2048,
            decode_output_length=512,
        ),
        source=ResourceContractSource.SYNTHETIC_FIXTURE,
        predicted_load_sec=5.0,
        predicted_run_sec=run_sec,
        predicted_peak_vram_mb=8_000,
        peak_vram_mb_upper_bound=8_000,
        peak_vram_mb_evidence=ResourceEvidence(
            method="point_estimate_only", sample_count=1
        ),
    )


def scheduler_core(
    workflow: Workflow,
    entries: tuple[ResourceContract, ...],
    *,
    policy: SchedulerPolicy,
    priority_weight: float = 1.0,
) -> SchedulerCore:
    accelerators = (
        AcceleratorConfig(
            hostname="node",
            gpu_kind="v100",
            local_index=0,
            total_mem_mb=V100_MEM_MB,
        ),
    )
    return SchedulerCore(
        workflow,
        scheduler_config=SchedulerConfig(policy=policy, accelerators=accelerators),
        predictions=ResourceContractCache(version=1, entries=entries),
        priority_weight=priority_weight,
    )


def graph(
    adjacency: dict[str, tuple[str, ...]],
    dependencies: dict[str, tuple[str, ...]],
    topological_order: tuple[str, ...],
    entry_node: str,
    terminal_node: str,
) -> WorkflowGraph:
    return WorkflowGraph(
        adjacency=adjacency,
        dependencies=dependencies,
        topological_order=topological_order,
        entry_node=entry_node,
        terminal_node=terminal_node,
    )


def load_replica(core: SchedulerCore, session_id: str, node_id: str) -> GrantInfo:
    task_id = core.begin_node(session_id, node_id, [f"input-{session_id}"])
    acquire_id = core.request_acquire(task_id, input_tokens=1100, created_at=1.0)
    actions = core.tick_once(now=2.0)
    load = next(action for action in actions if isinstance(action, LoadReplicaAction))
    core.complete_load(
        load.replica_id,
        ReplicaLoadResult(
            physical_gpu_id=load.accelerator.local_index,
            duration_sec=1.0,
            idle_vram_mb=4_000,
            block_size=16,
            num_gpu_blocks=100,
            gpu_kv_tokens=1_600,
        ),
        backend_handle=f"backend-{load.replica_id}",
        now=3.0,
    )
    core.tick_once(now=3.0)
    grant = core.poll_grant(acquire_id)
    assert grant is not None
    return grant


# --- critical_path_remaining_latency (pure DP) --------------------------------


def test_critical_path_remaining_latency_chain() -> None:
    chain = graph(
        {"a": ("b",), "b": ("c",), "c": ()},
        {"a": (), "b": ("a",), "c": ("b",)},
        ("a", "b", "c"),
        "a",
        "c",
    )

    remaining = critical_path_remaining_latency(chain, {"a": 1.0, "b": 2.0, "c": 3.0})

    assert remaining == {"a": 6.0, "b": 5.0, "c": 3.0}


def test_critical_path_remaining_latency_uses_longest_branch() -> None:
    diamond = graph(
        {"a": ("b", "c"), "b": ("d",), "c": ("d",), "d": ()},
        {"a": (), "b": ("a",), "c": ("a",), "d": ("b", "c")},
        ("a", "b", "c", "d"),
        "a",
        "d",
    )

    # Critical path is a -> c -> d = 1 + 5 + 1 = 7, longer than a -> b -> d = 4.
    remaining = critical_path_remaining_latency(
        diamond, {"a": 1.0, "b": 2.0, "c": 5.0, "d": 1.0}
    )

    assert remaining == {"a": 7.0, "b": 3.0, "c": 6.0, "d": 1.0}


def test_critical_path_remaining_latency_defaults_missing_costs_to_zero() -> None:
    chain = graph(
        {"a": ("b",), "b": ("c",), "c": ()},
        {"a": (), "b": ("a",), "c": ("b",)},
        ("a", "b", "c"),
        "a",
        "c",
    )

    remaining = critical_path_remaining_latency(chain, {"b": 2.0})

    assert remaining == {"a": 2.0, "b": 2.0, "c": 0.0}


# --- Kairos global remaining-latency ordering ---------------------------------


def test_kairos_orders_globally_by_remaining_latency_ignoring_weight() -> None:
    slow = single_agent_workflow("wf-slow", "slow", "model-slow")
    fast = single_agent_workflow("wf-fast", "fast", "model-fast")
    entries = (
        prediction("model-slow", run_sec=10.0),
        prediction("model-fast", run_sec=1.0),
    )
    core = scheduler_core(slow, entries, policy="kairos", priority_weight=100.0)
    core.register_workflow(fast, priority_weight=1.0)

    # Remaining latency comes from the frozen cache's point estimates.
    assert core._remaining_latency["wf-slow"]["slow"] == 10.0
    assert core._remaining_latency["wf-fast"]["fast"] == 1.0

    # The slow workflow arrives first (lower request_seq) and carries 100x weight.
    core.register_session("s-slow", "wf-slow")
    slow_task = core.begin_node("s-slow", "slow", ["in-slow"])
    slow_acquire = core.request_acquire(slow_task, input_tokens=1100, created_at=1.0)
    core.register_session("s-fast", "wf-fast")
    fast_task = core.begin_node("s-fast", "fast", ["in-fast"])
    fast_acquire = core.request_acquire(fast_task, input_tokens=1100, created_at=2.0)

    ordered = core._order_pending(tuple(core.pending_acquires.values()))

    # Shortest remaining latency wins despite later arrival and lower weight.
    assert [pending.acquire_id for pending in ordered] == [fast_acquire, slow_acquire]


def test_kairos_breaks_remaining_latency_ties_by_arrival_order() -> None:
    first = single_agent_workflow("wf-a", "a", "model-x")
    second = single_agent_workflow("wf-b", "b", "model-x")
    core = scheduler_core(first, (prediction("model-x", run_sec=3.0),), policy="kairos")
    core.register_workflow(second)

    core.register_session("s-a", "wf-a")
    task_a = core.begin_node("s-a", "a", ["in-a"])
    acquire_a = core.request_acquire(task_a, input_tokens=1100, created_at=1.0)
    core.register_session("s-b", "wf-b")
    task_b = core.begin_node("s-b", "b", ["in-b"])
    acquire_b = core.request_acquire(task_b, input_tokens=1100, created_at=2.0)

    ordered = core._order_pending(tuple(core.pending_acquires.values()))

    # Equal remaining latency (shared model) falls back to arrival order.
    assert [pending.acquire_id for pending in ordered] == [acquire_a, acquire_b]


def test_non_kairos_policy_keeps_weighted_interleave_ordering() -> None:
    slow = single_agent_workflow("wf-slow", "slow", "model-slow")
    fast = single_agent_workflow("wf-fast", "fast", "model-fast")
    entries = (
        prediction("model-slow", run_sec=10.0),
        prediction("model-fast", run_sec=1.0),
    )
    core = scheduler_core(slow, entries, policy="cache")
    core.register_workflow(fast)

    core.register_session("s-slow", "wf-slow")
    slow_task = core.begin_node("s-slow", "slow", ["in-slow"])
    core.request_acquire(slow_task, input_tokens=1100, created_at=1.0)
    core.register_session("s-fast", "wf-fast")
    fast_task = core.begin_node("s-fast", "fast", ["in-fast"])
    core.request_acquire(fast_task, input_tokens=1100, created_at=2.0)

    values = tuple(core.pending_acquires.values())

    # A non-Kairos policy must not apply global SRPT; it keeps weighted interleave.
    assert core._order_pending(values) == core._interleave_by_workflow(values)


# --- Kairos disables prefetch and predictive eviction (fifo-equivalent) --------


def test_kairos_disables_near_ready_prefetch() -> None:
    core = scheduler_core(
        chain_workflow("chain", "upstream", "near", "model-near"),
        (prediction("model-near", run_sec=4.0),),
        policy="kairos",
    )
    core.register_session("near-session", "chain")
    core.begin_node("near-session", "upstream", ["input-near"])
    core.record_running_upstream("near-session", "upstream", finish_at=20.0)

    assert core.near_ready_tasks() == ()
    assert core.tick_once(now=20.0) == []


def test_cache_still_prefetches_same_chain() -> None:
    core = scheduler_core(
        chain_workflow("chain", "upstream", "near", "model-near"),
        (prediction("model-near", run_sec=4.0),),
        policy="cache",
    )
    core.register_session("near-session", "chain")
    core.begin_node("near-session", "upstream", ["input-near"])
    core.record_running_upstream("near-session", "upstream", finish_at=20.0)

    near = core.near_ready_tasks()

    # Same scenario, but cache prefetches — proving Kairos actively suppresses it.
    assert len(near) == 1
    assert near[0].node_id == "near"


def test_kairos_eviction_uses_no_reload_cost() -> None:
    core = scheduler_core(
        single_agent_workflow("solo", "agent", "model-solo"),
        (prediction("model-solo", run_sec=4.0),),
        policy="kairos",
    )
    core.register_session("s1", "solo")
    grant = load_replica(core, "s1", "agent")

    # No reload cost -> select_eviction_victim falls back to longest-idle (LRU).
    assert core._reload_cost(core.replicas[grant.replica_id]) is None


def test_cache_eviction_uses_load_estimate_reload_cost() -> None:
    core = scheduler_core(
        single_agent_workflow("solo", "agent", "model-solo"),
        (prediction("model-solo", run_sec=4.0),),
        policy="cache",
    )
    core.register_session("s1", "solo")
    grant = load_replica(core, "s1", "agent")

    # Cache keeps a reuse-distance-aware reload cost from the load estimate.
    assert core._reload_cost(core.replicas[grant.replica_id]) == 5.0
