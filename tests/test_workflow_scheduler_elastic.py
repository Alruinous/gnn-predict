from __future__ import annotations

import pytest
from pydantic import ValidationError

from workflow.artifacts import (
    AcceleratorConfig,
    ResourceContract,
    ResourceContractCache,
    ResourceContractSource,
    ResourceEvidence,
    SchedulerConfig,
)
from workflow.policy import EvictionCandidate, select_eviction_victim
from workflow.replica import ReplicaLoadResult
from workflow.scheduler import (
    GrantInfo,
    LoadReplicaAction,
    SchedulerCore,
)
from workflow.schema import Workflow
from workflow.types import ModelReplicaState

from test_workflow_scheduler_resources import agent_report, finish_start

# A queue that outlasts a load is the whole precondition for scaling out, so these
# fixtures make decode dominate load rather than the other way round.
LOAD_SEC = 1.0
RUN_SEC = 20.0
V100_MEM_MB = 16_000


def _entry(
    model_name: str,
    batch_size: int,
    sequence_length: int,
    output_tokens: int,
    *,
    peak_vram_mb: float = 11_000.0,
) -> ResourceContract:
    return ResourceContract(
        key={
            "model_name": model_name,
            "phase": "decode",
            "gpu_name": "v100",
            "batch_size": batch_size,
            "sequence_length": sequence_length,
            "decode_output_length": output_tokens,
        },
        source=ResourceContractSource.SYNTHETIC_FIXTURE,
        predicted_load_sec=LOAD_SEC,
        predicted_run_sec=RUN_SEC,
        predicted_peak_vram_mb=peak_vram_mb,
        peak_vram_mb_upper_bound=peak_vram_mb,
        peak_vram_mb_evidence=ResourceEvidence(
            method="point_estimate_only", sample_count=1
        ),
    )


def _predictions(
    model_names: tuple[str, ...] = ("test-model",),
    *,
    max_num_seqs: int = 1,
    peak_vram_mb: float = 11_000.0,
) -> ResourceContractCache:
    entries = [
        _entry(model_name, batch_size, 2048, output_tokens, peak_vram_mb=peak_vram_mb)
        for model_name in model_names
        for batch_size in range(1, max_num_seqs + 1)
        for output_tokens in (128, 512, 1024)
    ]
    return ResourceContractCache(version=1, entries=tuple(entries))


def _agent_node(name: str, model_name: str, max_num_seqs: int) -> dict[str, object]:
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
                "max_num_seqs": max_num_seqs,
                "max_num_batched_tokens": 4096,
            },
        },
        "prompt_template": "{content}",
    }


def one_model_workflow(max_num_seqs: int = 1) -> Workflow:
    return Workflow.model_validate(
        {
            "workflow_name": "agent-workflow",
            "nodes": [_agent_node("agent", "test-model", max_num_seqs)],
            "edges": [],
        }
    )


def two_model_workflow() -> Workflow:
    return Workflow.model_validate(
        {
            "workflow_name": "two-model-workflow",
            "nodes": [
                {"name": "start", "type": "function", "function": "start"},
                _agent_node("left", "test-model", 1),
                _agent_node("right", "other-model", 1),
                {"name": "end", "type": "function", "function": "end"},
            ],
            "edges": [
                {"source": "start", "target": "left"},
                {"source": "start", "target": "right"},
                {"source": "left", "target": "end"},
                {"source": "right", "target": "end"},
            ],
        }
    )


def elastic_core(
    *,
    elastic: bool = True,
    v100_count: int = 2,
    workflow: Workflow | None = None,
    predictions: ResourceContractCache | None = None,
    max_num_seqs: int = 1,
    max_replicas_per_model: int = 2,
) -> SchedulerCore:
    accelerators = tuple(
        AcceleratorConfig(
            hostname=f"v100-node-{index}",
            gpu_kind="v100",
            local_index=0,
            total_mem_mb=V100_MEM_MB,
        )
        for index in range(v100_count)
    )
    return SchedulerCore(
        workflow or one_model_workflow(max_num_seqs),
        scheduler_config=SchedulerConfig(
            accelerators=accelerators,
            policy="cache",
            elastic_replicas=elastic,
            max_replicas_per_model=max_replicas_per_model,
            scale_out_margin_sec=0.0,
            scale_in_idle_sec=0.0,
        ),
        predictions=predictions or _predictions(max_num_seqs=max_num_seqs),
    )


def request_agent(
    core: SchedulerCore,
    session_id: str,
    *,
    workflow_name: str = "agent-workflow",
    node_id: str = "agent",
    now: float = 1.0,
) -> tuple[str, str]:
    if session_id not in core.sessions:
        core.register_session(session_id, workflow_name)
        if "start" in core.sessions[session_id].task_ids:
            finish_start(core, session_id)
    task_id = core.begin_node(session_id, node_id, [f"input-{session_id}-{node_id}"])
    acquire_id = core.request_acquire(task_id, input_tokens=1100, created_at=now)
    return task_id, acquire_id


def complete_load(core: SchedulerCore, action: LoadReplicaAction, now: float) -> None:
    core.complete_load(
        action.replica_id,
        ReplicaLoadResult(
            physical_gpu_id=action.accelerator.local_index,
            duration_sec=LOAD_SEC,
            idle_vram_mb=8_000,
            block_size=16,
            num_gpu_blocks=100,
            gpu_kv_tokens=1_600,
        ),
        backend_handle=f"backend-{action.replica_id}",
        now=now,
    )


def load_and_grant(
    core: SchedulerCore,
    acquire_id: str,
    *,
    now: float = 2.0,
) -> tuple[GrantInfo, LoadReplicaAction]:
    actions = core.tick_once(now=now)
    action = next(item for item in actions if isinstance(item, LoadReplicaAction))
    complete_load(core, action, now + LOAD_SEC)
    core.tick_once(now=now + LOAD_SEC)
    grant = core.poll_grant(acquire_id)
    assert grant is not None
    return grant, action


def test_elastic_replicas_require_the_cache_policy() -> None:
    for policy in ("fifo", "history", "kairos"):
        with pytest.raises(ValidationError, match="elastic_replicas requires"):
            SchedulerConfig(policy=policy, elastic_replicas=True)
    SchedulerConfig(policy="cache", elastic_replicas=True)


def test_scale_out_adds_a_replica_when_the_queue_outlasts_a_load() -> None:
    core = elastic_core()
    _, first = request_agent(core, "s1")
    _, action = load_and_grant(core, first)
    request_agent(core, "s2", now=4.0)

    scale_out = [
        item
        for item in core.tick_once(now=5.0)
        if isinstance(item, LoadReplicaAction) and item.reason == "scale_out"
    ]

    assert len(scale_out) == 1
    assert scale_out[0].replica_id != action.replica_id
    assert scale_out[0].group_size == 2
    assert scale_out[0].scale_out_gain_sec is not None
    assert scale_out[0].scale_out_gain_sec > 0


def test_one_replica_per_pair_holds_when_elastic_is_off() -> None:
    core = elastic_core(elastic=False)
    _, first = request_agent(core, "s1")
    load_and_grant(core, first)
    request_agent(core, "s2", now=4.0)

    assert core.tick_once(now=5.0) == []
    assert len(core.replicas) == 1


def test_scale_out_respects_the_replica_cap() -> None:
    core = elastic_core(v100_count=3, max_replicas_per_model=2)
    _, first = request_agent(core, "s1")
    load_and_grant(core, first)
    request_agent(core, "s2", now=4.0)
    second = core.tick_once(now=5.0)
    assert len(second) == 1
    complete_load(core, second[0], 6.0)
    core.tick_once(now=6.0)
    request_agent(core, "s3", now=7.0)

    assert core.tick_once(now=8.0) == []
    assert len(core.replicas) == 2


def test_blocked_scale_out_still_grants_ready_work_in_the_same_tick() -> None:
    """A rejected scale-out must not starve a request an existing replica can serve."""
    core = elastic_core(v100_count=1, max_num_seqs=2)
    _, first = request_agent(core, "s1")
    grant, _ = load_and_grant(core, first)
    _, second = request_agent(core, "s2", now=4.0)

    # The only card is taken, so no scale-out is possible; the resident replica
    # still has batch capacity and must absorb the request immediately.
    assert core.tick_once(now=5.0) == []
    second_grant = core.poll_grant(second)
    assert second_grant is not None
    assert second_grant.replica_id == grant.replica_id
    assert len(core.replicas) == 1


def test_routing_prefers_an_idle_sibling_over_another_scale_out() -> None:
    core = elastic_core(v100_count=3, max_replicas_per_model=3)
    _, first = request_agent(core, "s1")
    first_grant, first_action = load_and_grant(core, first)
    request_agent(core, "s2", now=4.0)
    scale_out = core.tick_once(now=5.0)
    assert len(scale_out) == 1
    sibling = scale_out[0]
    assert isinstance(sibling, LoadReplicaAction)
    complete_load(core, sibling, 6.0)
    core.tick_once(now=6.0)
    core.complete(
        first_grant.task_id,
        agent_report(core, first_grant),
        acquire_id=first,
    )

    _, third = request_agent(core, "s3", now=7.0)
    actions = core.tick_once(now=8.0)

    assert actions == []
    third_grant = core.poll_grant(third)
    assert third_grant is not None
    assert third_grant.replica_id in (first_action.replica_id, sibling.replica_id)
    assert len(core.replicas) == 2


def test_scale_out_spares_the_last_card_a_waiting_model_could_use() -> None:
    core = elastic_core(
        v100_count=2,
        workflow=two_model_workflow(),
        predictions=_predictions(("test-model", "other-model")),
    )
    _, left = request_agent(
        core, "s1", workflow_name="two-model-workflow", node_id="left"
    )
    load_and_grant(core, left)
    request_agent(core, "s2", workflow_name="two-model-workflow", node_id="left", now=4.0)
    # other-model has queued work and is not resident, so the free card is its only
    # possible home and a second test-model replica must not take it.
    request_agent(
        core, "s1", workflow_name="two-model-workflow", node_id="right", now=4.0
    )

    actions = core.tick_once(now=5.0)

    scale_out = [
        item
        for item in actions
        if isinstance(item, LoadReplicaAction) and item.reason == "scale_out"
    ]
    assert scale_out == []
    loads = [item for item in actions if isinstance(item, LoadReplicaAction)]
    assert len(loads) == 1
    assert loads[0].deployment.model_name == "other-model"


def test_drain_marks_a_spare_replica_for_a_starved_model() -> None:
    core = elastic_core(
        v100_count=2,
        workflow=two_model_workflow(),
        predictions=_predictions(("test-model", "other-model")),
    )
    _, left = request_agent(
        core, "s1", workflow_name="two-model-workflow", node_id="left"
    )
    load_and_grant(core, left)
    request_agent(core, "s2", workflow_name="two-model-workflow", node_id="left", now=4.0)
    scale_out = core.tick_once(now=5.0)
    assert len(scale_out) == 1
    complete_load(core, scale_out[0], 6.0)
    core.tick_once(now=6.0)
    assert all(
        replica.state == ModelReplicaState.BUSY for replica in core.replicas.values()
    )

    # Both cards now run test-model and neither is evictable, so the only way to
    # make room for other-model is to stop feeding one of the spares.
    request_agent(
        core, "s1", workflow_name="two-model-workflow", node_id="right", now=7.0
    )
    core.tick_once(now=8.0)

    drained = [
        replica
        for replica in core.replicas.values()
        if replica.drain_requested_at is not None
    ]
    assert len(drained) == 1
    events = core.take_drain_events()
    assert len(events) == 1
    assert events[0].replica_id == drained[0].replica_id


def test_redundant_candidate_is_evicted_before_a_sole_residency() -> None:
    redundant = EvictionCandidate(
        replica_id="r0002",
        state=ModelReplicaState.IDLE,
        idle_since=100.0,
        reuse_distance_sec=0.0,
        reload_cost_sec=0.0,
        redundant=True,
    )
    sole = EvictionCandidate(
        replica_id="r0001",
        state=ModelReplicaState.IDLE,
        idle_since=1.0,
        reuse_distance_sec=1_000.0,
        reload_cost_sec=0.0,
    )

    assert select_eviction_victim([sole, redundant]) is redundant
    # Without a redundant member the ordering is the historical distance ranking.
    assert select_eviction_victim([sole]) is sole


def test_a_shallow_queue_does_not_scale_out() -> None:
    """Waiting one service round must beat paying a cold start."""
    # RUN_SEC=20 against LOAD_SEC=1 makes the queue win easily, so raise the margin
    # above a single round: one waiting request is then not worth a replica.
    core = elastic_core()
    core.scheduler_config = core.scheduler_config.model_copy(
        update={"scale_out_margin_sec": RUN_SEC * 2}
    )
    _, first = request_agent(core, "s1")
    load_and_grant(core, first)
    request_agent(core, "s2", now=4.0)

    assert core.tick_once(now=5.0) == []
    assert len(core.replicas) == 1


def test_scale_out_gain_grows_with_queue_depth() -> None:
    core = elastic_core(v100_count=3)
    _, first = request_agent(core, "s1")
    load_and_grant(core, first)

    gains = []
    for index in range(2, 6):
        _, acquire = request_agent(core, f"s{index}", now=4.0)
        pending = core.pending_acquires[acquire]
        decision = core._ready_decisions(5.0)[acquire]
        gains.append(core._scale_out_gain(pending, decision, 5.0))

    # Each extra queued request pushes this one a further service round out, so the
    # case for another replica strengthens monotonically — which is also why the
    # marginal gain shrinks once a second replica doubles the slots.
    assert gains == sorted(gains)
    assert gains[-1] > gains[0]
