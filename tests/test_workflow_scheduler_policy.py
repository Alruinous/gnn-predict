from __future__ import annotations

import math
from typing import Any, Literal

import pytest

from workflow.artifacts import (
    AcceleratorConfig,
    PredictionCache,
    PredictionEntry,
    SchedulerConfig,
)
from workflow.replica import ReplicaLoadResult
from workflow.scheduler import (
    AgentTaskRuntimeReport,
    EvictReplicaAction,
    FunctionTaskRuntimeReport,
    GrantInfo,
    LoadReplicaAction,
    OutputReport,
    RuntimeHistoryRecord,
    SchedulerCore,
)
from workflow.schema import AgentNodeConfig, Workflow
from workflow.types import (
    ModelReplicaState,
    NodeTaskState,
    SessionState,
    WorkflowModelFeatureKey,
)


def agent_node(name: str, model_name: str) -> dict[str, Any]:
    return {
        "name": name,
        "type": "agent",
        "model": {"name": model_name},
        "execution": {
            "model_path": f"/models/{model_name}",
            "dtype": "float16",
            "serving": {
                "max_model_len": 4096,
                "max_num_seqs": 1,
                "max_num_batched_tokens": 4096,
            },
        },
        "token_budget": {
            "min_max_new_tokens": 512,
            "default_max_new_tokens": 512,
            "max_max_new_tokens": 512,
        },
        "prompt_template": "{content}",
    }


def near_workflow() -> Workflow:
    return Workflow.model_validate(
        {
            "nodes": [
                {"name": "upstream", "type": "function", "function": "upstream"},
                agent_node("near", "model-near"),
            ],
            "edges": [{"source": "upstream", "target": "near"}],
        }
    )


def eviction_workflow() -> Workflow:
    return Workflow.model_validate(
        {
            "nodes": [
                {"name": "start", "type": "function", "function": "start"},
                agent_node("resident", "model-resident"),
                agent_node("waiting", "model-waiting"),
                {"name": "end", "type": "function", "function": "end"},
            ],
            "edges": [
                {"source": "start", "target": "resident"},
                {"source": "start", "target": "waiting"},
                {"source": "resident", "target": "end"},
                {"source": "waiting", "target": "end"},
            ],
        }
    )


def multi_eviction_workflow() -> Workflow:
    return Workflow.model_validate(
        {
            "nodes": [
                {"name": "root", "type": "function", "function": "root"},
                {"name": "up_a", "type": "function", "function": "up_a"},
                {"name": "up_c", "type": "function", "function": "up_c"},
                agent_node("agent_a", "model-a"),
                agent_node("agent_b", "model-b"),
                agent_node("agent_c", "model-c"),
                {"name": "end", "type": "function", "function": "end"},
            ],
            "edges": [
                {"source": "root", "target": "up_a"},
                {"source": "root", "target": "up_c"},
                {"source": "root", "target": "agent_b"},
                {"source": "up_a", "target": "agent_a"},
                {"source": "up_c", "target": "agent_c"},
                {"source": "agent_a", "target": "end"},
                {"source": "agent_b", "target": "end"},
                {"source": "agent_c", "target": "end"},
            ],
        }
    )


def prediction(model_name: str, *, run_sec: float = 4.0) -> PredictionEntry:
    return PredictionEntry(
        key=WorkflowModelFeatureKey(
            model_name=model_name,
            phase="decode",
            gpu_name="v100",
            batch_size=1,
            sequence_length=2048,
            decode_output_length=512,
        ),
        predicted_load_sec=5.0,
        predicted_run_sec=run_sec,
        predicted_peak_vram_mb=8_000,
    )


def policy_core(
    workflow: Workflow,
    *,
    accelerator_count: int = 1,
) -> SchedulerCore:
    model_names = tuple(
        node.model.name for node in workflow.nodes if isinstance(node, AgentNodeConfig)
    )
    accelerators = tuple(
        AcceleratorConfig(
            hostname="node",
            gpu_kind="v100",
            local_index=index,
            total_mem_mb=16_000,
        )
        for index in range(accelerator_count)
    )
    return SchedulerCore(
        workflow,
        scheduler_config=SchedulerConfig(
            accelerators=accelerators,
            eps_time_sec=0.5,
            history_ema_alpha=0.5,
        ),
        predictions=PredictionCache(
            version=1,
            entries=tuple(prediction(model_name) for model_name in model_names),
        ),
    )


def finish_function(core: SchedulerCore, session_id: str, node_id: str) -> str:
    task_id = core.begin_node(session_id, node_id, [f"input-{session_id}-{node_id}"])
    task = core.tasks[task_id]
    core.complete(
        task_id,
        FunctionTaskRuntimeReport(
            task_id=task_id,
            session_id=session_id,
            node_id=node_id,
            input_item_ids=list(task.input_item_ids),
            started_at=1.0,
            finished_at=2.0,
            duration_sec=1.0,
            status="success",
        ),
    )
    core.finish_node(
        task_id,
        OutputReport(
            task_id=task_id,
            output_item_ids=[f"output-{task_id}"],
            persisted_terminal_result=False,
        ),
    )
    return task_id


def request_agent(
    core: SchedulerCore,
    session_id: str,
    node_id: str,
    *,
    created_at: float,
) -> tuple[str, str]:
    task_id = core.begin_node(session_id, node_id, [f"input-{session_id}-{node_id}"])
    acquire_id = core.request_acquire(task_id, input_tokens=1100, created_at=created_at)
    return task_id, acquire_id


def complete_load_and_grant(
    core: SchedulerCore,
    acquire_id: str,
    *,
    now: float,
) -> GrantInfo:
    actions = core.tick_once(now=now)
    load = next(action for action in actions if isinstance(action, LoadReplicaAction))
    core.complete_load(
        load.replica_id,
        ReplicaLoadResult(
            physical_gpu_id=load.accelerator.local_index,
            duration_sec=1.0,
            idle_vram_mb=4_000,
        ),
        backend_handle=f"backend-{load.replica_id}",
        now=now + 1,
    )
    core.tick_once(now=now + 1)
    grant = core.poll_grant(acquire_id)
    assert grant is not None
    return grant


def agent_report(
    core: SchedulerCore,
    grant: GrantInfo,
    *,
    output_tokens: int = 256,
    hit_token_limit: bool = False,
    duration_sec: float = 1.0,
    status: Literal["success", "failed", "oom"] = "success",
) -> AgentTaskRuntimeReport:
    task = core.tasks[grant.task_id]
    return AgentTaskRuntimeReport(
        acquire_id=grant.acquire_id,
        task_id=task.task_id,
        session_id=task.session_id,
        node_id=task.node_id,
        input_item_ids=list(task.input_item_ids),
        model_key=grant.model_key,
        accelerator_id=grant.accelerator_ids[0],
        gpu_kind=grant.gpu_kind,
        input_tokens=grant.input_tokens,
        granted_max_new_tokens=grant.granted_max_new_tokens,
        output_tokens=output_tokens,
        hit_token_limit=hit_token_limit,
        started_at=10.0,
        finished_at=10.0 + duration_sec,
        duration_sec=duration_sec,
        status=status,
        error_type=None if status == "success" else "RuntimeError",
    )


def history_report(index: int) -> AgentTaskRuntimeReport:
    return AgentTaskRuntimeReport(
        acquire_id=f"acquire-{index}",
        task_id=f"task-{index}",
        session_id=f"session-{index}",
        node_id="agent",
        input_item_ids=[f"input-{index}"],
        model_key="model-key",
        accelerator_id="node/v100:0",
        gpu_kind="v100",
        input_tokens=128,
        granted_max_new_tokens=512,
        output_tokens=index,
        hit_token_limit=index % 2 == 0,
        started_at=float(index),
        finished_at=float(index + 1),
        duration_sec=float(index),
        status="oom" if index % 10 == 0 else "success",
        error_type="OutOfMemoryError" if index % 10 == 0 else None,
    )


def seed_idle_replica(
    core: SchedulerCore,
    session_id: str,
    node_id: Literal["agent_a", "agent_b"],
    *,
    now: float,
) -> GrantInfo:
    core.register_session(session_id)
    finish_function(core, session_id, "root")
    if node_id == "agent_a":
        finish_function(core, session_id, "up_a")
    task_id, acquire_id = request_agent(
        core,
        session_id,
        node_id,
        created_at=now,
    )
    grant = complete_load_and_grant(core, acquire_id, now=now + 1)
    core.complete(task_id, agent_report(core, grant), acquire_id=acquire_id)
    core.sessions[session_id].state = SessionState.COMPLETED
    return grant


def test_runtime_history_is_keyed_bounded_and_aggregated() -> None:
    core = policy_core(near_workflow())
    for index in range(1, 71):
        core.record_runtime_report(history_report(index))

    history = core.history[("agent", "model-key", "v100")]
    assert isinstance(history, RuntimeHistoryRecord)
    assert len(history.samples) == 64
    assert history.output_tokens_p90 == 64
    assert history.hit_limit_rate == 0.5
    assert history.oom_count == 7
    assert math.isclose(history.duration_sec_ema, 69.0)


def test_near_ready_prefetch_uses_explicit_deadline() -> None:
    core = policy_core(near_workflow())
    core.register_session("near-session")
    core.begin_node("near-session", "upstream", ["input-near"])
    core.record_running_upstream("near-session", "upstream", finish_at=20.0)

    near = core.near_ready_tasks()

    assert len(near) == 1
    assert near[0].node_id == "near"
    assert near[0].upstream_eta == 20.0
    assert near[0].load_sec == 5.0
    assert near[0].prefetch_at == 14.5
    assert core.tick_once(now=14.49) == []
    actions = core.tick_once(now=14.5)
    assert len(actions) == 1
    assert isinstance(actions[0], LoadReplicaAction)
    assert actions[0].reason == "near_ready_prefetch"


def test_ready_work_outranks_near_ready_prefetch() -> None:
    core = policy_core(near_workflow())
    core.register_session("near-session")
    core.begin_node("near-session", "upstream", ["input-near"])
    core.record_running_upstream("near-session", "upstream", finish_at=20.0)
    core.register_session("ready-session")
    finish_function(core, "ready-session", "upstream")
    _, ready_acquire = request_agent(
        core,
        "ready-session",
        "near",
        created_at=3.0,
    )

    actions = core.tick_once(now=14.5)

    assert actions
    assert isinstance(actions[0], LoadReplicaAction)
    assert actions[0].reason == "ready_load"
    assert core.poll_grant(ready_acquire) is None


def test_ready_load_evicts_idle_victim_then_recomputes() -> None:
    core = policy_core(eviction_workflow())
    core.register_session("s1")
    finish_function(core, "s1", "start")
    resident_task, resident_acquire = request_agent(
        core,
        "s1",
        "resident",
        created_at=3.0,
    )
    resident_grant = complete_load_and_grant(core, resident_acquire, now=4.0)
    core.complete(
        resident_task,
        agent_report(core, resident_grant),
        acquire_id=resident_acquire,
    )
    _, waiting_acquire = request_agent(
        core,
        "s1",
        "waiting",
        created_at=7.0,
    )

    actions = core.tick_once(now=8.0)

    assert len(actions) == 1
    eviction = actions[0]
    assert isinstance(eviction, EvictReplicaAction)
    assert eviction.reason == "ready_load"
    assert eviction.reload_cost_sec == 5.0
    assert eviction.replica_id == resident_grant.replica_id
    assert core.replicas[eviction.replica_id].state == ModelReplicaState.EVICTING
    core.complete_eviction(eviction.replica_id)
    assert eviction.replica_id not in core.replicas
    load_actions = core.tick_once(now=9.0)
    assert len(load_actions) == 1
    assert isinstance(load_actions[0], LoadReplicaAction)
    assert core.poll_grant(waiting_acquire) is None


def test_suspect_cleanup_precedes_ready_scheduling() -> None:
    core = policy_core(eviction_workflow(), accelerator_count=2)
    core.register_session("s1")
    finish_function(core, "s1", "start")
    resident_task, resident_acquire = request_agent(
        core,
        "s1",
        "resident",
        created_at=3.0,
    )
    grant = complete_load_and_grant(core, resident_acquire, now=4.0)
    core.complete(
        resident_task,
        agent_report(core, grant, status="oom"),
        acquire_id=resident_acquire,
    )
    _, waiting_acquire = request_agent(
        core,
        "s1",
        "waiting",
        created_at=6.0,
    )

    actions = core.tick_once(now=7.0)

    assert len(actions) == 1
    assert isinstance(actions[0], EvictReplicaAction)
    assert actions[0].reason == "suspect_cleanup"
    assert core.replicas[grant.replica_id].state == ModelReplicaState.EVICTING
    assert core.poll_grant(waiting_acquire) is None


def test_eviction_prefers_no_future_demand_over_near_reuse() -> None:
    core = policy_core(multi_eviction_workflow(), accelerator_count=2)
    grant_a = seed_idle_replica(core, "seed-a", "agent_a", now=3.0)
    grant_b = seed_idle_replica(core, "seed-b", "agent_b", now=7.0)
    core.register_session("near-a")
    finish_function(core, "near-a", "root")
    core.begin_node("near-a", "up_a", ["near-a-input"])
    core.record_running_upstream("near-a", "up_a", finish_at=20.0)
    core.tasks[
        core.sessions["near-a"].task_ids["agent_b"]
    ].state = NodeTaskState.CANCELLED
    core.tasks[
        core.sessions["near-a"].task_ids["agent_c"]
    ].state = NodeTaskState.CANCELLED
    core.register_session("ready-c")
    finish_function(core, "ready-c", "root")
    finish_function(core, "ready-c", "up_c")
    _, _ = request_agent(core, "ready-c", "agent_c", created_at=10.0)
    core.tasks[
        core.sessions["ready-c"].task_ids["agent_a"]
    ].state = NodeTaskState.CANCELLED
    core.tasks[
        core.sessions["ready-c"].task_ids["agent_b"]
    ].state = NodeTaskState.CANCELLED

    eviction = core.tick_once(now=10.0)[0]

    assert isinstance(eviction, EvictReplicaAction)
    assert eviction.replica_id == grant_b.replica_id
    assert eviction.replica_id != grant_a.replica_id
    assert eviction.reuse_distance_sec is not None
    assert math.isinf(eviction.reuse_distance_sec)


def test_eviction_unknown_reuse_falls_back_to_longest_idle() -> None:
    core = policy_core(multi_eviction_workflow(), accelerator_count=2)
    grant_a = seed_idle_replica(core, "seed-a", "agent_a", now=3.0)
    grant_b = seed_idle_replica(core, "seed-b", "agent_b", now=7.0)
    replica_a = core.replicas[grant_a.replica_id]
    replica_b = core.replicas[grant_b.replica_id]
    replica_a.created_at = 100.0
    replica_a.idle_since = 0.0
    replica_b.idle_since = 5.0
    core.register_session("ready-c")
    finish_function(core, "ready-c", "root")
    finish_function(core, "ready-c", "up_c")
    request_agent(core, "ready-c", "agent_c", created_at=10.0)

    eviction = core.tick_once(now=10.0)[0]

    assert isinstance(eviction, EvictReplicaAction)
    assert eviction.replica_id == grant_a.replica_id
    assert eviction.reuse_distance_sec is None


def test_near_prefetch_does_not_evict_ready_pair() -> None:
    core = policy_core(multi_eviction_workflow(), accelerator_count=2)
    grant_a = seed_idle_replica(core, "seed-a", "agent_a", now=3.0)
    grant_b = seed_idle_replica(core, "seed-b", "agent_b", now=7.0)
    core.register_session("ready-a")
    finish_function(core, "ready-a", "root")
    finish_function(core, "ready-a", "up_a")
    _, ready_acquire = request_agent(
        core,
        "ready-a",
        "agent_a",
        created_at=10.0,
    )
    core.register_session("near-c")
    finish_function(core, "near-c", "root")
    core.begin_node("near-c", "up_c", ["near-c-input"])
    core.record_running_upstream("near-c", "up_c", finish_at=15.0)

    assert core.tick_once(now=10.0) == []
    assert core.poll_grant(ready_acquire) is not None
    eviction = core.tick_once(now=10.1)[0]

    assert isinstance(eviction, EvictReplicaAction)
    assert eviction.reason == "near_ready_prefetch"
    assert eviction.replica_id == grant_b.replica_id
    assert eviction.replica_id != grant_a.replica_id
    assert eviction.session_id == "near-c"
    assert eviction.node_id == "agent_c"
    assert eviction.prefetch_at is not None


def test_failed_eviction_is_explicit_and_keeps_ownership() -> None:
    core = policy_core(eviction_workflow())
    core.register_session("s1")
    finish_function(core, "s1", "start")
    task_id, acquire_id = request_agent(core, "s1", "resident", created_at=3.0)
    grant = complete_load_and_grant(core, acquire_id, now=4.0)
    core.complete(task_id, agent_report(core, grant), acquire_id=acquire_id)
    _, _ = request_agent(core, "s1", "waiting", created_at=7.0)
    eviction = core.tick_once(now=8.0)[0]
    assert isinstance(eviction, EvictReplicaAction)

    core.fail_eviction(eviction.replica_id, "KillError", "actor stayed alive")

    replica = core.replicas[eviction.replica_id]
    assert replica.state == ModelReplicaState.SUSPECT
    assert replica.eviction_error_type == "KillError"
    accelerator_id = replica.accelerator_ids[0]
    assert core.accelerators[accelerator_id].replica_id == replica.replica_id


def test_tick_never_evicts_busy_or_loading_replicas() -> None:
    core = policy_core(eviction_workflow())
    core.register_session("s1")
    finish_function(core, "s1", "start")
    _, _resident_acquire = request_agent(core, "s1", "resident", created_at=3.0)
    load = core.tick_once(now=4.0)[0]
    assert isinstance(load, LoadReplicaAction)
    _, _ = request_agent(core, "s1", "waiting", created_at=5.0)

    assert core.tick_once(now=6.0) == []
    assert core.replicas[load.replica_id].state == ModelReplicaState.LOADING
