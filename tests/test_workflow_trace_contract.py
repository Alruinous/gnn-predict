from __future__ import annotations

import asyncio
import json
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

import pytest
from langchain.agents import AgentState
from langchain_core.messages import AIMessage

import workflow.scheduler as scheduler_module
from workflow.artifacts import (
    AcceleratorConfig,
    ResourceContract,
    ResourceContractCache,
    ResourceContractSource,
    ResourceEvidence,
    SchedulerConfig,
)
from workflow.controller import WorkflowController
from workflow.replica import ModelDeploymentConfig, ReplicaLoadResult
from workflow.scheduler import (
    AgentTaskRuntimeReport,
    EvictReplicaAction,
    GrantInfo,
    LoadReplicaAction,
    ModelReplicaRecord,
)
from workflow.schema import ServingConfig, Workflow
from workflow.types import ModelReplicaState, TraceEvent, WorkflowModelFeatureKey
from workflow.worker import message_text

TRACE_SECRET = "full-intermediate-agent-text-must-not-leak"


def state(text: str) -> AgentState:
    return AgentState(messages=[AIMessage(content=text)])


def fanout_fanin_workflow() -> Workflow:
    return Workflow.model_validate(
        {
            "workflow_name": "fanout-fanin-workflow",
            "nodes": [
                {
                    "name": "split",
                    "type": "function",
                    "function": "split",
                    "routing": "targeted",
                },
                {"name": "left", "type": "function", "function": "left"},
                {"name": "right", "type": "function", "function": "right"},
                {"name": "merge", "type": "function", "function": "merge"},
            ],
            "edges": [
                {"source": "split", "target": "left"},
                {"source": "split", "target": "right"},
                {"source": "left", "target": "merge"},
                {"source": "right", "target": "merge"},
            ],
        }
    )


def split(
    inputs: Mapping[str, object],
    states: Mapping[str, AgentState],
    parameters: Mapping[str, object],
) -> dict[str, AgentState]:
    return {"left": state(TRACE_SECRET), "right": state(TRACE_SECRET)}


def left(
    inputs: Mapping[str, object],
    states: Mapping[str, AgentState],
    parameters: Mapping[str, object],
) -> AgentState:
    return state(f"left:{message_text(states['split'])}")


def right(
    inputs: Mapping[str, object],
    states: Mapping[str, AgentState],
    parameters: Mapping[str, object],
) -> AgentState:
    return state(f"right:{message_text(states['split'])}")


def merge(
    inputs: Mapping[str, object],
    states: Mapping[str, AgentState],
    parameters: Mapping[str, object],
) -> AgentState:
    return state(f"{message_text(states['left'])}|{message_text(states['right'])}")


def read_trace(path: Path) -> list[TraceEvent]:
    return [
        TraceEvent.model_validate(json.loads(line))
        for line in path.read_text(encoding="utf-8").splitlines()
    ]


def trace_actor(
    workflow: Workflow,
    *,
    predictions: ResourceContractCache | None = None,
    replica_factory: object | None = None,
) -> Any:
    return scheduler_module._SchedulerActor(
        workflow=workflow,
        scheduler_config=scheduler_config(),
        predictions=predictions,
        trace_writer=MemoryTraceWriter(),
        replica_factory=replica_factory,
        run_id="trace-contract",
    )


class MemoryTraceWriter:
    def append_batch(self, events: list[TraceEvent]) -> int:
        return events[-1].event_seq if events else -1


class ImmediateReplica:
    def load(self) -> ReplicaLoadResult:
        return ReplicaLoadResult(
            physical_gpu_id=0,
            duration_sec=0.25,
            idle_vram_mb=900,
            block_size=16,
            num_gpu_blocks=100,
            gpu_kv_tokens=1_600,
        )

    def shutdown(self) -> None:
        return None


def immediate_replica_factory(action: LoadReplicaAction) -> ImmediateReplica:
    return ImmediateReplica()


def scheduler_config() -> SchedulerConfig:
    return SchedulerConfig(
        accelerators=(
            AcceleratorConfig(
                hostname="local",
                gpu_kind="v100",
                local_index=0,
                total_mem_mb=16_000,
            ),
        ),
        eps_mem_mb=100,
        eps_time_sec=0.5,
    )


def agent_workflow() -> Workflow:
    return Workflow.model_validate(
        {
            "workflow_name": "agent-workflow",
            "nodes": [
                {
                    "name": "agent",
                    "type": "agent",
                    "model": {"name": "test-model"},
                    "execution": {
                        "model_path": "/models/test-model",
                        "max_new_tokens": 8,
                        "dtype": "float16",
                        "serving": {
                            "max_model_len": 1024,
                            "max_num_seqs": 1,
                            "max_num_batched_tokens": 1024,
                        },
                    },
                    "prompt_template": "{content}",
                }
            ],
            "edges": [],
        }
    )


def prefetch_workflow() -> Workflow:
    return Workflow.model_validate(
        {
            "workflow_name": "prefetch-workflow",
            "nodes": [
                {
                    "name": "upstream",
                    "type": "function",
                    "function": "upstream",
                },
                {
                    "name": "agent",
                    "type": "agent",
                    "model": {"name": "test-model"},
                    "execution": {
                        "model_path": "/models/test-model",
                        "max_new_tokens": 8,
                        "dtype": "float16",
                        "serving": {
                            "max_model_len": 1024,
                            "max_num_seqs": 1,
                            "max_num_batched_tokens": 1024,
                        },
                    },
                    "prompt_template": "{content}",
                },
            ],
            "edges": [{"source": "upstream", "target": "agent"}],
        }
    )


def predictions() -> ResourceContractCache:
    return ResourceContractCache(
        version=7,
        entries=(
            ResourceContract(
                key=WorkflowModelFeatureKey(
                    model_name="test-model",
                    phase="decode",
                    gpu_name="v100",
                    batch_size=1,
                    sequence_length=128,
                    decode_output_length=8,
                ),
                source=ResourceContractSource.SYNTHETIC_FIXTURE,
                predicted_load_sec=5.0,
                predicted_run_sec=2.5,
                predicted_peak_vram_mb=1_024,
                peak_vram_mb_upper_bound=1_024,
                peak_vram_mb_evidence=ResourceEvidence(
                    method="point_estimate_only", sample_count=1
                ),
                predicted_power_watts=80,
                predictor_metadata={"checkpoint": "trace-test"},
            ),
        ),
    )


def events(actor: Any, event_type: str) -> list[TraceEvent]:
    return [event for event in actor._trace_buffer if event.event_type == event_type]


def prepare_grant(actor: Any) -> tuple[str, GrantInfo]:
    actor.core.register_session("agent-session", "agent-workflow")
    task_id = actor.core.begin_node("agent-session", "agent", ["agent-input"])
    acquire_id = actor.core.request_acquire(task_id, 64, created_at=10.0)
    actions = actor.core.tick_once(now=10.0)
    assert len(actions) == 1
    action = actions[0]
    assert isinstance(action, LoadReplicaAction)
    actor.core.complete_load(
        action.replica_id,
        ReplicaLoadResult(
            physical_gpu_id=0,
            duration_sec=0.25,
            idle_vram_mb=900,
            block_size=16,
            num_gpu_blocks=100,
            gpu_kv_tokens=1_600,
        ),
        backend_handle=object(),
        now=10.25,
    )
    assert actor.core.tick_once(now=10.25) == []
    grant = actor.core.poll_grant(acquire_id)
    assert grant is not None
    return task_id, grant


def make_near_ready(actor: Any, session_id: str) -> None:
    actor.core.register_session(session_id, "prefetch-workflow")
    actor.core.begin_node(session_id, "upstream", [f"input-{session_id}"])
    actor.core.record_running_upstream(
        session_id,
        "upstream",
        finish_at=20.0,
    )


def test_function_fanout_fanin_trace_preserves_item_provenance(
    ray_session: None,
    tmp_path: Path,
) -> None:
    controller = WorkflowController(
        fanout_fanin_workflow(),
        functions={"split": split, "left": left, "right": right, "merge": merge},
        scheduler_config=SchedulerConfig(),
        output_dir=tmp_path,
        run_id="data-plane-trace",
    )
    controller.start()
    try:
        controller.submit("session-1", {"content": TRACE_SECRET})
        controller.drain_and_stop(timeout_sec=30.0)
    finally:
        controller.stop_now(timeout_sec=0.1)

    trace = read_trace(tmp_path / "workflow_trace.jsonl")
    item_enqueued = [event for event in trace if event.event_type == "item_enqueued"]
    item_dequeued = [event for event in trace if event.event_type == "item_dequeued"]
    item_emitted = [event for event in trace if event.event_type == "item_emitted"]
    expected_edges = Counter(
        {
            (None, "split"): 1,
            ("split", "left"): 1,
            ("split", "right"): 1,
            ("left", "merge"): 1,
            ("right", "merge"): 1,
        }
    )

    assert (
        Counter((event.source_node, event.target_node) for event in item_enqueued)
        == expected_edges
    )
    assert (
        Counter((event.source_node, event.target_node) for event in item_dequeued)
        == expected_edges
    )
    assert Counter(
        (event.source_node, event.target_node) for event in item_emitted
    ) == expected_edges - Counter({(None, "split"): 1})
    assert all(
        event.item_id and event.session_id == "session-1" for event in item_enqueued
    )
    assert all(event.ts > 0 for event in item_enqueued + item_dequeued + item_emitted)

    waits = [
        event
        for event in trace
        if event.event_type == "fanin_wait" and event.node_id == "merge"
    ]
    ready = [
        event
        for event in trace
        if event.event_type == "fanin_ready" and event.node_id == "merge"
    ]
    assert len(waits) == 1
    assert len(ready) == 1
    assert sorted(cast(list[str], waits[0].payload["waiting_for_sources"])) in (
        ["left"],
        ["right"],
    )
    assert sorted(cast(list[str], ready[0].payload["source_nodes"])) == [
        "left",
        "right",
    ]
    assert len(cast(list[str], ready[0].payload["input_item_ids"])) == 2

    serialized = json.dumps(
        [event.model_dump(mode="json") for event in trace],
        ensure_ascii=False,
    )
    assert TRACE_SECRET not in serialized
    assert not (
        {"messages", "message", "agent_state", "output_text", "text"}
        & {key for event in trace for key in event.payload}
    )


def test_scheduler_trace_records_tick_and_load_decision_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    actor = trace_actor(agent_workflow(), predictions=predictions())
    actor.core.register_session("session-1", "agent-workflow")
    task_id = actor.core.begin_node("session-1", "agent", ["item-1"])
    actor.core.request_acquire(task_id, 64, created_at=40.0)
    monkeypatch.setattr(scheduler_module.time, "time", lambda: 50.0)

    actor._run_scheduling_pass()

    tick = events(actor, "scheduler_tick")[-1]
    decision = events(actor, "scheduler_decision")[-1]
    assert tick.ts == 50.0
    assert tick.payload["pending_acquire_count"] == 1
    assert tick.payload["action_count"] == 1
    assert decision.payload["action_type"] == "load_replica"
    assert decision.payload["reason"] == "ready_load"
    assert decision.replica_id is not None
    assert decision.accelerator_id == "local/v100:0"
    assert decision.gpu_kind == "v100"


def test_request_infeasible_trace_preserves_fixed_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    actor = trace_actor(agent_workflow(), predictions=predictions())
    actor.core.register_session("session-1", "agent-workflow")
    task_id = actor.core.begin_node("session-1", "agent", ["item-1"])
    actor.core.request_acquire(task_id, 1_000, created_at=40.0)
    monkeypatch.setattr(scheduler_module.time, "time", lambda: 50.0)

    actor._run_scheduling_pass()

    event = events(actor, "request_infeasible")[-1]
    assert event.ts == 50.0
    assert event.session_id == "session-1"
    assert event.task_id == task_id
    assert event.node_id == "agent"
    assert event.payload == {
        "reason": "input_bucket_missing",
        "input_tokens": 1_000,
        "max_new_tokens": 8,
    }


def test_agent_trace_uses_runtime_timestamp_and_prediction_payload() -> None:
    cache = predictions()
    actor = trace_actor(agent_workflow(), predictions=cache)
    task_id, grant = prepare_grant(actor)

    actor._record_new_grants()

    selected = events(actor, "placement_selected")[-1]
    prediction = cache.lookup(grant.prediction_key)
    assert selected.payload == {
        "input_tokens": 64,
        "max_new_tokens": 8,
        "admitted_batch_size": 1,
        "replica_inflight_at_grant": 1,
        "prediction_cache_version": 7,
        "prediction_key": prediction.key.model_dump(mode="json"),
        "predicted_load_sec": 5.0,
        "predicted_run_sec": 2.5,
        "predicted_peak_vram_mb": 1_024.0,
        "predicted_power_watts": 80.0,
        "predictor_metadata": {"checkpoint": "trace-test"},
    }

    report = AgentTaskRuntimeReport(
        acquire_id=grant.acquire_id,
        task_id=task_id,
        session_id="agent-session",
        node_id="agent",
        input_item_ids=["agent-input"],
        model_key=grant.model_key,
        accelerator_id=grant.accelerator_ids[0],
        gpu_kind=grant.gpu_kind,
        input_tokens=64,
        max_new_tokens=8,
        output_tokens=6,
        hit_token_limit=False,
        started_at=100.0,
        finished_at=105.0,
        duration_sec=5.0,
        status="success",
    )
    actor._apply_complete(task_id, report, grant.acquire_id)

    finished = events(actor, "task_execution_finished")[-1]
    assert finished.ts == report.finished_at
    assert finished.payload == {
        "status": "success",
        "duration_sec": 5.0,
        "started_at": 100.0,
        "finished_at": 105.0,
        "input_tokens": 64,
        "max_new_tokens": 8,
        "output_tokens": 6,
        "hit_token_limit": False,
        "finish_reason": None,
        "queue_time_sec": None,
        "time_to_first_token_sec": None,
        "replica_inflight_at_start": 1,
        "admitted_batch_size": 1,
        "engine_failed": False,
        "prediction_key": prediction.key.model_dump(mode="json"),
        "predicted_load_sec": 5.0,
        "predicted_run_sec": 2.5,
        "predicted_peak_vram_mb": 1_024.0,
        "predicted_power_watts": 80.0,
    }


def test_prefetch_trace_records_started_and_finished(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def run_scenario() -> list[TraceEvent]:
        actor = trace_actor(
            prefetch_workflow(),
            predictions=predictions(),
            replica_factory=immediate_replica_factory,
        )
        make_near_ready(actor, "near-session")
        monkeypatch.setattr(scheduler_module.time, "time", lambda: 14.5)
        actor._run_scheduling_pass()
        actor._dispatch_manual_actions()
        command = await asyncio.wait_for(actor._commands.get(), timeout=0.5)
        assert command.name == "load_succeeded"
        actor._apply_command(command)
        return list(actor._trace_buffer)

    trace = asyncio.run(run_scenario())
    started = next(event for event in trace if event.event_type == "prefetch_started")
    finished = next(event for event in trace if event.event_type == "prefetch_finished")
    loaded = next(event for event in trace if event.event_type == "model_load_finished")
    assert started.session_id == "near-session"
    assert started.node_id == "agent"
    assert started.payload == {
        "prefetch_at": 14.5,
        "expected_load_sec": 5.0,
    }
    assert finished.session_id == "near-session"
    assert finished.node_id == "agent"
    assert finished.replica_id == started.replica_id
    assert finished.payload == {
        "prefetch_at": 14.5,
        "duration_sec": 0.25,
    }
    assert loaded.payload["block_size"] == 16
    assert loaded.payload["num_gpu_blocks"] == 100
    assert loaded.payload["gpu_kv_tokens"] == 1_600


def test_prefetch_trace_records_resident_skip(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    actor = trace_actor(prefetch_workflow(), predictions=predictions())
    make_near_ready(actor, "resident-session")
    action = actor.core.tick_once(now=14.5)[0]
    assert isinstance(action, LoadReplicaAction)
    actor.core.complete_load(
        action.replica_id,
        ReplicaLoadResult(
            physical_gpu_id=0,
            duration_sec=0.25,
            idle_vram_mb=900,
            block_size=16,
            num_gpu_blocks=100,
            gpu_kv_tokens=1_600,
        ),
        backend_handle=object(),
        now=14.5,
    )
    make_near_ready(actor, "skipped-session")
    monkeypatch.setattr(scheduler_module.time, "time", lambda: 14.6)

    actor._run_scheduling_pass()

    skipped = next(
        event
        for event in events(actor, "prefetch_skipped")
        if event.session_id == "skipped-session"
    )
    assert skipped.node_id == "agent"
    assert skipped.model_key == action.deployment.model_key
    assert skipped.payload == {
        "reason": "replica_resident",
        "prefetch_at": 14.5,
    }


def test_prefetch_trace_distinguishes_loading_replica(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    actor = trace_actor(prefetch_workflow(), predictions=predictions())
    make_near_ready(actor, "loading-session")
    action = actor.core.tick_once(now=14.5)[0]
    assert isinstance(action, LoadReplicaAction)
    make_near_ready(actor, "skipped-session")
    monkeypatch.setattr(scheduler_module.time, "time", lambda: 14.6)

    actor._run_scheduling_pass()

    skipped = next(
        event
        for event in events(actor, "prefetch_skipped")
        if event.session_id == "skipped-session"
    )
    assert skipped.payload == {
        "reason": "already_loading",
        "prefetch_at": 14.5,
    }


def test_prefetch_trace_records_capacity_skip(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    actor = trace_actor(prefetch_workflow(), predictions=predictions())
    make_near_ready(actor, "near-session")
    deployment = ModelDeploymentConfig(
        model_name="blocking-model",
        model_path="/models/blocking-model",
        dtype="float16",
        serving=ServingConfig(
            max_model_len=1024,
            max_num_seqs=1,
            max_num_batched_tokens=1024,
        ),
    )
    accelerator = next(iter(actor.core.accelerators.values()))
    replica = ModelReplicaRecord(
        replica_id="blocking-replica",
        deployment=deployment,
        model_key=deployment.model_key,
        gpu_kind="v100",
        accelerator_ids=(accelerator.config.accelerator_id,),
        state=ModelReplicaState.LOADING,
        backend_handle=object(),
        created_at=1.0,
        expected_load_sec=5.0,
    )
    accelerator.replica_id = replica.replica_id
    actor.core.replicas[replica.replica_id] = replica
    actor.core._replica_pairs[(replica.model_key, replica.gpu_kind)] = (
        replica.replica_id
    )
    monkeypatch.setattr(scheduler_module.time, "time", lambda: 14.5)

    actor._run_scheduling_pass()

    skipped = events(actor, "prefetch_skipped")[-1]
    assert skipped.session_id == "near-session"
    assert skipped.payload == {
        "reason": "capacity_unavailable",
        "prefetch_at": 14.5,
    }


def test_prefetch_trace_records_ready_work_priority() -> None:
    actor = trace_actor(prefetch_workflow(), predictions=predictions())
    make_near_ready(actor, "near-session")
    near = actor.core.near_ready_tasks()
    candidate = near[0]
    ready_action = LoadReplicaAction(
        replica_id="ready-replica",
        deployment=candidate.deployment,
        accelerator=next(iter(actor.core.accelerators.values())).config,
        reason="ready_load",
        expected_load_sec=1.0,
    )

    actor._record_prefetch_skips(near, [ready_action], 14.5)

    skipped = events(actor, "prefetch_skipped")[-1]
    assert skipped.payload == {
        "reason": "ready_work_priority",
        "prefetch_at": 14.5,
    }


def test_prefetch_eviction_is_not_recorded_as_a_skip() -> None:
    actor = trace_actor(prefetch_workflow(), predictions=predictions())
    make_near_ready(actor, "near-session")
    near = actor.core.near_ready_tasks()
    candidate = near[0]
    action = EvictReplicaAction(
        replica_id="blocking-replica",
        accelerator_ids=("local/v100:0",),
        reason="near_ready_prefetch",
        session_id=candidate.session_id,
        node_id=candidate.node_id,
        prefetch_at=candidate.prefetch_at,
    )

    actor._record_prefetch_skips(near, [action], 14.5)

    assert events(actor, "prefetch_skipped") == []


@pytest.mark.parametrize(
    "payload",
    [
        {"text": TRACE_SECRET},
        {"agent_state": state(TRACE_SECRET)},
    ],
)
def test_trace_boundary_rejects_full_text_and_agent_state(
    payload: dict[str, object],
) -> None:
    actor = trace_actor(agent_workflow(), predictions=predictions())

    with pytest.raises(ValueError, match="trace payload"):
        actor._record_trace("unsafe", payload=payload)
