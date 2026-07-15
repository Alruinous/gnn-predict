from __future__ import annotations

from typing import Literal

import pytest

import workflow.scheduler as scheduler_module
from workflow.artifacts import (
    AcceleratorConfig,
    PredictionCache,
    PredictionEntry,
    SchedulerConfig,
)
from workflow.replica import ReplicaLoadResult
from workflow.scheduler import (
    AcquireAlreadyGrantedError,
    AgentTaskRuntimeReport,
    EvictReplicaAction,
    FunctionTaskRuntimeReport,
    GrantInfo,
    LoadReplicaAction,
    OutputReport,
    SchedulerCore,
    SessionInactiveError,
)
from workflow.schema import Workflow
from workflow.types import (
    ModelReplicaState,
    NodeTaskState,
    SessionState,
    WorkflowModelFeatureKey,
)


def agent_workflow(
    *,
    max_attempts: int = 1,
    max_num_seqs: int = 1,
) -> Workflow:
    return Workflow.model_validate(
        {
            "nodes": [
                {
                    "name": "agent",
                    "type": "agent",
                    "model": {"name": "test-model"},
                    "execution": {
                        "model_path": "/models/test-model",
                        "max_new_tokens": 512,
                        "dtype": "float16",
                        "serving": {
                            "max_model_len": 4096,
                            "max_num_seqs": max_num_seqs,
                            "max_num_batched_tokens": 4096,
                        },
                    },
                    "prompt_template": "{content}",
                    "retry": {"max_attempts": max_attempts},
                }
            ],
            "edges": [],
        }
    )


def branched_agent_workflow() -> Workflow:
    agent = {
        "type": "agent",
        "model": {"name": "test-model"},
        "execution": {
            "model_path": "/models/test-model",
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
    return Workflow.model_validate(
        {
            "nodes": [
                {"name": "start", "type": "function", "function": "start"},
                {"name": "left", **agent},
                {"name": "right", **agent},
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


def prediction_entry(
    gpu_kind: Literal["v100", "a100"],
    batch_size: int,
    sequence_length: int,
    output_tokens: int,
    peak_vram_mb: float,
    run_sec: float,
) -> PredictionEntry:
    return PredictionEntry(
        key=WorkflowModelFeatureKey(
            model_name="test-model",
            phase="decode",
            gpu_name=gpu_kind,
            batch_size=batch_size,
            sequence_length=sequence_length,
            decode_output_length=output_tokens,
        ),
        predicted_load_sec=5.0,
        predicted_run_sec=run_sec,
        predicted_peak_vram_mb=peak_vram_mb,
    )


def resource_core(
    *,
    max_attempts: int = 1,
    workflow: Workflow | None = None,
    accelerators: tuple[AcceleratorConfig, ...] | None = None,
    max_num_seqs: int = 1,
) -> SchedulerCore:
    configured_accelerators = (
        accelerators
        if accelerators is not None
        else (
            AcceleratorConfig(
                hostname="v100-node",
                gpu_kind="v100",
                local_index=0,
                total_mem_mb=16_000,
            ),
            AcceleratorConfig(
                hostname="a100-node",
                gpu_kind="a100",
                local_index=0,
                total_mem_mb=40_000,
            ),
        )
    )
    entries: list[PredictionEntry] = []
    for batch_size in range(1, max_num_seqs + 1):
        for sequence_length in (1024, 2048):
            entries.extend(
                (
                    prediction_entry(
                        "v100", batch_size, sequence_length, 128, 11_000, 3.0
                    ),
                    prediction_entry(
                        "v100", batch_size, sequence_length, 512, 15_000, 4.0
                    ),
                    prediction_entry(
                        "v100", batch_size, sequence_length, 1024, 17_000, 5.0
                    ),
                    prediction_entry(
                        "a100", batch_size, sequence_length, 128, 11_000, 2.5
                    ),
                    prediction_entry(
                        "a100", batch_size, sequence_length, 512, 15_000, 3.0
                    ),
                    prediction_entry(
                        "a100", batch_size, sequence_length, 1024, 25_000, 3.5
                    ),
                )
            )
    predictions = PredictionCache(version=1, entries=tuple(entries))
    return SchedulerCore(
        workflow
        or agent_workflow(
            max_attempts=max_attempts,
            max_num_seqs=max_num_seqs,
        ),
        scheduler_config=SchedulerConfig(accelerators=configured_accelerators),
        predictions=predictions,
    )


def finish_start(core: SchedulerCore, session_id: str) -> None:
    task_id = core.begin_node(session_id, "start", [f"input-{session_id}"])
    task = core.tasks[task_id]
    core.complete(
        task_id,
        FunctionTaskRuntimeReport(
            task_id=task_id,
            session_id=session_id,
            node_id="start",
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
            output_item_ids=[f"start-{session_id}"],
            persisted_terminal_result=False,
        ),
    )


def request_agent(
    core: SchedulerCore,
    session_id: str,
    now: float = 1.0,
    input_tokens: int = 1100,
) -> tuple[str, str]:
    core.register_session(session_id)
    task_id = core.begin_node(session_id, "agent", [f"input-{session_id}"])
    acquire_id = core.request_acquire(
        task_id,
        input_tokens=input_tokens,
        created_at=now,
    )
    return task_id, acquire_id


def load_and_grant(
    core: SchedulerCore,
    acquire_id: str,
    *,
    now: float = 2.0,
) -> tuple[GrantInfo, LoadReplicaAction]:
    actions = core.tick_once(now=now)
    assert len(actions) == 1
    action = actions[0]
    assert isinstance(action, LoadReplicaAction)
    core.complete_load(
        action.replica_id,
        ReplicaLoadResult(
            physical_gpu_id=action.accelerator.local_index,
            duration_sec=1.0,
            idle_vram_mb=8_000,
            block_size=16,
            num_gpu_blocks=100,
            gpu_kv_tokens=1_600,
        ),
        backend_handle=f"backend-{action.replica_id}",
        now=now + 1,
    )
    assert core.tick_once(now=now + 1) == []
    grant = core.poll_grant(acquire_id)
    assert grant is not None
    return grant, action


def agent_report(
    core: SchedulerCore,
    grant: GrantInfo,
    *,
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
        max_new_tokens=grant.max_new_tokens,
        output_tokens=256 if status == "success" else 0,
        hit_token_limit=False,
        replica_inflight_at_start=grant.admitted_batch_size,
        started_at=4.0,
        finished_at=5.0,
        duration_sec=1.0,
        status=status,
        error_type=None if status == "success" else "OutOfMemoryError",
    )


def test_ready_acquire_reserves_load_then_grants_the_idle_replica() -> None:
    core = resource_core()
    task_id, acquire_id = request_agent(core, "s1")

    actions = core.tick_once(now=2.0)

    assert len(actions) == 1
    action = actions[0]
    assert isinstance(action, LoadReplicaAction)
    assert action.reason == "ready_load"
    assert action.accelerator.accelerator_id == "v100-node/v100:0"
    replica = core.replicas[action.replica_id]
    assert replica.state == ModelReplicaState.LOADING
    assert (
        core.accelerators[action.accelerator.accelerator_id].replica_id
        == action.replica_id
    )
    assert core.tasks[task_id].state == NodeTaskState.ACQUIRING
    assert core.poll_grant(acquire_id) is None

    core.complete_load(
        action.replica_id,
        ReplicaLoadResult(
            physical_gpu_id=0,
            duration_sec=1.0,
            idle_vram_mb=8_000,
            block_size=16,
            num_gpu_blocks=100,
            gpu_kv_tokens=1_600,
        ),
        backend_handle="backend",
        now=3.0,
    )
    assert core.tick_once(now=3.0) == []
    grant = core.poll_grant(acquire_id)
    assert grant is not None
    assert grant.replica_id == action.replica_id
    assert grant.max_new_tokens == 512
    assert grant.backend_handle == "backend"
    assert core.tasks[task_id].state == NodeTaskState.RUNNING
    assert core.replicas[action.replica_id].state == ModelReplicaState.BUSY


def test_ready_acquires_use_request_insertion_fifo(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    core = resource_core()
    core.register_session("s1")
    core.register_session("s2")
    first_task = core.begin_node("s1", "agent", ["input-s1"])
    second_task = core.begin_node("s2", "agent", ["input-s2"])
    acquire_ids = iter(("z-first", "a-second", "replica"))
    monkeypatch.setattr(scheduler_module, "uuid4", lambda: next(acquire_ids))
    first_acquire = core.request_acquire(first_task, 1100, 1.0)
    second_acquire = core.request_acquire(second_task, 1100, 1.0)

    assert list(core.pending_order) == [first_acquire, second_acquire]
    action = core.tick_once(now=2.0)[0]
    core.complete_load(
        action.replica_id,
        ReplicaLoadResult(
            physical_gpu_id=0,
            duration_sec=1.0,
            idle_vram_mb=8_000,
            block_size=16,
            num_gpu_blocks=100,
            gpu_kv_tokens=1_600,
        ),
        backend_handle="backend",
        now=3.0,
    )
    core.tick_once(now=3.0)

    assert core.poll_grant(first_acquire) is not None
    assert core.poll_grant(second_acquire) is None


def test_same_model_kind_has_one_replica_and_waits_while_busy() -> None:
    core = resource_core()
    _, first_acquire = request_agent(core, "s1")
    first_grant, action = load_and_grant(core, first_acquire)
    _, second_acquire = request_agent(core, "s2", now=4.0)

    assert core.tick_once(now=5.0) == []
    assert core.poll_grant(second_acquire) is None
    assert len(core.replicas) == 1

    core.complete(
        first_grant.task_id,
        agent_report(core, first_grant),
        acquire_id=first_acquire,
    )
    assert core.tick_once(now=6.0) == []
    second_grant = core.poll_grant(second_acquire)
    assert second_grant is not None
    assert second_grant.replica_id == action.replica_id


def test_busy_replica_grants_batch_envelopes_up_to_capacity() -> None:
    core = resource_core(max_num_seqs=3)
    requests = [
        request_agent(
            core,
            f"s{index}",
            now=float(index),
            input_tokens=500 if index == 1 else 1100,
        )
        for index in range(1, 4)
    ]

    action = core.tick_once(now=4.0)[0]
    assert isinstance(action, LoadReplicaAction)
    core.complete_load(
        action.replica_id,
        ReplicaLoadResult(
            physical_gpu_id=0,
            duration_sec=1.0,
            idle_vram_mb=8_000,
            block_size=16,
            num_gpu_blocks=100,
            gpu_kv_tokens=1_600,
        ),
        backend_handle="backend",
        now=5.0,
    )
    assert core.tick_once(now=5.0) == []

    grants = [core.poll_grant(acquire_id) for _, acquire_id in requests]
    assert all(grant is not None for grant in grants)
    typed_grants = [grant for grant in grants if grant is not None]
    assert [grant.admitted_batch_size for grant in typed_grants] == [1, 2, 3]
    assert [grant.prediction_key.batch_size for grant in typed_grants] == [1, 2, 3]
    assert typed_grants[0].prediction_key.sequence_length == 1024
    assert typed_grants[1].prediction_key.sequence_length == 2048
    assert len({grant.replica_id for grant in typed_grants}) == 1
    replica = core.replicas[action.replica_id]
    assert replica.state == ModelReplicaState.BUSY
    assert replica.active_acquire_ids == {acquire_id for _, acquire_id in requests}

    _, fourth_acquire = request_agent(core, "s4", now=6.0)
    assert core.tick_once(now=6.0) == []
    assert core.poll_grant(fourth_acquire) is None


def test_busy_replica_releases_out_of_order_and_refills_capacity() -> None:
    core = resource_core(max_num_seqs=3)
    requests = [request_agent(core, f"s{index}") for index in range(1, 4)]
    action = core.tick_once(now=2.0)[0]
    assert isinstance(action, LoadReplicaAction)
    core.complete_load(
        action.replica_id,
        ReplicaLoadResult(
            physical_gpu_id=0,
            duration_sec=1.0,
            idle_vram_mb=8_000,
            block_size=16,
            num_gpu_blocks=100,
            gpu_kv_tokens=1_600,
        ),
        backend_handle="backend",
        now=3.0,
    )
    core.tick_once(now=3.0)
    grants = [core.poll_grant(acquire_id) for _, acquire_id in requests]
    assert all(grant is not None for grant in grants)
    first, second, third = [grant for grant in grants if grant is not None]

    core.complete(
        second.task_id,
        agent_report(core, second),
        acquire_id=second.acquire_id,
    )
    replica = core.replicas[action.replica_id]
    assert replica.state == ModelReplicaState.BUSY
    assert replica.active_acquire_ids == {first.acquire_id, third.acquire_id}

    _, fourth_acquire = request_agent(core, "s4", now=4.0)
    core.tick_once(now=4.0)
    fourth = core.poll_grant(fourth_acquire)
    assert fourth is not None
    assert fourth.replica_id == action.replica_id
    assert fourth.admitted_batch_size == 3


def test_joint_batch_infeasibility_waits_until_replica_is_idle() -> None:
    core = resource_core(max_num_seqs=2)
    assert core.predictions is not None
    core.predictions = PredictionCache(
        version=core.predictions.version,
        entries=tuple(
            entry.model_copy(update={"predicted_peak_vram_mb": 20_000.0})
            if entry.key.gpu_name == "v100" and entry.key.batch_size == 2
            else entry
            for entry in core.predictions.entries
        ),
    )
    _, first_acquire = request_agent(core, "s1")
    first, action = load_and_grant(core, first_acquire)
    _, second_acquire = request_agent(core, "s2", now=4.0)

    assert core.tick_once(now=5.0) == []
    assert core.poll_grant(second_acquire) is None
    core.complete(
        first.task_id,
        agent_report(core, first),
        acquire_id=first.acquire_id,
    )
    assert core.tick_once(now=6.0) == []
    second = core.poll_grant(second_acquire)
    assert second is not None
    assert second.replica_id == action.replica_id
    assert second.admitted_batch_size == 1


def test_missing_higher_batch_prediction_waits_until_replica_is_idle() -> None:
    core = resource_core(max_num_seqs=2)
    assert core.predictions is not None
    core.predictions = PredictionCache(
        version=core.predictions.version,
        entries=tuple(
            entry for entry in core.predictions.entries if entry.key.batch_size == 1
        ),
    )
    _, first_acquire = request_agent(core, "s1")
    first, action = load_and_grant(core, first_acquire)
    _, second_acquire = request_agent(core, "s2", now=4.0)

    assert core.tick_once(now=5.0) == []
    assert core.poll_grant(second_acquire) is None
    core.complete(
        first.task_id,
        agent_report(core, first),
        acquire_id=first.acquire_id,
    )
    assert core.tick_once(now=6.0) == []
    second = core.poll_grant(second_acquire)
    assert second is not None
    assert second.replica_id == action.replica_id
    assert second.admitted_batch_size == 1


def test_suspect_replica_drains_remaining_leases_before_eviction() -> None:
    core = resource_core(max_num_seqs=2, max_attempts=2)
    requests = [request_agent(core, session_id) for session_id in ("s1", "s2")]
    action = core.tick_once(now=2.0)[0]
    assert isinstance(action, LoadReplicaAction)
    core.complete_load(
        action.replica_id,
        ReplicaLoadResult(
            physical_gpu_id=0,
            duration_sec=1.0,
            idle_vram_mb=8_000,
            block_size=16,
            num_gpu_blocks=100,
            gpu_kv_tokens=1_600,
        ),
        backend_handle="backend",
        now=3.0,
    )
    core.tick_once(now=3.0)
    first = core.poll_grant(requests[0][1])
    second = core.poll_grant(requests[1][1])
    assert first is not None and second is not None

    decision = core.complete(
        first.task_id,
        agent_report(core, first, status="oom"),
        acquire_id=first.acquire_id,
    )
    assert decision.retry_acquire is True
    replica = core.replicas[action.replica_id]
    assert replica.state == ModelReplicaState.SUSPECT
    assert replica.active_acquire_ids == {second.acquire_id}
    assert core.tick_once(now=4.0) == []

    core.complete(
        second.task_id,
        agent_report(core, second),
        acquire_id=second.acquire_id,
    )
    eviction = core.tick_once(now=5.0)
    assert len(eviction) == 1
    assert isinstance(eviction[0], EvictReplicaAction)
    assert eviction[0].reason == "suspect_cleanup"


def test_agent_completion_releases_replica_before_output_emission() -> None:
    core = resource_core()
    task_id, acquire_id = request_agent(core, "s1")
    grant, _ = load_and_grant(core, acquire_id)

    decision = core.complete(
        task_id,
        agent_report(core, grant),
        acquire_id=acquire_id,
    )

    assert decision.emit_output is True
    assert decision.retry_acquire is False
    assert core.tasks[task_id].state == NodeTaskState.EMITTING
    assert core.replicas[grant.replica_id].state == ModelReplicaState.IDLE
    assert grant.acquire_id not in core.grants


def test_late_failed_agent_completion_releases_lease_and_cancels_task() -> None:
    core = resource_core()
    task_id, acquire_id = request_agent(core, "s1")
    grant, _ = load_and_grant(core, acquire_id)
    core.sessions["s1"].state = SessionState.FAILED

    decision = core.complete(
        task_id,
        agent_report(core, grant, status="failed"),
        acquire_id=acquire_id,
    )

    assert decision.emit_output is False
    assert decision.retry_acquire is False
    assert core.tasks[task_id].state == NodeTaskState.CANCELLED
    assert core.replicas[grant.replica_id].state == ModelReplicaState.IDLE
    assert acquire_id not in core.grants


def test_granted_acquire_cannot_be_cancelled() -> None:
    core = resource_core()
    _, acquire_id = request_agent(core, "s1")
    load_and_grant(core, acquire_id)

    with pytest.raises(AcquireAlreadyGrantedError, match="already granted"):
        core.cancel_acquire(acquire_id)


def test_load_identity_mismatch_is_atomic() -> None:
    core = resource_core()
    _, acquire_id = request_agent(core, "s1")
    actions = core.tick_once(now=2.0)
    action = actions[0]
    assert isinstance(action, LoadReplicaAction)

    with pytest.raises(ValueError, match="physical GPU"):
        core.complete_load(
            action.replica_id,
            ReplicaLoadResult(
                physical_gpu_id=7,
                duration_sec=1.0,
                idle_vram_mb=8_000,
                block_size=16,
                num_gpu_blocks=100,
                gpu_kv_tokens=1_600,
            ),
            backend_handle="backend",
            now=3.0,
        )

    assert core.replicas[action.replica_id].state == ModelReplicaState.LOADING
    assert core.replicas[action.replica_id].backend_handle is None
    assert core.poll_grant(acquire_id) is None


def test_load_rebinds_same_host_reservation_to_reported_local_index() -> None:
    accelerators = (
        AcceleratorConfig(
            hostname="gpu-node",
            gpu_kind="v100",
            local_index=0,
            total_mem_mb=16_000,
        ),
        AcceleratorConfig(
            hostname="gpu-node",
            gpu_kind="v100",
            local_index=1,
            total_mem_mb=16_000,
        ),
    )
    core = resource_core(accelerators=accelerators)
    _, acquire_id = request_agent(core, "s1")
    action = core.tick_once(now=2.0)[0]
    assert isinstance(action, LoadReplicaAction)
    assert action.accelerator.local_index == 0

    core.complete_load(
        action.replica_id,
        ReplicaLoadResult(
            physical_gpu_id=1,
            duration_sec=1.0,
            idle_vram_mb=8_000,
            block_size=16,
            num_gpu_blocks=100,
            gpu_kv_tokens=1_600,
        ),
        backend_handle="backend",
        now=3.0,
    )
    core.tick_once(now=3.0)

    grant = core.poll_grant(acquire_id)
    assert grant is not None
    assert grant.accelerator_ids == ("gpu-node/v100:1",)
    assert core.accelerators["gpu-node/v100:0"].replica_id is None
    assert core.accelerators["gpu-node/v100:1"].replica_id == action.replica_id


def test_scheduler_config_rejects_mixed_gpu_kinds_on_one_hostname() -> None:
    accelerators = (
        AcceleratorConfig(
            hostname="mixed-node",
            gpu_kind="v100",
            local_index=0,
            total_mem_mb=16_000,
        ),
        AcceleratorConfig(
            hostname="mixed-node",
            gpu_kind="a100",
            local_index=1,
            total_mem_mb=40_000,
        ),
    )

    with pytest.raises(ValueError, match="GPU kind"):
        SchedulerConfig(accelerators=accelerators)


def test_load_rejects_missing_backend_before_state_mutation() -> None:
    core = resource_core()
    _, acquire_id = request_agent(core, "s1")
    action = core.tick_once(now=2.0)[0]

    with pytest.raises(ValueError, match="backend"):
        core.complete_load(
            action.replica_id,
            ReplicaLoadResult(
                physical_gpu_id=0,
                duration_sec=1.0,
                idle_vram_mb=8_000,
                block_size=16,
                num_gpu_blocks=100,
                gpu_kv_tokens=1_600,
            ),
            backend_handle=None,
            now=3.0,
        )

    assert core.replicas[action.replica_id].state == ModelReplicaState.LOADING
    assert core.replicas[action.replica_id].backend_handle is None
    assert core.poll_grant(acquire_id) is None


def test_load_normalizes_numeric_string_gpu_identity() -> None:
    core = resource_core()
    _, _acquire_id = request_agent(core, "s1")
    action = core.tick_once(now=2.0)[0]
    assert isinstance(action, LoadReplicaAction)

    core.complete_load(
        action.replica_id,
        ReplicaLoadResult(
            physical_gpu_id="0",
            duration_sec=1.0,
            idle_vram_mb=8_000,
            block_size=16,
            num_gpu_blocks=100,
            gpu_kv_tokens=1_600,
        ),
        backend_handle="backend",
        now=3.0,
    )

    assert core.replicas[action.replica_id].physical_gpu_id == 0


def test_load_rejects_non_numeric_gpu_identity_atomically() -> None:
    core = resource_core()
    _, acquire_id = request_agent(core, "s1")
    action = core.tick_once(now=2.0)[0]
    assert isinstance(action, LoadReplicaAction)

    with pytest.raises(ValueError, match="physical GPU"):
        core.complete_load(
            action.replica_id,
            ReplicaLoadResult(
                physical_gpu_id="gpu-zero",
                duration_sec=1.0,
                idle_vram_mb=8_000,
                block_size=16,
                num_gpu_blocks=100,
                gpu_kv_tokens=1_600,
            ),
            backend_handle="backend",
            now=3.0,
        )

    assert core.replicas[action.replica_id].state == ModelReplicaState.LOADING
    assert core.replicas[action.replica_id].backend_handle is None
    assert core.poll_grant(acquire_id) is None


def test_severe_cuda_failure_marks_replica_suspect() -> None:
    core = resource_core()
    task_id, acquire_id = request_agent(core, "s1")
    grant, _ = load_and_grant(core, acquire_id)
    report = agent_report(core, grant, status="failed").model_copy(
        update={"error_type": "CUDARuntimeError"}
    )

    decision = core.complete(task_id, report, acquire_id=acquire_id)

    assert decision.emit_output is False
    assert core.replicas[grant.replica_id].state == ModelReplicaState.SUSPECT
    assert core.tasks[task_id].state == NodeTaskState.FAILED


def test_resource_ledger_rejects_extra_replica_pair_index() -> None:
    core = resource_core()
    core._replica_pairs[("missing-model", "v100")] = "missing-replica"

    with pytest.raises(RuntimeError, match="pairs"):
        core.tick_once(now=1.0)


def test_resource_ledger_rejects_overlapping_accelerator_ownership() -> None:
    core = resource_core()
    request_agent(core, "s1")
    action = core.tick_once(now=2.0)[0]
    assert isinstance(action, LoadReplicaAction)
    accelerator_id = action.accelerator.accelerator_id
    core.replicas[action.replica_id].accelerator_ids = (
        accelerator_id,
        accelerator_id,
    )

    with pytest.raises(RuntimeError, match="overlaps"):
        core.tick_once(now=3.0)


def test_infeasible_request_cancels_same_session_snapshot_siblings() -> None:
    core = resource_core(workflow=branched_agent_workflow())
    core.register_session("s1")
    finish_start(core, "s1")
    left_id = core.begin_node("s1", "left", ["left-input"])
    right_id = core.begin_node("s1", "right", ["right-input"])
    core.request_acquire(left_id, input_tokens=9_000, created_at=1.0)
    core.request_acquire(right_id, input_tokens=9_000, created_at=1.0)

    assert core.tick_once(now=2.0) == []
    assert core.tasks[left_id].state == NodeTaskState.FAILED
    assert core.tasks[right_id].state == NodeTaskState.CANCELLED


def test_failed_agent_retries_until_node_attempt_limit() -> None:
    core = resource_core(max_attempts=2)
    task_id, first_acquire = request_agent(core, "s1")
    first_grant, _ = load_and_grant(core, first_acquire)

    first_decision = core.complete(
        task_id,
        agent_report(core, first_grant, status="failed"),
        acquire_id=first_acquire,
    )

    assert first_decision.retry_acquire is True
    assert core.tasks[task_id].state == NodeTaskState.ACQUIRING
    assert core.tasks[task_id].execution_attempts == 1
    second_acquire = core.request_acquire(
        task_id,
        input_tokens=1100,
        created_at=6.0,
    )
    assert core.tick_once(now=7.0) == []
    second_grant = core.poll_grant(second_acquire)
    assert second_grant is not None

    second_decision = core.complete(
        task_id,
        agent_report(core, second_grant, status="failed"),
        acquire_id=second_acquire,
    )

    assert second_decision.retry_acquire is False
    assert core.tasks[task_id].state == NodeTaskState.FAILED
    assert core.tasks[task_id].execution_attempts == 2
    assert core.sessions["s1"].state == SessionState.FAILED


def test_first_oom_penalizes_exact_bucket_and_allows_one_reacquire() -> None:
    core = resource_core()
    task_id, acquire_id = request_agent(core, "s1")
    grant, _ = load_and_grant(core, acquire_id)

    decision = core.complete(
        task_id,
        agent_report(core, grant, status="oom"),
        acquire_id=acquire_id,
    )

    assert decision.emit_output is False
    assert decision.retry_acquire is True
    assert core.tasks[task_id].state == NodeTaskState.ACQUIRING
    assert core.replicas[grant.replica_id].state == ModelReplicaState.SUSPECT
    assert core.oom_penalties[grant.prediction_key] > 0
    retry_id = core.request_acquire(task_id, input_tokens=1100, created_at=6.0)
    retry_actions = core.tick_once(now=7.0)
    assert len(retry_actions) == 1
    retry_action = retry_actions[0]
    assert isinstance(retry_action, EvictReplicaAction)
    assert retry_action.reason == "suspect_cleanup"
    core.complete_eviction(retry_action.replica_id)
    load_actions = core.tick_once(now=8.0)
    assert len(load_actions) == 1
    load_action = load_actions[0]
    assert isinstance(load_action, LoadReplicaAction)
    assert load_action.accelerator.gpu_kind == "a100"
    assert core.poll_grant(retry_id) is None


def test_second_oom_fails_only_the_affected_session() -> None:
    core = resource_core()
    task_id, first_acquire = request_agent(core, "failed")
    first_grant, _ = load_and_grant(core, first_acquire)
    first_decision = core.complete(
        task_id,
        agent_report(core, first_grant, status="oom"),
        acquire_id=first_acquire,
    )
    assert first_decision.retry_acquire
    second_acquire = core.request_acquire(task_id, input_tokens=1100, created_at=6.0)
    cleanup_actions = core.tick_once(now=7.0)
    assert len(cleanup_actions) == 1
    cleanup_action = cleanup_actions[0]
    assert isinstance(cleanup_action, EvictReplicaAction)
    core.complete_eviction(cleanup_action.replica_id)
    second_grant, _ = load_and_grant(core, second_acquire, now=8.0)
    _, healthy_acquire = request_agent(core, "healthy", now=9.0)

    decision = core.complete(
        task_id,
        agent_report(core, second_grant, status="oom"),
        acquire_id=second_acquire,
    )

    assert decision.emit_output is False
    assert decision.retry_acquire is False
    assert core.tasks[task_id].state == NodeTaskState.FAILED
    assert core.sessions["failed"].state == SessionState.FAILED
    assert core.sessions["healthy"].state == SessionState.ACTIVE
    assert healthy_acquire in core.pending_acquires


def test_agent_completion_validates_grant_identity() -> None:
    core = resource_core()
    task_id, acquire_id = request_agent(core, "s1")
    grant, _ = load_and_grant(core, acquire_id)
    mismatched = agent_report(core, grant).model_copy(
        update={"accelerator_id": "other/a100:0"}
    )

    with pytest.raises(ValueError, match="grant"):
        core.complete(task_id, mismatched, acquire_id=acquire_id)

    assert core.replicas[grant.replica_id].state == ModelReplicaState.BUSY


def test_agent_completion_validates_grant_task_identity() -> None:
    core = resource_core()
    task_id, acquire_id = request_agent(core, "s1")
    grant, _ = load_and_grant(core, acquire_id)
    report = agent_report(core, grant)
    core.grants[acquire_id] = grant.model_copy(update={"task_id": "other-task"})

    with pytest.raises(ValueError, match="grant"):
        core.complete(task_id, report, acquire_id=acquire_id)

    assert core.replicas[grant.replica_id].state == ModelReplicaState.BUSY


def test_infeasible_prediction_fails_the_session_explicitly() -> None:
    core = resource_core()
    core.register_session("s1")
    task_id = core.begin_node("s1", "agent", ["input-s1"])
    acquire_id = core.request_acquire(
        task_id,
        input_tokens=9_000,
        created_at=1.0,
    )

    assert core.tick_once(now=2.0) == []
    assert core.tasks[task_id].state == NodeTaskState.FAILED
    assert core.tasks[task_id].error_type == "RequestInfeasible"
    assert core.sessions["s1"].state == SessionState.FAILED
    with pytest.raises(SessionInactiveError, match="session is not active"):
        core.poll_grant(acquire_id)
