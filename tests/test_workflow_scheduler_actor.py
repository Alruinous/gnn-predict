from __future__ import annotations

import asyncio
import threading
import time
from collections.abc import Callable
from contextlib import suppress
from typing import Any, Literal, cast

import pytest
import ray
from ray.exceptions import RayActorError, RayTaskError

import workflow.scheduler as scheduler_module
from workflow.artifacts import (
    AcceleratorConfig,
    PredictionCache,
    PredictionEntry,
    SchedulerConfig,
)
from workflow.replica import (
    ModelDeploymentConfig,
    ReplicaLoadResult,
)
from workflow.scheduler import (
    AgentTaskRuntimeReport,
    CompleteDecision,
    FunctionTaskRuntimeReport,
    GrantInfo,
    LoadReplicaAction,
    OutputReport,
    SchedulerActor,
)
from workflow.schema import ServingConfig, Workflow
from workflow.types import (
    ModelReplicaState,
    SessionState,
    TraceEvent,
    WorkflowModelFeatureKey,
)

_remote = cast(Any, ray.remote)


@_remote
class RecordingTraceWriter:
    def __init__(self) -> None:
        self.events: list[TraceEvent] = []

    def append_batch(self, events: list[TraceEvent]) -> int:
        self.events.extend(events)
        return events[-1].event_seq if events else -1

    def event_types(self) -> list[str]:
        return [event.event_type for event in self.events]


@_remote(max_concurrency=2)
class BlockingTraceWriter:
    def __init__(self) -> None:
        self.events: list[TraceEvent] = []
        self.released = asyncio.Event()

    async def append_batch(self, events: list[TraceEvent]) -> int:
        self.events.extend(events)
        await self.released.wait()
        return events[-1].event_seq if events else -1

    def release(self) -> None:
        self.released.set()


@_remote
class FailingTraceWriter:
    def append_batch(self, events: list[TraceEvent]) -> int:
        raise RuntimeError("trace append failed")


@_remote
class FakeWorker:
    def __init__(self) -> None:
        self.sessions: list[str] = []

    def cancel_session(self, session_id: str) -> None:
        self.sessions.append(session_id)

    def cancelled_sessions(self) -> list[str]:
        return list(self.sessions)


class LocalReplicaHandle:
    def shutdown(self) -> None:
        return None


@_remote
class FakeReplica:
    def __init__(self, deployment: ModelDeploymentConfig) -> None:
        self.deployment = deployment

    def load(self) -> ReplicaLoadResult:
        return ReplicaLoadResult(
            physical_gpu_id=0,
            duration_sec=0.01,
            idle_vram_mb=900,
            block_size=16,
            num_gpu_blocks=100,
            gpu_kv_tokens=1_600,
        )

    def ping(self) -> str:
        return self.deployment.model_name

    def shutdown(self) -> None:
        return None


@_remote(max_concurrency=2)
class BlockingMismatchedReplica:
    def __init__(self, deployment: ModelDeploymentConfig) -> None:
        self.deployment = deployment
        self.released = asyncio.Event()

    async def load(self) -> ReplicaLoadResult:
        await self.released.wait()
        return ReplicaLoadResult(
            physical_gpu_id=7,
            duration_sec=0.01,
            idle_vram_mb=900,
            block_size=16,
            num_gpu_blocks=100,
            gpu_kv_tokens=1_600,
        )

    def release(self) -> None:
        self.released.set()

    def ping(self) -> str:
        return self.deployment.model_name

    def shutdown(self) -> None:
        return None


class ImmediateTraceWriter:
    def append_batch(self, events: list[TraceEvent]) -> int:
        return events[-1].event_seq if events else -1


def fake_replica_factory(action: LoadReplicaAction) -> object:
    return cast(Any, FakeReplica).remote(action.deployment)


MISMATCHED_REPLICA_NAME = "workflow-test-mismatched-replica"


def mismatched_replica_factory(action: LoadReplicaAction) -> object:
    return (
        cast(Any, BlockingMismatchedReplica)
        .options(name=MISMATCHED_REPLICA_NAME)
        .remote(action.deployment)
    )


def function_workflow() -> Workflow:
    return Workflow.model_validate(
        {
            "nodes": [
                {
                    "name": "function",
                    "type": "function",
                    "function": "identity",
                }
            ],
            "edges": [],
        }
    )


def agent_workflow() -> Workflow:
    return Workflow.model_validate(
        {
            "nodes": [
                {
                    "name": "agent",
                    "type": "agent",
                    "model": {"name": "test-model"},
                    "execution": {
                        "model_path": "/models/test-model",
                        "max_new_tokens": 16,
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


def scheduler_config(*, max_tick_interval_sec: float = 0.5) -> SchedulerConfig:
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
        max_tick_interval_sec=max_tick_interval_sec,
    )


def predictions() -> PredictionCache:
    return PredictionCache(
        version=1,
        entries=(
            PredictionEntry(
                key=WorkflowModelFeatureKey(
                    model_name="test-model",
                    phase="decode",
                    gpu_name="v100",
                    batch_size=1,
                    sequence_length=128,
                    decode_output_length=8,
                ),
                predicted_load_sec=5.0,
                predicted_run_sec=0.1,
                predicted_peak_vram_mb=1_000,
            ),
            PredictionEntry(
                key=WorkflowModelFeatureKey(
                    model_name="test-model",
                    phase="decode",
                    gpu_name="v100",
                    batch_size=1,
                    sequence_length=128,
                    decode_output_length=16,
                ),
                predicted_load_sec=5.0,
                predicted_run_sec=0.2,
                predicted_peak_vram_mb=1_200,
            ),
        ),
    )


def scheduler_actor(
    trace_writer: object,
    *,
    workflow: Workflow | None = None,
    config: SchedulerConfig | None = None,
    with_resources: bool = False,
    replica_factory: object = fake_replica_factory,
) -> Any:
    actor = SchedulerActor.remote(
        workflow=workflow or function_workflow(),
        scheduler_config=config or SchedulerConfig(max_tick_interval_sec=0.5),
        predictions=predictions() if with_resources else None,
        trace_writer=trace_writer,
        replica_factory=replica_factory,
    )
    ray.get(actor.get_run_failure.remote(), timeout=15)
    return actor


def function_report(task_id: str, session_id: str) -> FunctionTaskRuntimeReport:
    return FunctionTaskRuntimeReport(
        task_id=task_id,
        session_id=session_id,
        node_id="function",
        input_item_ids=["item-1"],
        started_at=1.0,
        finished_at=2.0,
        duration_sec=1.0,
        status="success",
    )


def agent_report(
    grant: GrantInfo,
    session_id: str,
    *,
    status: Literal["success", "failed", "oom"] = "success",
) -> AgentTaskRuntimeReport:
    return AgentTaskRuntimeReport(
        acquire_id=grant.acquire_id,
        task_id=grant.task_id,
        session_id=session_id,
        node_id="agent",
        input_item_ids=["item-1"],
        model_key=grant.model_key,
        accelerator_id=grant.accelerator_ids[0],
        gpu_kind=grant.gpu_kind,
        input_tokens=grant.input_tokens,
        max_new_tokens=grant.max_new_tokens,
        output_tokens=4 if status == "success" else 0,
        hit_token_limit=False,
        started_at=1.0,
        finished_at=2.0,
        duration_sec=1.0,
        status=status,
        error_type=None if status == "success" else "OutOfMemoryError",
    )


def local_scheduler(
    *,
    config: SchedulerConfig | None = None,
) -> Any:
    return scheduler_module._SchedulerActor(
        workflow=function_workflow(),
        scheduler_config=config or SchedulerConfig(),
        predictions=None,
        trace_writer=ImmediateTraceWriter(),
        replica_factory=fake_replica_factory,
    )


def named_actor(name: str) -> Any | None:
    try:
        return ray.get_actor(name)
    except ValueError:
        return None


def stop_actor(actor: Any, run_ref: ray.ObjectRef[Any]) -> None:
    ray.get(actor.stop_loop.remote(), timeout=5)
    ray.get(run_ref, timeout=5)


def wait_for_grant(actor: Any, acquire_id: str, timeout_sec: float = 10) -> GrantInfo:
    deadline = time.monotonic() + timeout_sec
    while time.monotonic() < deadline:
        grant = ray.get(actor.poll_grant.remote(acquire_id), timeout=5)
        if grant is not None:
            assert isinstance(grant, GrantInfo)
            return grant
        time.sleep(0.02)
    raise TimeoutError(acquire_id)


def test_mutations_wait_for_run_loop_and_commands_wake_it_early(
    ray_session: None,
    wait_until: Callable[..., bool],
) -> None:
    trace_writer = cast(Any, RecordingTraceWriter).remote()
    actor = scheduler_actor(
        trace_writer,
        config=SchedulerConfig(max_tick_interval_sec=10.0),
    )
    register_ref = actor.register_session.remote("s1", 1.0)
    ready, _ = ray.wait([register_ref], timeout=0.1)
    assert ready == []
    ray.get(actor.get_run_failure.remote(), timeout=15)

    started = time.monotonic()
    run_ref = actor.run.remote()
    try:
        ray.get(register_ref, timeout=5)
        assert time.monotonic() - started < 2.0
        assert ray.get(actor.get_session_state.remote("s1")) == SessionState.ACTIVE
        assert ray.get(actor.drain_complete.remote()) is False

        task_id = ray.get(
            actor.begin_node.remote("s1", "function", ["item-1"]), timeout=5
        )
        decision = ray.get(
            actor.complete.remote(task_id, function_report(task_id, "s1")), timeout=5
        )
        assert decision == CompleteDecision(emit_output=True)
        ray.get(
            actor.finish_node.remote(
                task_id,
                OutputReport(
                    task_id=task_id,
                    output_item_ids=[],
                    persisted_terminal_result=True,
                ),
            ),
            timeout=5,
        )
        assert wait_until(lambda: ray.get(actor.drain_complete.remote()), timeout_sec=5)
        stop_actor(actor, run_ref)
    finally:
        for handle in (actor, trace_writer):
            with suppress(Exception):
                ray.kill(handle, no_restart=True)


def test_session_failure_dispatches_cancellation_to_registered_workers(
    ray_session: None,
    wait_until: Callable[..., bool],
) -> None:
    trace_writer = cast(Any, RecordingTraceWriter).remote()
    worker = cast(Any, FakeWorker).remote()
    actor = scheduler_actor(trace_writer)
    run_ref = actor.run.remote()
    ray.get(actor.register_workers.remote({"function": worker}), timeout=5)
    ray.get(actor.register_session.remote("failed", 1.0), timeout=5)
    task_id = ray.get(
        actor.begin_node.remote("failed", "function", ["item-1"]), timeout=5
    )

    ray.get(actor.fail_node.remote(task_id, "RuntimeError", "failed"), timeout=5)

    assert wait_until(
        lambda: ray.get(worker.cancelled_sessions.remote()) == ["failed"],
        timeout_sec=5,
    )
    assert ray.get(actor.get_session_state.remote("failed")) == SessionState.FAILED
    stop_actor(actor, run_ref)


def test_load_action_watcher_grants_without_a_second_request(
    ray_session: None,
) -> None:
    trace_writer = cast(Any, RecordingTraceWriter).remote()
    actor = scheduler_actor(
        trace_writer,
        workflow=agent_workflow(),
        config=scheduler_config(),
        with_resources=True,
    )
    run_ref = actor.run.remote()
    ray.get(actor.register_session.remote("s1", 1.0), timeout=5)
    task_id = ray.get(actor.begin_node.remote("s1", "agent", ["item-1"]), timeout=5)
    acquire_id = ray.get(actor.request_acquire.remote(task_id, 64, 2.0), timeout=5)

    grant = wait_for_grant(actor, acquire_id)

    backend = cast(Any, grant.backend_handle)
    assert ray.get(backend.ping.remote(), timeout=5) == "test-model"
    stop_actor(actor, run_ref)


def test_eviction_watcher_removes_replica_and_releases_backend(
    ray_session: None,
    wait_until: Callable[..., bool],
) -> None:
    trace_writer = cast(Any, RecordingTraceWriter).remote()
    actor = scheduler_actor(
        trace_writer,
        workflow=agent_workflow(),
        config=scheduler_config(),
        with_resources=True,
    )
    run_ref = actor.run.remote()
    ray.get(actor.register_session.remote("s1", 1.0), timeout=5)
    task_id = ray.get(actor.begin_node.remote("s1", "agent", ["item-1"]), timeout=5)
    acquire_id = ray.get(actor.request_acquire.remote(task_id, 64, 2.0), timeout=5)
    grant = wait_for_grant(actor, acquire_id)
    decision = ray.get(
        actor.complete.remote(
            task_id,
            agent_report(grant, "s1"),
            acquire_id=acquire_id,
        ),
        timeout=5,
    )
    assert decision.emit_output is True

    ray.get(actor.evict_replica.remote(grant.replica_id), timeout=5)

    assert wait_until(
        lambda: ray.get(actor.get_replica_state.remote(grant.replica_id)) is None,
        timeout_sec=5,
    )
    backend = cast(Any, grant.backend_handle)
    with pytest.raises(RayActorError):
        ray.get(backend.ping.remote(), timeout=5)
    stop_actor(actor, run_ref)


def test_drain_waits_for_trace_acknowledgement(
    ray_session: None,
    wait_until: Callable[..., bool],
) -> None:
    trace_writer = cast(Any, BlockingTraceWriter).remote()
    actor = scheduler_actor(trace_writer)
    run_ref = actor.run.remote()
    ray.get(actor.register_session.remote("s1", 1.0), timeout=5)
    task_id = ray.get(actor.begin_node.remote("s1", "function", ["item-1"]), timeout=5)
    ray.get(actor.fail_node.remote(task_id, "RuntimeError", "failed"), timeout=5)

    assert ray.get(actor.drain_complete.remote()) is False
    ray.get(trace_writer.release.remote(), timeout=5)
    assert wait_until(lambda: ray.get(actor.drain_complete.remote()), timeout_sec=5)
    stop_actor(actor, run_ref)


def test_trace_failure_terminates_the_observable_run_loop(ray_session: None) -> None:
    trace_writer = cast(Any, FailingTraceWriter).remote()
    actor = scheduler_actor(trace_writer)
    run_ref = actor.run.remote()

    with pytest.raises(RayTaskError, match="trace append failed"):
        ray.get(run_ref, timeout=10)


def test_load_completion_failure_rolls_back_and_kills_replica(
    ray_session: None,
    wait_until: Callable[..., bool],
) -> None:
    trace_writer = cast(Any, RecordingTraceWriter).remote()
    actor = scheduler_actor(
        trace_writer,
        workflow=agent_workflow(),
        config=scheduler_config(),
        with_resources=True,
        replica_factory=mismatched_replica_factory,
    )
    run_ref = actor.run.remote()
    backend: Any | None = None
    try:
        ray.get(actor.register_session.remote("s1", 1.0), timeout=5)
        task_id = ray.get(actor.begin_node.remote("s1", "agent", ["item-1"]), timeout=5)
        ray.get(actor.request_acquire.remote(task_id, 64, 2.0), timeout=5)
        assert wait_until(
            lambda: named_actor(MISMATCHED_REPLICA_NAME) is not None,
            timeout_sec=5,
        )
        backend = named_actor(MISMATCHED_REPLICA_NAME)
        assert backend is not None
        ray.get(backend.release.remote(), timeout=5)

        with pytest.raises(RayTaskError, match="physical GPU"):
            ray.get(run_ref, timeout=10)
        replicas_empty = ray.get(actor.replicas_empty.remote(), timeout=5)
        try:
            ray.get(backend.ping.remote(), timeout=2)
        except RayActorError:
            backend_alive = False
        else:
            backend_alive = True
        assert (replicas_empty, backend_alive) == (True, False)
    finally:
        if backend is not None:
            with suppress(Exception):
                ray.kill(backend, no_restart=True)
        ray.kill(actor, no_restart=True)


def test_fatal_command_rejects_later_command_in_same_batch() -> None:
    async def run_scenario() -> None:
        actor = local_scheduler()
        future = asyncio.get_running_loop().create_future()
        actor._commands.put_nowait(
            scheduler_module._SchedulerCommand(
                name="trace_failed",
                args=("fatal",),
                kwargs={},
                future=None,
            )
        )
        actor._commands.put_nowait(
            scheduler_module._SchedulerCommand(
                name="register_session",
                args=("late", 1.0),
                kwargs={},
                future=future,
            )
        )

        with pytest.raises(RuntimeError, match="trace append failed"):
            await actor.run()
        assert future.done()
        with pytest.raises(RuntimeError, match="trace append failed"):
            future.result()

    asyncio.run(run_scenario())


def test_eviction_watcher_reports_configured_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    released = threading.Event()

    def blocking_kill(handle: object, *, no_restart: bool) -> None:
        released.wait()

    monkeypatch.setattr(scheduler_module.ray, "kill", blocking_kill)

    async def run_scenario() -> None:
        actor = local_scheduler(config=SchedulerConfig(eviction_timeout_sec=0.01))
        watcher = asyncio.create_task(
            actor._watch_eviction("replica", LocalReplicaHandle())
        )
        try:
            command = await asyncio.wait_for(actor._commands.get(), timeout=0.2)
        finally:
            released.set()
            await watcher
        assert command.name == "eviction_failed"
        assert command.args[1] == "TimeoutError"

    asyncio.run(run_scenario())


def test_failed_replica_kill_keeps_handle_for_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    actor = local_scheduler()
    handle = LocalReplicaHandle()
    actor._replica_handles["replica"] = handle
    attempts = 0

    def flaky_kill(value: object, *, no_restart: bool) -> None:
        nonlocal attempts
        attempts += 1
        assert value is handle
        assert no_restart is True
        if attempts == 1:
            raise RuntimeError("kill failed")

    monkeypatch.setattr(scheduler_module.ray, "kill", flaky_kill)

    async def run_scenario() -> None:
        await actor._kill_replica_handles()
        assert actor._replica_handles == {"replica": handle}
        await actor._kill_replica_handles()
        assert actor._replica_handles == {}

    asyncio.run(run_scenario())


def test_force_cleanup_rejects_late_load_dispatch() -> None:
    actor = local_scheduler()

    def fail_if_called(action: LoadReplicaAction) -> object:
        raise AssertionError("replica factory must not run during cleanup")

    actor._replica_factory = fail_if_called
    action = LoadReplicaAction(
        replica_id="late-replica",
        deployment=ModelDeploymentConfig(
            model_name="test-model",
            model_path="/models/test-model",
            dtype="float16",
            serving=ServingConfig(
                max_model_len=1024,
                max_num_seqs=1,
                max_num_batched_tokens=1024,
            ),
        ),
        accelerator=AcceleratorConfig(
            hostname="local",
            gpu_kind="v100",
            local_index=0,
            total_mem_mb=16_000,
        ),
        reason="ready_load",
        expected_load_sec=1.0,
    )

    async def run_scenario() -> None:
        await actor.force_kill_replicas()
        actor._dispatch_load(action)
        command = actor._commands.get_nowait()
        assert command.name == "load_failed"
        assert command.args[1] == "RuntimeStopping"
        assert actor._replica_handles == {}

    asyncio.run(run_scenario())


def test_retryable_oom_records_retry_and_suspect_not_task_failure(
    ray_session: None,
    wait_until: Callable[..., bool],
) -> None:
    trace_writer = cast(Any, RecordingTraceWriter).remote()
    actor = scheduler_actor(
        trace_writer,
        workflow=agent_workflow(),
        config=scheduler_config(),
        with_resources=True,
    )
    run_ref = actor.run.remote()
    try:
        ray.get(actor.register_session.remote("s1", 1.0), timeout=5)
        task_id = ray.get(actor.begin_node.remote("s1", "agent", ["item-1"]), timeout=5)
        acquire_id = ray.get(actor.request_acquire.remote(task_id, 64, 2.0), timeout=5)
        grant = wait_for_grant(actor, acquire_id)
        decision = ray.get(
            actor.complete.remote(
                task_id,
                agent_report(grant, "s1", status="oom"),
                acquire_id=acquire_id,
            ),
            timeout=5,
        )
        assert decision.retry_acquire is True
        assert wait_until(
            lambda: (
                "task_execution_finished" in ray.get(trace_writer.event_types.remote())
            ),
            timeout_sec=5,
        )
        event_types = set(ray.get(trace_writer.event_types.remote()))
        assert {
            "task_retried",
            "model_suspect",
        }.issubset(event_types) and "task_failed" not in event_types
    finally:
        stop_actor(actor, run_ref)


def test_stop_rejects_later_mutation_while_final_trace_is_pending(
    ray_session: None,
) -> None:
    trace_writer = cast(Any, BlockingTraceWriter).remote()
    actor = scheduler_actor(trace_writer)
    run_ref = actor.run.remote()
    ray.get(actor.stop_loop.remote(), timeout=5)
    try:
        with pytest.raises(RayTaskError):
            ray.get(actor.register_session.remote("late", 1.0), timeout=5)
    finally:
        ray.get(trace_writer.release.remote(), timeout=5)
        ray.get(run_ref, timeout=5)


def test_run_finished_follows_lifecycle_watchers() -> None:
    class RecordingWriter:
        def __init__(self) -> None:
            self.event_types: list[str] = []

        def append_batch(self, events: list[TraceEvent]) -> int:
            self.event_types.extend(event.event_type for event in events)
            return events[-1].event_seq if events else -1

    writer = RecordingWriter()
    actor = local_scheduler()
    actor._trace_writer = writer

    async def finish_watcher() -> None:
        await asyncio.sleep(0.01)
        actor._record_trace("watcher_finished")
        await actor._enqueue_internal("stop_loop")

    async def run_scenario() -> None:
        actor._start_watcher(finish_watcher())
        run_task = asyncio.create_task(actor.run())
        await actor.stop_loop()
        await asyncio.wait_for(run_task, timeout=1.0)

    asyncio.run(run_scenario())

    assert writer.event_types[-2:] == ["watcher_finished", "run_finished"]
