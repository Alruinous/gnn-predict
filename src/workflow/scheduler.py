from __future__ import annotations

import asyncio
import inspect
import os
import time
from collections import deque
from collections.abc import Callable, Coroutine, Mapping
from contextlib import suppress
from dataclasses import dataclass
from math import ceil, inf
from pathlib import Path
from typing import Any, Literal, cast
from uuid import uuid4

import ray
from pydantic import BaseModel, ConfigDict, Field, NonNegativeInt, PositiveInt
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

from common.validate import NonEmptyStr
from workflow.artifacts import (
    AcceleratorConfig,
    GpuKind,
    PredictionCache,
    PredictionEntry,
    SchedulerConfig,
)
from workflow.policy import (
    EvictionCandidate,
    PlacementDecision,
    select_eviction_victim,
    select_placement,
)
from workflow.replica import (
    ModelDeploymentConfig,
    ModelReplicaActor,
    ReplicaLoadResult,
)
from workflow.schema import AgentNodeConfig, Workflow
from workflow.types import (
    ModelReplicaState,
    NodeTaskState,
    SessionState,
    TraceEvent,
    WorkflowModelFeatureKey,
)


class SessionInactiveError(ValueError):
    pass


class AcquireAlreadyGrantedError(ValueError):
    pass


class TaskCancelledError(ValueError):
    pass


def _normalize_physical_gpu_id(value: int | str) -> int:
    if isinstance(value, int):
        return value
    if not value.isascii() or not value.isdecimal():
        raise ValueError("reported physical GPU id must be a numeric string")
    return int(value)


def _is_severe_cuda_failure(report: AgentTaskRuntimeReport) -> bool:
    return (
        report.status == "failed"
        and report.error_type is not None
        and "cuda" in report.error_type.casefold()
    )


class StrictFrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class MutableRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, validate_assignment=True)


class FunctionTaskRuntimeReport(StrictFrozenModel):
    task_id: NonEmptyStr
    session_id: NonEmptyStr
    node_id: NonEmptyStr
    input_item_ids: list[NonEmptyStr]
    started_at: float
    finished_at: float
    duration_sec: float = Field(ge=0)
    status: Literal["success", "failed"]
    error_type: str | None = None
    error_message: str | None = None


class AgentTaskRuntimeReport(StrictFrozenModel):
    acquire_id: NonEmptyStr
    task_id: NonEmptyStr
    session_id: NonEmptyStr
    node_id: NonEmptyStr
    input_item_ids: list[NonEmptyStr]
    model_key: NonEmptyStr
    accelerator_id: NonEmptyStr
    gpu_kind: NonEmptyStr
    input_tokens: PositiveInt
    max_new_tokens: PositiveInt
    output_tokens: NonNegativeInt
    hit_token_limit: bool
    finish_reason: Literal["stop", "length", "abort"] | None = None
    queue_time_sec: float | None = Field(default=None, ge=0)
    time_to_first_token_sec: float | None = Field(default=None, ge=0)
    replica_inflight_at_start: PositiveInt = 1
    started_at: float
    finished_at: float
    duration_sec: float = Field(ge=0)
    status: Literal["success", "failed", "oom", "cancelled"]
    engine_failed: bool = False
    error_type: str | None = None


class ItemTraceReport(StrictFrozenModel):
    session_id: NonEmptyStr
    item_id: NonEmptyStr
    source_node: str | None
    target_node: NonEmptyStr
    emitted_at: float
    enqueued_at: float


class InputTraceReport(StrictFrozenModel):
    session_id: NonEmptyStr
    item_id: NonEmptyStr
    source_node: str | None
    target_node: NonEmptyStr
    dequeued_at: float
    fanin_state: Literal["wait", "ready"]
    waiting_for_sources: tuple[NonEmptyStr, ...] = ()
    source_nodes: tuple[NonEmptyStr, ...] = ()
    input_item_ids: tuple[NonEmptyStr, ...] = ()


class OutputReport(StrictFrozenModel):
    task_id: NonEmptyStr
    output_item_ids: list[NonEmptyStr]
    persisted_terminal_result: bool
    output_items: tuple[ItemTraceReport, ...] = ()


class CompleteDecision(StrictFrozenModel):
    emit_output: bool
    retry_acquire: bool = False


class CancelSessionAction(StrictFrozenModel):
    session_id: NonEmptyStr


class GrantInfo(StrictFrozenModel):
    acquire_id: NonEmptyStr
    task_id: NonEmptyStr
    replica_id: NonEmptyStr
    backend_handle: object
    accelerator_ids: tuple[NonEmptyStr, ...]
    model_key: NonEmptyStr
    gpu_kind: GpuKind
    input_tokens: PositiveInt
    max_new_tokens: PositiveInt
    admitted_batch_size: PositiveInt = 1
    prediction_key: WorkflowModelFeatureKey


class LoadReplicaAction(StrictFrozenModel):
    replica_id: NonEmptyStr
    deployment: ModelDeploymentConfig
    accelerator: AcceleratorConfig
    reason: Literal["ready_load", "near_ready_prefetch"]
    expected_load_sec: float = Field(gt=0)
    session_id: str | None = None
    node_id: str | None = None
    prefetch_at: float | None = None


class EvictReplicaAction(StrictFrozenModel):
    replica_id: NonEmptyStr
    accelerator_ids: tuple[NonEmptyStr, ...]
    reason: Literal[
        "suspect_cleanup",
        "ready_load",
        "near_ready_prefetch",
        "requested",
    ]
    reuse_distance_sec: float | None = None
    reload_cost_sec: float | None = None
    session_id: str | None = None
    node_id: str | None = None
    prefetch_at: float | None = None


class NearReadyTask(StrictFrozenModel):
    session_id: NonEmptyStr
    node_id: NonEmptyStr
    estimated_input_tokens: PositiveInt
    upstream_eta: float
    prefetch_at: float
    load_sec: float = Field(gt=0)
    deployment: ModelDeploymentConfig
    decision: PlacementDecision


HISTORY_SAMPLE_LIMIT = 64


class RuntimeHistoryRecord(MutableRecord):
    samples: list[AgentTaskRuntimeReport] = Field(default_factory=list)
    duration_sec_ema: float = Field(default=0, ge=0)
    oom_count: NonNegativeInt = 0

    def add(self, report: AgentTaskRuntimeReport, ema_alpha: float) -> None:
        self.duration_sec_ema = (
            report.duration_sec
            if not self.samples
            else ema_alpha * report.duration_sec
            + (1 - ema_alpha) * self.duration_sec_ema
        )
        self.samples.append(report)
        if len(self.samples) > HISTORY_SAMPLE_LIMIT:
            del self.samples[: len(self.samples) - HISTORY_SAMPLE_LIMIT]
        if report.status == "oom":
            self.oom_count += 1

    @property
    def output_tokens_p90(self) -> int:
        if not self.samples:
            return 0
        values = sorted(report.output_tokens for report in self.samples)
        return values[ceil(0.9 * len(values)) - 1]


class DurationHistoryRecord(MutableRecord):
    duration_sec_ema: float = Field(ge=0)
    samples: PositiveInt = 1

    def add(self, duration_sec: float, ema_alpha: float) -> None:
        self.duration_sec_ema = (
            ema_alpha * duration_sec + (1 - ema_alpha) * self.duration_sec_ema
        )
        self.samples += 1


class RunningTaskEstimate(StrictFrozenModel):
    session_id: NonEmptyStr
    node_id: NonEmptyStr
    finish_at: float


class SessionRecord(MutableRecord):
    session_id: NonEmptyStr
    state: SessionState = SessionState.ACTIVE
    task_ids: dict[str, str] = Field(default_factory=dict)


class NodeTaskRecord(MutableRecord):
    task_id: NonEmptyStr
    session_id: NonEmptyStr
    node_id: NonEmptyStr
    node_kind: Literal["agent", "function"]
    input_item_ids: list[NonEmptyStr] = Field(default_factory=list)
    state: NodeTaskState = NodeTaskState.PENDING
    runtime_report: FunctionTaskRuntimeReport | AgentTaskRuntimeReport | None = None
    output_report: OutputReport | None = None
    error_type: str | None = None
    error_message: str | None = None
    execution_attempts: NonNegativeInt = 0
    oom_attempts: NonNegativeInt = 0


class PendingAcquire(MutableRecord):
    acquire_id: NonEmptyStr
    task_id: NonEmptyStr
    input_tokens: PositiveInt
    created_at: float
    request_seq: NonNegativeInt


class AcceleratorRecord(MutableRecord):
    config: AcceleratorConfig
    replica_id: str | None = None


class ModelReplicaRecord(MutableRecord):
    replica_id: NonEmptyStr
    deployment: ModelDeploymentConfig
    model_key: NonEmptyStr
    gpu_kind: GpuKind
    accelerator_ids: tuple[NonEmptyStr, ...]
    state: ModelReplicaState
    expected_load_sec: float = Field(gt=0)
    backend_handle: object | None = None
    physical_gpu_id: int | str | None = None
    active_acquire_ids: set[str] = Field(default_factory=set)
    created_at: float
    idle_since: float | None = None
    load_duration_sec: float | None = Field(default=None, ge=0)
    idle_vram_mb: float | None = Field(default=None, ge=0)
    vllm_version: str | None = None
    engine_mode: str | None = None
    attention_backend: str | None = None
    block_size: int | None = Field(default=None, gt=0)
    num_gpu_blocks: int | None = Field(default=None, gt=0)
    gpu_kv_tokens: int | None = Field(default=None, gt=0)
    eviction_error_type: str | None = None
    eviction_error_message: str | None = None


class SchedulerCore:
    def __init__(
        self,
        workflow: Workflow,
        *,
        scheduler_config: SchedulerConfig | None = None,
        predictions: PredictionCache | None = None,
    ) -> None:
        self.workflow = workflow
        self.scheduler_config = scheduler_config or SchedulerConfig()
        self.predictions = predictions
        self.sessions: dict[str, SessionRecord] = {}
        self.tasks: dict[str, NodeTaskRecord] = {}
        self.pending_acquires: dict[str, PendingAcquire] = {}
        self.pending_order: deque[str] = deque()
        self.grants: dict[str, GrantInfo] = {}
        self._inactive_acquires: dict[str, str] = {}
        self.accelerators = {
            accelerator.accelerator_id: AcceleratorRecord(config=accelerator)
            for accelerator in self.scheduler_config.accelerators
        }
        self.replicas: dict[str, ModelReplicaRecord] = {}
        self.oom_penalties: dict[WorkflowModelFeatureKey, float] = {}
        self.history: dict[tuple[str, str, GpuKind], RuntimeHistoryRecord] = {}
        self.duration_history: dict[
            tuple[str, str, GpuKind, int], DurationHistoryRecord
        ] = {}
        self.load_history: dict[tuple[str, GpuKind], DurationHistoryRecord] = {}
        self.running_estimates: dict[tuple[str, str], RunningTaskEstimate] = {}
        self._nodes = workflow.node_map()
        self._replica_pairs: dict[tuple[str, GpuKind], str] = {}
        self._node_order = {
            node_id: index
            for index, node_id in enumerate(workflow.graph.topological_order)
        }
        self._actions: list[CancelSessionAction] = []
        self._next_request_seq = 0

    def register_session(self, session_id: str) -> None:
        if session_id in self.sessions:
            raise ValueError(f"session already registered: {session_id}")

        session = SessionRecord(session_id=session_id)
        for node in self.workflow.nodes:
            task_id = str(uuid4())
            task = NodeTaskRecord(
                task_id=task_id,
                session_id=session_id,
                node_id=node.name,
                node_kind=node.type,
            )
            session.task_ids[node.name] = task_id
            self.tasks[task_id] = task
        self.sessions[session_id] = session

    def begin_node(
        self,
        session_id: str,
        node_id: str,
        input_item_ids: list[str],
    ) -> str:
        session = self._session(session_id)
        if session.state != SessionState.ACTIVE:
            raise SessionInactiveError(f"session is not active: {session_id}")
        if node_id not in self._nodes:
            raise KeyError(f"unknown node: {node_id}")

        task = self.tasks[session.task_ids[node_id]]
        if task.state != NodeTaskState.PENDING:
            raise ValueError(f"node already begun: {session_id}/{node_id}")
        if not self._dependencies_completed(session, node_id):
            raise ValueError(f"node dependencies are not completed: {node_id}")

        task.input_item_ids = input_item_ids
        node = self._nodes[node_id]
        task.state = (
            NodeTaskState.ACQUIRING
            if isinstance(node, AgentNodeConfig)
            else NodeTaskState.RUNNING
        )
        return task.task_id

    def request_acquire(
        self,
        task_id: str,
        input_tokens: int,
        created_at: float,
    ) -> str:
        task = self._task(task_id)
        if task.node_kind != "agent" or task.state != NodeTaskState.ACQUIRING:
            raise ValueError(f"task is not an acquiring agent task: {task_id}")
        if any(
            pending.task_id == task_id for pending in self.pending_acquires.values()
        ):
            raise ValueError(f"task already has a pending acquire: {task_id}")

        acquire_id = str(uuid4())
        pending = PendingAcquire(
            acquire_id=acquire_id,
            task_id=task_id,
            input_tokens=input_tokens,
            created_at=created_at,
            request_seq=self._next_request_seq,
        )
        self.pending_acquires[acquire_id] = pending
        self.pending_order.append(acquire_id)
        self._next_request_seq += 1
        return acquire_id

    def poll_grant(self, acquire_id: str) -> GrantInfo | None:
        grant = self.grants.get(acquire_id)
        if grant is not None:
            return grant
        if acquire_id in self.pending_acquires:
            return None
        if acquire_id in self._inactive_acquires:
            session_id = self._inactive_acquires[acquire_id]
            raise SessionInactiveError(f"session is not active: {session_id}")
        raise KeyError(f"unknown acquire: {acquire_id}")

    def cancel_acquire(self, acquire_id: str) -> None:
        if acquire_id in self.grants:
            raise AcquireAlreadyGrantedError(
                f"acquire is already granted: {acquire_id}"
            )
        if acquire_id not in self.pending_acquires:
            raise KeyError(f"unknown acquire: {acquire_id}")
        self._remove_pending(acquire_id)

    def cancel_session(
        self,
        session_id: str,
        error_type: str,
        error_message: str,
    ) -> None:
        session = self._session(session_id)
        if session.state == SessionState.FAILED:
            return
        if session.state == SessionState.COMPLETED:
            raise ValueError(f"completed session cannot be cancelled: {session_id}")

        cancellable = {
            NodeTaskState.PENDING,
            NodeTaskState.ACQUIRING,
            NodeTaskState.EMITTING,
        }
        failed_task = next(
            (
                self.tasks[session.task_ids[node_id]]
                for node_id in self.workflow.graph.topological_order
                if self.tasks[session.task_ids[node_id]].state in cancellable
            ),
            None,
        )
        if failed_task is None:
            self._fail_session(session_id, "")
            return
        failed_task.state = NodeTaskState.FAILED
        failed_task.error_type = error_type
        failed_task.error_message = error_message
        self._fail_session(session_id, failed_task.task_id)

    def cancel_pending(self) -> tuple[str, ...]:
        cancelled: list[str] = []
        for session_id, session in tuple(self.sessions.items()):
            if session.state != SessionState.ACTIVE:
                continue
            self.cancel_session(session_id, "RuntimeStopped", "runtime stopped")
            cancelled.append(session_id)
        return tuple(cancelled)

    def record_runtime_report(self, report: AgentTaskRuntimeReport) -> None:
        if report.gpu_kind not in ("v100", "a100"):
            raise ValueError(f"unsupported GPU kind: {report.gpu_kind}")
        gpu_kind = cast(GpuKind, report.gpu_kind)
        key = (report.node_id, report.model_key, gpu_kind)
        history = self.history.setdefault(key, RuntimeHistoryRecord())
        history.add(report, self.scheduler_config.history_ema_alpha)
        if report.status == "success":
            duration_key = (*key, report.replica_inflight_at_start)
            duration = self.duration_history.get(duration_key)
            if duration is None:
                self.duration_history[duration_key] = DurationHistoryRecord(
                    duration_sec_ema=report.duration_sec
                )
            else:
                duration.add(
                    report.duration_sec,
                    self.scheduler_config.history_ema_alpha,
                )

    def record_running_upstream(
        self,
        session_id: str,
        node_id: str,
        *,
        finish_at: float,
    ) -> None:
        session = self._session(session_id)
        if session.state != SessionState.ACTIVE:
            raise SessionInactiveError(f"session is not active: {session_id}")
        if node_id not in session.task_ids:
            raise KeyError(f"unknown node: {node_id}")
        task = self.tasks[session.task_ids[node_id]]
        if task.state != NodeTaskState.RUNNING:
            raise ValueError(f"task is not running: {session_id}/{node_id}")
        self.running_estimates[(session_id, node_id)] = RunningTaskEstimate(
            session_id=session_id,
            node_id=node_id,
            finish_at=finish_at,
        )

    def near_ready_tasks(self) -> tuple[NearReadyTask, ...]:
        return tuple(self._collect_near_ready_tasks())

    def tick_once(
        self,
        now: float,
    ) -> list[LoadReplicaAction | EvictReplicaAction]:
        suspect_actions: list[LoadReplicaAction | EvictReplicaAction] = []
        suspect_actions.extend(self._start_suspect_cleanup())
        if suspect_actions:
            self._validate_resource_ledger()
            return suspect_actions

        decisions = self._ready_decisions()
        granted_ready = False
        for acquire_id in tuple(decisions):
            pending = self.pending_acquires.get(acquire_id)
            if pending is None:
                continue
            replica = self._replica_for_decision(pending, decisions[acquire_id])
            if replica is None:
                continue
            prediction = self._admission_prediction(
                decisions[acquire_id],
                replica,
            )
            if prediction is not None:
                self._grant(
                    pending,
                    decisions[acquire_id],
                    prediction,
                    replica,
                    now,
                )
                granted_ready = True

        actions: list[LoadReplicaAction | EvictReplicaAction] = []
        blocked_ready: list[tuple[PendingAcquire, PlacementDecision]] = []
        for acquire_id in tuple(decisions):
            pending = self.pending_acquires.get(acquire_id)
            if pending is None:
                continue
            decision = decisions[acquire_id]
            if self._replica_for_decision(pending, decision) is not None:
                continue
            accelerator = self._free_accelerator(decision)
            if accelerator is None:
                blocked_ready.append((pending, decision))
                continue
            actions.append(self._start_ready_load(pending, decision, accelerator, now))
        if actions:
            self._validate_resource_ledger()
            return actions
        if granted_ready:
            self._validate_resource_ledger()
            return []

        near = self._collect_near_ready_tasks()
        protected_pairs = self._ready_pairs(decisions)
        eviction_actions: list[LoadReplicaAction | EvictReplicaAction] = []
        for pending, decision in blocked_ready:
            if pending.acquire_id not in self.pending_acquires:
                continue
            eviction = self._start_eviction_for_decision(
                decision,
                reason="ready_load",
                now=now,
                protected_pairs=protected_pairs,
                near=near,
            )
            if eviction is not None:
                eviction_actions.append(eviction)
        if eviction_actions:
            self._validate_resource_ledger()
            return eviction_actions

        policy_actions: list[LoadReplicaAction | EvictReplicaAction] = []
        for candidate in near:
            if now < candidate.prefetch_at:
                continue
            if self._replica_for_near(candidate) is not None:
                continue
            accelerator = self._free_accelerator(candidate.decision)
            if accelerator is not None:
                policy_actions.append(
                    self._start_prefetch_load(candidate, accelerator, now)
                )
                continue
            eviction = self._start_eviction_for_decision(
                candidate.decision,
                reason="near_ready_prefetch",
                now=now,
                protected_pairs=protected_pairs,
                near=near,
                prefetch_candidate=candidate,
            )
            if eviction is not None:
                policy_actions.append(eviction)
        self._validate_resource_ledger()
        return policy_actions

    def complete_load(
        self,
        replica_id: str,
        result: ReplicaLoadResult,
        *,
        backend_handle: object,
        now: float,
    ) -> None:
        replica = self._replica(replica_id)
        if replica.state != ModelReplicaState.LOADING:
            raise ValueError(f"replica is not loading: {replica_id}")
        if len(replica.accelerator_ids) != 1:
            raise RuntimeError("current model replicas require one accelerator")
        accelerator = self.accelerators[replica.accelerator_ids[0]]
        if backend_handle is None:
            raise ValueError("loaded replica requires a backend handle")
        physical_gpu_id = _normalize_physical_gpu_id(result.physical_gpu_id)
        if accelerator.replica_id != replica_id:
            raise RuntimeError("accelerator ownership changed during model load")
        self._bind_reported_accelerator(replica, accelerator, physical_gpu_id)

        replica.backend_handle = backend_handle
        replica.physical_gpu_id = physical_gpu_id
        replica.load_duration_sec = result.duration_sec
        replica.idle_vram_mb = result.idle_vram_mb
        replica.vllm_version = result.vllm_version
        replica.engine_mode = result.engine_mode
        replica.attention_backend = result.attention_backend
        replica.block_size = result.block_size
        replica.num_gpu_blocks = result.num_gpu_blocks
        replica.gpu_kv_tokens = result.gpu_kv_tokens
        replica.idle_since = now
        replica.state = ModelReplicaState.IDLE
        load_key = (replica.model_key, replica.gpu_kind)
        load_history = self.load_history.get(load_key)
        if load_history is None:
            self.load_history[load_key] = DurationHistoryRecord(
                duration_sec_ema=result.duration_sec
            )
        else:
            load_history.add(
                result.duration_sec,
                self.scheduler_config.history_ema_alpha,
            )
        self._validate_resource_ledger()

    def _bind_reported_accelerator(
        self,
        replica: ModelReplicaRecord,
        reserved: AcceleratorRecord,
        physical_gpu_id: int,
    ) -> None:
        reported = next(
            (
                accelerator
                for accelerator in self.accelerators.values()
                if accelerator.config.hostname == reserved.config.hostname
                and accelerator.config.local_index == physical_gpu_id
            ),
            None,
        )
        if reported is None or reported.config.gpu_kind != replica.gpu_kind:
            raise ValueError(
                "reported physical GPU does not match a configured accelerator"
            )
        if reported is reserved:
            return

        reported_owner_id = reported.replica_id
        reported_owner = (
            self.replicas.get(reported_owner_id)
            if reported_owner_id is not None
            else None
        )
        if reported_owner_id is not None and (
            reported_owner is None
            or reported_owner.state != ModelReplicaState.LOADING
            or reported_owner.accelerator_ids != (reported.config.accelerator_id,)
        ):
            raise ValueError("reported physical GPU is already occupied")

        reserved.replica_id = reported_owner_id
        if reported_owner is not None:
            reported_owner.accelerator_ids = (reserved.config.accelerator_id,)
        reported.replica_id = replica.replica_id
        replica.accelerator_ids = (reported.config.accelerator_id,)

    def fail_load(
        self,
        replica_id: str,
        error_type: str,
        error_message: str,
    ) -> None:
        replica = self._replica(replica_id)
        if replica.state != ModelReplicaState.LOADING:
            raise ValueError(f"replica is not loading: {replica_id}")
        pair = (replica.model_key, replica.gpu_kind)
        if self._replica_pairs.get(pair) != replica_id:
            raise RuntimeError("replica pair changed during model load")
        for accelerator_id in replica.accelerator_ids:
            if self.accelerators[accelerator_id].replica_id != replica_id:
                raise RuntimeError("accelerator ownership changed during model load")

        del self._replica_pairs[pair]
        for accelerator_id in replica.accelerator_ids:
            self.accelerators[accelerator_id].replica_id = None
        del self.replicas[replica_id]
        self._validate_resource_ledger()

    def request_eviction(self, replica_id: str) -> EvictReplicaAction:
        replica = self._replica(replica_id)
        return self._mark_evicting(
            replica,
            reason="requested",
            reuse_distance_sec=None,
            reload_cost_sec=self._reload_cost(replica),
        )

    def complete_eviction(self, replica_id: str) -> None:
        replica = self._replica(replica_id)
        if replica.state != ModelReplicaState.EVICTING:
            raise ValueError(f"replica is not evicting: {replica_id}")
        pair = (replica.model_key, replica.gpu_kind)
        if self._replica_pairs.get(pair) != replica_id:
            raise RuntimeError("replica pair changed during eviction")
        accelerator_ids = replica.accelerator_ids
        for accelerator_id in accelerator_ids:
            if self.accelerators[accelerator_id].replica_id != replica_id:
                raise RuntimeError("accelerator ownership changed during eviction")

        del self._replica_pairs[pair]
        for accelerator_id in accelerator_ids:
            self.accelerators[accelerator_id].replica_id = None
        del self.replicas[replica_id]
        self._validate_resource_ledger()

    def fail_eviction(
        self,
        replica_id: str,
        error_type: str,
        error_message: str,
    ) -> None:
        replica = self._replica(replica_id)
        if replica.state != ModelReplicaState.EVICTING:
            raise ValueError(f"replica is not evicting: {replica_id}")
        replica.eviction_error_type = error_type
        replica.eviction_error_message = error_message
        replica.state = ModelReplicaState.SUSPECT
        self._validate_resource_ledger()

    def complete(
        self,
        task_id: str,
        runtime_report: FunctionTaskRuntimeReport | AgentTaskRuntimeReport,
        acquire_id: str | None = None,
    ) -> CompleteDecision:
        if isinstance(runtime_report, AgentTaskRuntimeReport):
            if self.predictions is None:
                raise NotImplementedError(
                    "agent completion belongs to the resource lifecycle"
                )
            return self._complete_agent(
                task_id,
                runtime_report,
                acquire_id,
            )
        if acquire_id is not None:
            raise ValueError("function completion cannot include an acquire id")

        task = self._task(task_id)
        if task.node_kind != "function":
            raise ValueError(f"task is not a function task: {task_id}")
        if task.state != NodeTaskState.RUNNING:
            raise ValueError(f"function task is not running: {task_id}")
        self._validate_runtime_report(task, runtime_report)

        self.running_estimates.pop((task.session_id, task.node_id), None)
        task.runtime_report = runtime_report
        session = self.sessions[task.session_id]
        if session.state == SessionState.FAILED:
            task.state = NodeTaskState.CANCELLED
            return CompleteDecision(emit_output=False)
        if runtime_report.status == "failed":
            task.state = NodeTaskState.FAILED
            task.error_type = runtime_report.error_type
            task.error_message = runtime_report.error_message
            self._fail_session(task.session_id, task.task_id)
            return CompleteDecision(emit_output=False)

        if session.state != SessionState.ACTIVE:
            raise SessionInactiveError(f"session is not active: {task.session_id}")

        task.state = NodeTaskState.EMITTING
        return CompleteDecision(emit_output=True)

    def finish_node(self, task_id: str, output_report: OutputReport) -> None:
        task = self._task(task_id)
        if task.state == NodeTaskState.CANCELLED:
            raise TaskCancelledError(f"task was cancelled: {task_id}")
        if task.state != NodeTaskState.EMITTING:
            raise ValueError(f"task is not emitting: {task_id}")
        if output_report.task_id != task_id:
            raise ValueError("output report does not match task")
        if output_report.output_items:
            if [item.item_id for item in output_report.output_items] != (
                output_report.output_item_ids
            ):
                raise ValueError("output item reports do not match output ids")
            successors = set(self.workflow.graph.adjacency[task.node_id])
            if any(
                item.session_id != task.session_id
                or item.source_node != task.node_id
                or item.target_node not in successors
                for item in output_report.output_items
            ):
                raise ValueError("output item report does not match task routing")
        is_terminal = task.node_id == self.workflow.graph.terminal_node
        if is_terminal and not output_report.persisted_terminal_result:
            raise ValueError("terminal result was not persisted")

        task.output_report = output_report
        task.state = NodeTaskState.COMPLETED
        if is_terminal:
            self.sessions[task.session_id].state = SessionState.COMPLETED

    def fail_node(
        self,
        task_id: str,
        error_type: str,
        error_message: str,
    ) -> None:
        task = self._task(task_id)
        if task.state == NodeTaskState.CANCELLED:
            raise TaskCancelledError(f"task was cancelled: {task_id}")
        if task.state in {
            NodeTaskState.COMPLETED,
            NodeTaskState.FAILED,
            NodeTaskState.CANCELLED,
        }:
            raise ValueError(f"task is already terminal: {task_id}")

        task.state = NodeTaskState.FAILED
        task.error_type = error_type
        task.error_message = error_message
        self.running_estimates.pop((task.session_id, task.node_id), None)
        self._fail_session(task.session_id, task_id)

    def take_actions(self) -> list[CancelSessionAction]:
        actions = self._actions
        self._actions = []
        return actions

    def drain_complete(self) -> bool:
        if any(
            session.state not in {SessionState.COMPLETED, SessionState.FAILED}
            for session in self.sessions.values()
        ):
            return False
        if self.pending_acquires:
            return False
        active_states = {
            NodeTaskState.ACQUIRING,
            NodeTaskState.RUNNING,
            NodeTaskState.EMITTING,
        }
        return not any(task.state in active_states for task in self.tasks.values())

    def _session(self, session_id: str) -> SessionRecord:
        try:
            return self.sessions[session_id]
        except KeyError:
            raise KeyError(f"unknown session: {session_id}") from None

    def _task(self, task_id: str) -> NodeTaskRecord:
        try:
            return self.tasks[task_id]
        except KeyError:
            raise KeyError(f"unknown task: {task_id}") from None

    def _dependencies_completed(
        self,
        session: SessionRecord,
        node_id: str,
    ) -> bool:
        return all(
            self.tasks[session.task_ids[dependency]].state
            in {NodeTaskState.EMITTING, NodeTaskState.COMPLETED}
            for dependency in self.workflow.graph.dependencies[node_id]
        )

    @staticmethod
    def _validate_runtime_report(
        task: NodeTaskRecord,
        report: FunctionTaskRuntimeReport,
    ) -> None:
        identity = (
            report.task_id,
            report.session_id,
            report.node_id,
            report.input_item_ids,
        )
        expected = (
            task.task_id,
            task.session_id,
            task.node_id,
            task.input_item_ids,
        )
        if identity != expected:
            raise ValueError("runtime report does not match task")

    def _ready_decisions(self) -> dict[str, PlacementDecision]:
        if not self.pending_acquires:
            return {}
        predictions = self.predictions
        if predictions is None:
            raise RuntimeError("agent scheduling requires a prediction cache")

        pending_values = tuple(self.pending_acquires.values())
        ordered = sorted(
            pending_values,
            key=(
                (lambda pending: pending.request_seq)
                if self.scheduler_config.policy == "fifo"
                else lambda pending: (
                    self._node_order[self.tasks[pending.task_id].node_id],
                    pending.request_seq,
                )
            ),
        )
        decisions: dict[str, PlacementDecision] = {}
        for snapshot in ordered:
            pending = self.pending_acquires.get(snapshot.acquire_id)
            if pending is None:
                continue
            task = self.tasks[pending.task_id]
            if self.sessions[task.session_id].state != SessionState.ACTIVE:
                continue
            node = self._nodes[task.node_id]
            if not isinstance(node, AgentNodeConfig):
                raise RuntimeError("pending acquire belongs to a function task")
            decision = self._placement_decision(
                node,
                pending.input_tokens,
                predictions,
            )
            if not decision.feasible:
                task.state = NodeTaskState.FAILED
                task.error_type = "RequestInfeasible"
                task.error_message = decision.reason
                self._fail_session(task.session_id, task.task_id)
                continue
            decisions[pending.acquire_id] = decision
        return decisions

    def _placement_decision(
        self,
        node: AgentNodeConfig,
        input_tokens: int,
        predictions: PredictionCache,
    ) -> PlacementDecision:
        return select_placement(
            node=node,
            input_tokens=input_tokens,
            accelerators=tuple(record.config for record in self.accelerators.values()),
            predictions=predictions,
            oom_penalties=self.oom_penalties,
            eps_mem_mb=self.scheduler_config.eps_mem_mb,
        )

    def _collect_near_ready_tasks(self) -> list[NearReadyTask]:
        if self.scheduler_config.policy == "fifo":
            return []
        predictions = self.predictions
        if predictions is None:
            return []
        near: list[NearReadyTask] = []
        for session in self.sessions.values():
            if session.state != SessionState.ACTIVE:
                continue
            for node_id in self.workflow.graph.topological_order:
                node = self._nodes[node_id]
                task = self.tasks[session.task_ids[node_id]]
                if not isinstance(node, AgentNodeConfig):
                    continue
                if task.state != NodeTaskState.PENDING:
                    continue
                dependencies = self.workflow.graph.dependencies[node_id]
                if not dependencies:
                    continue
                estimates: list[float] = []
                eligible = True
                for dependency in dependencies:
                    dependency_task = self.tasks[session.task_ids[dependency]]
                    if dependency_task.state == NodeTaskState.COMPLETED:
                        continue
                    estimate = self.running_estimates.get(
                        (session.session_id, dependency)
                    )
                    if (
                        dependency_task.state != NodeTaskState.RUNNING
                        or estimate is None
                    ):
                        eligible = False
                        break
                    estimates.append(estimate.finish_at)
                if not eligible or not estimates:
                    continue
                input_tokens = self._estimate_near_input_tokens(node, dependencies)
                decision = self._placement_decision(node, input_tokens, predictions)
                if not decision.feasible:
                    continue
                gpu_kind = decision.gpu_kind
                if gpu_kind is None:
                    raise RuntimeError("feasible placement has no GPU kind")
                load_sec = decision.predicted_load_sec
                if load_sec is None:
                    raise RuntimeError("feasible placement has no load prediction")
                deployment = ModelDeploymentConfig.from_node(node)
                if self.scheduler_config.policy == "history":
                    load_history = self.load_history.get(
                        (deployment.model_key, gpu_kind)
                    )
                    if load_history is None:
                        continue
                    load_sec = load_history.duration_sec_ema
                upstream_eta = max(estimates)
                near.append(
                    NearReadyTask(
                        session_id=session.session_id,
                        node_id=node_id,
                        estimated_input_tokens=input_tokens,
                        upstream_eta=upstream_eta,
                        prefetch_at=(
                            upstream_eta - load_sec - self.scheduler_config.eps_time_sec
                        ),
                        load_sec=load_sec,
                        deployment=deployment,
                        decision=decision,
                    )
                )
        return sorted(
            near,
            key=lambda candidate: (
                candidate.upstream_eta,
                self._node_order[candidate.node_id],
                candidate.session_id,
            ),
        )

    def _estimate_near_input_tokens(
        self,
        node: AgentNodeConfig,
        dependencies: tuple[str, ...],
    ) -> int:
        estimated_outputs = 0
        for dependency in dependencies:
            samples = (
                [
                    history.output_tokens_p90
                    for (history_node_id, _, _), history in self.history.items()
                    if history_node_id == dependency and history.samples
                ]
                if self.scheduler_config.policy == "history"
                else []
            )
            estimated_outputs += (
                max(samples) if samples else node.execution.max_new_tokens
            )
        prompt_overhead = max(1, len(node.prompt_template.split()))
        return max(1, estimated_outputs + prompt_overhead)

    def _replica_for_decision(
        self,
        pending: PendingAcquire,
        decision: PlacementDecision,
    ) -> ModelReplicaRecord | None:
        gpu_kind = decision.gpu_kind
        if gpu_kind is None:
            raise RuntimeError("feasible placement has no GPU kind")
        task = self.tasks[pending.task_id]
        node = self._nodes[task.node_id]
        if not isinstance(node, AgentNodeConfig):
            raise RuntimeError("pending acquire belongs to a function task")
        model_key = ModelDeploymentConfig.from_node(node).model_key
        replica_id = self._replica_pairs.get((model_key, gpu_kind))
        return self.replicas.get(replica_id) if replica_id is not None else None

    def _free_accelerator(
        self,
        decision: PlacementDecision,
    ) -> AcceleratorRecord | None:
        gpu_kind = decision.gpu_kind
        effective_vram_mb = decision.effective_vram_mb
        if gpu_kind is None or effective_vram_mb is None:
            raise RuntimeError("feasible placement is incomplete")
        candidates = [
            accelerator
            for accelerator in self.accelerators.values()
            if accelerator.config.gpu_kind == gpu_kind
            and accelerator.replica_id is None
            and accelerator.config.total_mem_mb >= effective_vram_mb
        ]
        return min(
            candidates,
            key=lambda accelerator: (
                accelerator.config.total_mem_mb,
                accelerator.config.accelerator_id,
            ),
            default=None,
        )

    def _start_ready_load(
        self,
        pending: PendingAcquire,
        decision: PlacementDecision,
        accelerator: AcceleratorRecord,
        now: float,
    ) -> LoadReplicaAction:
        gpu_kind = decision.gpu_kind
        if gpu_kind is None:
            raise RuntimeError("feasible placement has no GPU kind")
        task = self.tasks[pending.task_id]
        node = self._nodes[task.node_id]
        if not isinstance(node, AgentNodeConfig):
            raise RuntimeError("pending acquire belongs to a function task")
        deployment = ModelDeploymentConfig.from_node(node)
        pair = (deployment.model_key, gpu_kind)
        if pair in self._replica_pairs:
            raise RuntimeError(f"replica pair already exists: {pair}")
        if accelerator.replica_id is not None:
            raise RuntimeError("accelerator is already reserved")
        expected_load_sec = decision.predicted_load_sec
        if expected_load_sec is None:
            raise RuntimeError("feasible placement has no load prediction")
        expected_load_sec = self._load_cost(
            deployment,
            gpu_kind,
            expected_load_sec,
        )

        replica_id = str(uuid4())
        replica = ModelReplicaRecord(
            replica_id=replica_id,
            deployment=deployment,
            model_key=deployment.model_key,
            gpu_kind=gpu_kind,
            accelerator_ids=(accelerator.config.accelerator_id,),
            state=ModelReplicaState.LOADING,
            expected_load_sec=expected_load_sec,
            created_at=now,
        )
        accelerator.replica_id = replica_id
        self.replicas[replica_id] = replica
        self._replica_pairs[pair] = replica_id
        self._validate_resource_ledger()
        return LoadReplicaAction(
            replica_id=replica_id,
            deployment=deployment,
            accelerator=accelerator.config,
            reason="ready_load",
            expected_load_sec=expected_load_sec,
        )

    def _start_prefetch_load(
        self,
        near: NearReadyTask,
        accelerator: AcceleratorRecord,
        now: float,
    ) -> LoadReplicaAction:
        gpu_kind = near.decision.gpu_kind
        if gpu_kind is None:
            raise RuntimeError("feasible placement has no GPU kind")
        pair = (near.deployment.model_key, gpu_kind)
        if pair in self._replica_pairs:
            raise RuntimeError(f"replica pair already exists: {pair}")
        if accelerator.replica_id is not None:
            raise RuntimeError("accelerator is already reserved")

        replica_id = str(uuid4())
        replica = ModelReplicaRecord(
            replica_id=replica_id,
            deployment=near.deployment,
            model_key=near.deployment.model_key,
            gpu_kind=gpu_kind,
            accelerator_ids=(accelerator.config.accelerator_id,),
            state=ModelReplicaState.LOADING,
            expected_load_sec=near.load_sec,
            created_at=now,
        )
        accelerator.replica_id = replica_id
        self.replicas[replica_id] = replica
        self._replica_pairs[pair] = replica_id
        self._validate_resource_ledger()
        return LoadReplicaAction(
            replica_id=replica_id,
            deployment=near.deployment,
            accelerator=accelerator.config,
            reason="near_ready_prefetch",
            expected_load_sec=near.load_sec,
            session_id=near.session_id,
            node_id=near.node_id,
            prefetch_at=near.prefetch_at,
        )

    def _replica_for_near(
        self,
        near: NearReadyTask,
    ) -> ModelReplicaRecord | None:
        gpu_kind = near.decision.gpu_kind
        if gpu_kind is None:
            raise RuntimeError("feasible placement has no GPU kind")
        replica_id = self._replica_pairs.get((near.deployment.model_key, gpu_kind))
        return self.replicas.get(replica_id) if replica_id is not None else None

    def _ready_pairs(
        self,
        decisions: dict[str, PlacementDecision],
    ) -> set[tuple[str, GpuKind]]:
        pairs: set[tuple[str, GpuKind]] = set()
        for acquire_id, decision in decisions.items():
            pending = self.pending_acquires.get(acquire_id)
            if pending is None:
                continue
            gpu_kind = decision.gpu_kind
            if gpu_kind is None:
                raise RuntimeError("feasible placement has no GPU kind")
            task = self.tasks[pending.task_id]
            node = self._nodes[task.node_id]
            if not isinstance(node, AgentNodeConfig):
                raise RuntimeError("pending acquire belongs to a function task")
            pairs.add((ModelDeploymentConfig.from_node(node).model_key, gpu_kind))
        return pairs

    def _start_suspect_cleanup(self) -> list[EvictReplicaAction]:
        actions: list[EvictReplicaAction] = []
        for replica in sorted(self.replicas.values(), key=lambda item: item.replica_id):
            if replica.state != ModelReplicaState.SUSPECT:
                continue
            if replica.active_acquire_ids:
                continue
            actions.append(
                self._mark_evicting(
                    replica,
                    reason="suspect_cleanup",
                    reuse_distance_sec=None,
                    reload_cost_sec=self._reload_cost(replica),
                )
            )
        return actions

    def _start_eviction_for_decision(
        self,
        decision: PlacementDecision,
        *,
        reason: Literal["ready_load", "near_ready_prefetch"],
        now: float,
        protected_pairs: set[tuple[str, GpuKind]],
        near: list[NearReadyTask],
        prefetch_candidate: NearReadyTask | None = None,
    ) -> EvictReplicaAction | None:
        gpu_kind = decision.gpu_kind
        effective_vram_mb = decision.effective_vram_mb
        if gpu_kind is None or effective_vram_mb is None:
            raise RuntimeError("feasible placement is incomplete")
        candidates: list[EvictionCandidate] = []
        for accelerator in self.accelerators.values():
            if accelerator.config.gpu_kind != gpu_kind:
                continue
            if accelerator.config.total_mem_mb < effective_vram_mb:
                continue
            replica_id = accelerator.replica_id
            if replica_id is None:
                continue
            replica = self.replicas[replica_id]
            pair = (replica.model_key, replica.gpu_kind)
            if pair in protected_pairs:
                continue
            if replica.state not in (ModelReplicaState.IDLE, ModelReplicaState.SUSPECT):
                continue
            if replica.active_acquire_ids:
                continue
            candidates.append(
                EvictionCandidate(
                    replica_id=replica_id,
                    state=replica.state,
                    idle_since=(
                        replica.idle_since
                        if replica.idle_since is not None
                        else replica.created_at
                    ),
                    reuse_distance_sec=self._future_reuse_distance(
                        replica,
                        now,
                        near,
                    ),
                    reload_cost_sec=self._reload_cost(replica),
                )
            )
        victim = select_eviction_victim(candidates)
        if victim is None:
            return None
        return self._mark_evicting(
            self.replicas[victim.replica_id],
            reason=reason,
            reuse_distance_sec=victim.reuse_distance_sec,
            reload_cost_sec=victim.reload_cost_sec,
            prefetch_candidate=prefetch_candidate,
        )

    def _mark_evicting(
        self,
        replica: ModelReplicaRecord,
        *,
        reason: Literal[
            "suspect_cleanup",
            "ready_load",
            "near_ready_prefetch",
            "requested",
        ],
        reuse_distance_sec: float | None,
        reload_cost_sec: float | None,
        prefetch_candidate: NearReadyTask | None = None,
    ) -> EvictReplicaAction:
        if replica.state not in (ModelReplicaState.IDLE, ModelReplicaState.SUSPECT):
            raise ValueError(f"replica is not evictable: {replica.replica_id}")
        if replica.active_acquire_ids:
            raise ValueError(f"replica still has active requests: {replica.replica_id}")
        replica.state = ModelReplicaState.EVICTING
        replica.eviction_error_type = None
        replica.eviction_error_message = None
        return EvictReplicaAction(
            replica_id=replica.replica_id,
            accelerator_ids=replica.accelerator_ids,
            reason=reason,
            reuse_distance_sec=reuse_distance_sec,
            reload_cost_sec=reload_cost_sec,
            session_id=(
                prefetch_candidate.session_id
                if prefetch_candidate is not None
                else None
            ),
            node_id=(
                prefetch_candidate.node_id if prefetch_candidate is not None else None
            ),
            prefetch_at=(
                prefetch_candidate.prefetch_at
                if prefetch_candidate is not None
                else None
            ),
        )

    def _future_reuse_distance(
        self,
        replica: ModelReplicaRecord,
        now: float,
        near: list[NearReadyTask],
    ) -> float | None:
        distances = [
            max(0.0, candidate.upstream_eta - now)
            for candidate in near
            if candidate.deployment.model_key == replica.model_key
            and candidate.decision.gpu_kind == replica.gpu_kind
        ]
        if distances:
            return min(distances)
        for session in self.sessions.values():
            if session.state != SessionState.ACTIVE:
                continue
            for node_id, task_id in session.task_ids.items():
                task = self.tasks[task_id]
                node = self._nodes[node_id]
                if task.state != NodeTaskState.PENDING:
                    continue
                if not isinstance(node, AgentNodeConfig):
                    continue
                if ModelDeploymentConfig.from_node(node).model_key == replica.model_key:
                    return None
        return inf

    def _reload_cost(self, replica: ModelReplicaRecord) -> float | None:
        if self.scheduler_config.policy == "fifo":
            return None
        if self.scheduler_config.policy == "cache":
            return replica.expected_load_sec
        history = self.load_history.get((replica.model_key, replica.gpu_kind))
        return history.duration_sec_ema if history is not None else None

    def _load_cost(
        self,
        deployment: ModelDeploymentConfig,
        gpu_kind: GpuKind,
        predicted_load_sec: float,
    ) -> float:
        if self.scheduler_config.policy != "history":
            return predicted_load_sec
        history = self.load_history.get((deployment.model_key, gpu_kind))
        return history.duration_sec_ema if history is not None else predicted_load_sec

    def _grant(
        self,
        pending: PendingAcquire,
        decision: PlacementDecision,
        prediction: PredictionEntry,
        replica: ModelReplicaRecord,
        now: float,
    ) -> None:
        if replica.state not in (ModelReplicaState.IDLE, ModelReplicaState.BUSY):
            raise RuntimeError("replica is not accepting requests")
        if replica.backend_handle is None:
            raise RuntimeError("loaded replica has no backend handle")
        max_new_tokens = decision.max_new_tokens
        gpu_kind = decision.gpu_kind
        if max_new_tokens is None or gpu_kind is None:
            raise RuntimeError("feasible placement decision is incomplete")
        admitted_batch_size = len(replica.active_acquire_ids) + 1
        if prediction.key.batch_size != admitted_batch_size:
            raise RuntimeError("batch prediction does not match replica occupancy")
        task = self.tasks[pending.task_id]
        node = self._nodes[task.node_id]
        if not isinstance(node, AgentNodeConfig):
            raise RuntimeError("pending acquire belongs to a function task")
        grant = GrantInfo(
            acquire_id=pending.acquire_id,
            task_id=pending.task_id,
            replica_id=replica.replica_id,
            backend_handle=replica.backend_handle,
            accelerator_ids=replica.accelerator_ids,
            model_key=replica.model_key,
            gpu_kind=gpu_kind,
            input_tokens=pending.input_tokens,
            max_new_tokens=max_new_tokens,
            admitted_batch_size=admitted_batch_size,
            prediction_key=prediction.key,
        )
        replica.state = ModelReplicaState.BUSY
        replica.active_acquire_ids.add(pending.acquire_id)
        task.state = NodeTaskState.RUNNING
        self._remove_pending(pending.acquire_id)
        self.grants[pending.acquire_id] = grant
        duration: float | None = None
        if self.scheduler_config.policy == "cache":
            duration = prediction.predicted_run_sec
        elif self.scheduler_config.policy == "history":
            history = self.duration_history.get(
                (task.node_id, replica.model_key, gpu_kind, admitted_batch_size)
            )
            if history is not None:
                duration = history.duration_sec_ema
        if duration is not None:
            self.running_estimates[(task.session_id, task.node_id)] = (
                RunningTaskEstimate(
                    session_id=task.session_id,
                    node_id=task.node_id,
                    finish_at=now + duration,
                )
            )
        self._validate_resource_ledger()

    def _admission_prediction(
        self,
        decision: PlacementDecision,
        replica: ModelReplicaRecord,
    ) -> PredictionEntry | None:
        if replica.state not in (ModelReplicaState.IDLE, ModelReplicaState.BUSY):
            return None
        batch_size = len(replica.active_acquire_ids) + 1
        if batch_size > replica.deployment.serving.max_num_seqs:
            return None
        decision_key = decision.prediction_key
        gpu_kind = decision.gpu_kind
        if decision_key is None or gpu_kind is None:
            raise RuntimeError("feasible placement is incomplete")
        active_keys = [
            self.grants[acquire_id].prediction_key
            for acquire_id in replica.active_acquire_ids
        ]
        sequence_length = max(
            [decision_key.sequence_length]
            + [key.sequence_length for key in active_keys]
        )
        output_length = max(
            [decision_key.decode_output_length]
            + [key.decode_output_length for key in active_keys]
        )
        predictions = self.predictions
        if predictions is None:
            raise RuntimeError("agent scheduling requires a prediction cache")
        try:
            prediction = predictions.lookup_decode(
                model_name=replica.deployment.model_name,
                gpu_kind=gpu_kind,
                batch_size=batch_size,
                sequence_length=sequence_length,
                decode_output_length=output_length,
            )
        except KeyError:
            if batch_size == 1:
                raise RuntimeError("batch-1 placement prediction is missing") from None
            return None
        accelerator = self.accelerators[replica.accelerator_ids[0]].config
        effective_vram_mb = (
            prediction.predicted_peak_vram_mb
            + self.scheduler_config.eps_mem_mb
            + self.oom_penalties.get(prediction.key, 0.0)
        )
        if effective_vram_mb > accelerator.total_mem_mb:
            return None
        return prediction

    def _complete_agent(
        self,
        task_id: str,
        report: AgentTaskRuntimeReport,
        acquire_id: str | None,
    ) -> CompleteDecision:
        task = self._task(task_id)
        if task.node_kind != "agent":
            raise ValueError(f"task is not an agent task: {task_id}")
        if task.state != NodeTaskState.RUNNING:
            raise ValueError(f"agent task is not running: {task_id}")
        if acquire_id is None:
            raise ValueError("agent completion requires an acquire id")
        grant = self.grants.get(acquire_id)
        if grant is None:
            raise ValueError(f"agent completion has no active grant: {acquire_id}")
        self._validate_agent_report(task, grant, report, acquire_id)

        self._release_grant(grant, report)
        self.running_estimates.pop((task.session_id, task.node_id), None)
        self.record_runtime_report(report)
        task.runtime_report = report
        task.execution_attempts += 1
        session = self.sessions[task.session_id]
        if session.state == SessionState.FAILED:
            task.state = NodeTaskState.CANCELLED
            return CompleteDecision(emit_output=False)
        if session.state != SessionState.ACTIVE:
            raise SessionInactiveError(f"session is not active: {task.session_id}")
        if report.status == "success":
            task.state = NodeTaskState.EMITTING
            return CompleteDecision(emit_output=True)
        if report.status == "oom":
            self._record_oom(task, grant)
            if task.oom_attempts == 1:
                task.state = NodeTaskState.ACQUIRING
                return CompleteDecision(emit_output=False, retry_acquire=True)
        elif report.status == "failed":
            node = self._nodes[task.node_id]
            if not isinstance(node, AgentNodeConfig):
                raise RuntimeError("agent task has a function node config")
            if task.execution_attempts < node.retry.max_attempts:
                task.state = NodeTaskState.ACQUIRING
                return CompleteDecision(emit_output=False, retry_acquire=True)

        task.state = NodeTaskState.FAILED
        task.error_type = report.error_type
        self._fail_session(task.session_id, task.task_id)
        return CompleteDecision(emit_output=False)

    @staticmethod
    def _validate_agent_report(
        task: NodeTaskRecord,
        grant: GrantInfo,
        report: AgentTaskRuntimeReport,
        acquire_id: str,
    ) -> None:
        identity = (
            report.acquire_id,
            report.task_id,
            report.session_id,
            report.node_id,
            report.input_item_ids,
            report.model_key,
            report.accelerator_id,
            report.gpu_kind,
            report.input_tokens,
            report.max_new_tokens,
        )
        expected = (
            acquire_id,
            task.task_id,
            task.session_id,
            task.node_id,
            task.input_item_ids,
            grant.model_key,
            grant.accelerator_ids[0],
            grant.gpu_kind,
            grant.input_tokens,
            grant.max_new_tokens,
        )
        if (
            identity != expected
            or grant.acquire_id != acquire_id
            or grant.task_id != task.task_id
        ):
            raise ValueError("agent runtime report does not match grant")

    def _release_grant(
        self,
        grant: GrantInfo,
        report: AgentTaskRuntimeReport,
    ) -> None:
        replica = self._replica(grant.replica_id)
        if replica.state not in (ModelReplicaState.BUSY, ModelReplicaState.SUSPECT):
            raise RuntimeError("grant does not own an active replica")
        if grant.acquire_id not in replica.active_acquire_ids:
            raise RuntimeError("grant does not own the busy replica")
        replica.active_acquire_ids.remove(grant.acquire_id)
        if (
            report.status == "oom"
            or report.engine_failed
            or _is_severe_cuda_failure(report)
        ):
            replica.state = ModelReplicaState.SUSPECT
        elif replica.state != ModelReplicaState.SUSPECT:
            replica.state = (
                ModelReplicaState.BUSY
                if replica.active_acquire_ids
                else ModelReplicaState.IDLE
            )
        if not replica.active_acquire_ids:
            replica.idle_since = report.finished_at
        del self.grants[grant.acquire_id]
        self._validate_resource_ledger()

    def _record_oom(self, task: NodeTaskRecord, grant: GrantInfo) -> None:
        task.oom_attempts += 1
        penalty = max(self.scheduler_config.eps_mem_mb, 1.0)
        self.oom_penalties[grant.prediction_key] = (
            self.oom_penalties.get(grant.prediction_key, 0.0) + penalty
        )

    def _replica(self, replica_id: str) -> ModelReplicaRecord:
        try:
            return self.replicas[replica_id]
        except KeyError:
            raise KeyError(f"unknown replica: {replica_id}") from None

    def _remove_pending(self, acquire_id: str) -> None:
        del self.pending_acquires[acquire_id]
        try:
            self.pending_order.remove(acquire_id)
        except ValueError:
            raise RuntimeError("pending acquire order is inconsistent") from None

    def _validate_resource_ledger(self) -> None:
        seen_accelerators: set[str] = set()
        seen_pairs: set[tuple[str, GpuKind]] = set()
        active_acquires: set[str] = set()
        for pair, replica_id in self._replica_pairs.items():
            replica = self.replicas.get(replica_id)
            if replica is None or (replica.model_key, replica.gpu_kind) != pair:
                raise RuntimeError("model replica pairs index is inconsistent")
        for replica in self.replicas.values():
            pair = (replica.model_key, replica.gpu_kind)
            if (
                pair in seen_pairs
                or self._replica_pairs.get(pair) != replica.replica_id
            ):
                raise RuntimeError("model replica pairs must be unique")
            seen_pairs.add(pair)
            if (
                replica.state == ModelReplicaState.BUSY
                and not replica.active_acquire_ids
            ):
                raise RuntimeError("busy replica must have active requests")
            if (
                replica.state not in (ModelReplicaState.BUSY, ModelReplicaState.SUSPECT)
                and replica.active_acquire_ids
            ):
                raise RuntimeError("inactive replica cannot own active requests")
            if active_acquires & replica.active_acquire_ids:
                raise RuntimeError("active request ownership overlaps")
            active_acquires.update(replica.active_acquire_ids)
            for accelerator_id in replica.accelerator_ids:
                if accelerator_id in seen_accelerators:
                    raise RuntimeError("accelerator ownership overlaps")
                seen_accelerators.add(accelerator_id)
                accelerator = self.accelerators.get(accelerator_id)
                if accelerator is None or accelerator.replica_id != replica.replica_id:
                    raise RuntimeError("replica and accelerator ledgers disagree")
        for accelerator_id, accelerator in self.accelerators.items():
            if accelerator.replica_id is None:
                continue
            replica = self.replicas.get(accelerator.replica_id)
            if replica is None or accelerator_id not in replica.accelerator_ids:
                raise RuntimeError("accelerator and replica ledgers disagree")
        if active_acquires != set(self.grants):
            raise RuntimeError("active requests and grant ledger disagree")
        for acquire_id, grant in self.grants.items():
            replica = self.replicas.get(grant.replica_id)
            if replica is None or acquire_id not in replica.active_acquire_ids:
                raise RuntimeError("grant and replica ledgers disagree")

    def _fail_session(self, session_id: str, failed_task_id: str) -> None:
        session = self.sessions[session_id]
        if session.state == SessionState.COMPLETED:
            raise ValueError(f"completed session cannot fail: {session_id}")

        first_failure = session.state == SessionState.ACTIVE
        session.state = SessionState.FAILED
        task_ids = set(session.task_ids.values())
        for acquire_id, pending in list(self.pending_acquires.items()):
            if pending.task_id in task_ids:
                self._inactive_acquires[acquire_id] = session_id
                self._remove_pending(acquire_id)

        cancellable = {
            NodeTaskState.PENDING,
            NodeTaskState.ACQUIRING,
            NodeTaskState.EMITTING,
        }
        for task_id in task_ids:
            task = self.tasks[task_id]
            if task_id != failed_task_id and task.state in cancellable:
                task.state = NodeTaskState.CANCELLED

        if first_failure:
            self._actions.append(CancelSessionAction(session_id=session_id))


ReplicaFactory = Callable[[LoadReplicaAction], object]


@dataclass(slots=True)
class _SchedulerCommand:
    name: str
    args: tuple[object, ...]
    kwargs: dict[str, object]
    future: asyncio.Future[object] | None


class _SchedulerActor:
    def __init__(
        self,
        *,
        workflow: Workflow,
        scheduler_config: SchedulerConfig,
        predictions: PredictionCache | None,
        trace_writer: object,
        replica_factory: object | None = None,
        ray_node_ids: Mapping[str, str] | None = None,
        run_id: str = "workflow-run",
    ) -> None:
        self.core = SchedulerCore(
            workflow,
            scheduler_config=scheduler_config,
            predictions=predictions,
        )
        self.run_id = run_id
        self._config = scheduler_config
        self._trace_writer = trace_writer
        self._replica_factory = replica_factory
        self._ray_node_ids = dict(ray_node_ids or {})
        self._commands: asyncio.Queue[_SchedulerCommand] = asyncio.Queue()
        self._workers: dict[str, object] = {}
        self._replica_handles: dict[str, object] = {}
        self._load_actions: dict[str, LoadReplicaAction] = {}
        self._watcher_tasks: set[asyncio.Task[None]] = set()
        self._manual_actions: list[LoadReplicaAction | EvictReplicaAction] = []
        self._trace_buffer: list[TraceEvent] = []
        self._trace_inflight = False
        self._trace_last_ack = -1
        self._next_event_seq = 0
        self._known_grants: set[str] = set()
        self._fresh_replicas: set[str] = set()
        self._prefetch_skip_keys: set[tuple[str, str, float, str]] = set()
        self._terminal_events: dict[str, asyncio.Event] = {}
        self._latency_required: set[str] = set()
        self._session_latencies: dict[str, float] = {}
        self._run_started = False
        self._loop_finished = False
        self._stop_requested = False
        self._run_finished_recorded = False
        self._force_cleanup_started = False
        self._evict_all_requested = False
        self._run_failure: RuntimeError | None = None
        self._flush_failure_trace = False

    async def run(self) -> None:
        if self._run_started:
            raise RuntimeError("scheduler run loop has already started")
        self._run_started = True
        self._record_trace("run_started")
        self._submit_trace_batch()
        try:
            while True:
                commands = await self._next_command_batch()
                self._apply_command_batch(commands)

                if (
                    self._run_failure is None
                    and not self._stop_requested
                    and self._commands_require_scheduling(commands)
                ):
                    self._run_scheduling_pass()
                if self._run_failure is None:
                    self._dispatch_cancel_actions()
                    if self._evict_all_requested:
                        self._dispatch_evict_all()
                    self._dispatch_manual_actions()
                self._submit_trace_batch()
                if self._run_failure is not None:
                    if not self._flush_failure_trace or self._trace_is_idle():
                        raise self._run_failure
                    continue
                if (
                    self._stop_requested
                    and not self._run_finished_recorded
                    and self._commands.empty()
                    and self._trace_is_idle()
                    and not any(not task.done() for task in self._watcher_tasks)
                ):
                    self._record_trace("run_finished")
                    self._run_finished_recorded = True
                    self._submit_trace_batch()
                if self._can_stop():
                    break
        except Exception as error:
            failure = (
                error
                if isinstance(error, RuntimeError)
                else RuntimeError(f"scheduler run loop failed: {error}")
            )
            self._run_failure = failure
            self._fail_queued_commands(failure)
            if failure is error:
                raise
            raise failure from error
        finally:
            self._loop_finished = True
            await self._cleanup_runtime()

    async def register_session(
        self,
        session_id: str,
        submitted_at: float,
        require_latency: bool = False,
    ) -> None:
        await self._submit(
            "register_session",
            session_id,
            submitted_at,
            require_latency,
        )

    async def register_workers(self, workers: Mapping[str, object]) -> None:
        await self._submit("register_workers", dict(workers))

    async def begin_node(
        self,
        session_id: str,
        node_id: str,
        input_item_ids: list[str],
    ) -> str:
        value = await self._submit(
            "begin_node",
            session_id,
            node_id,
            input_item_ids,
        )
        if not isinstance(value, str):
            raise TypeError("begin_node returned an invalid task id")
        return value

    async def observe_input(self, report: InputTraceReport) -> None:
        await self._submit("observe_input", report)

    async def record_item_enqueued(self, report: ItemTraceReport) -> None:
        await self._submit("record_item_enqueued", report)

    async def request_acquire(
        self,
        task_id: str,
        input_tokens: int,
        created_at: float,
    ) -> str:
        value = await self._submit(
            "request_acquire",
            task_id,
            input_tokens,
            created_at,
        )
        if not isinstance(value, str):
            raise TypeError("request_acquire returned an invalid acquire id")
        return value

    async def cancel_acquire(self, acquire_id: str) -> None:
        await self._submit("cancel_acquire", acquire_id)

    async def complete(
        self,
        task_id: str,
        runtime_report: FunctionTaskRuntimeReport | AgentTaskRuntimeReport,
        acquire_id: str | None = None,
    ) -> CompleteDecision:
        value = await self._submit(
            "complete",
            task_id,
            runtime_report,
            acquire_id,
        )
        if not isinstance(value, CompleteDecision):
            raise TypeError("complete returned an invalid decision")
        return value

    async def finish_node(self, task_id: str, output_report: OutputReport) -> None:
        await self._submit("finish_node", task_id, output_report)

    async def fail_node(
        self,
        task_id: str,
        error_type: str,
        error_message: str,
    ) -> None:
        await self._submit("fail_node", task_id, error_type, error_message)

    async def cancel_session(
        self,
        session_id: str,
        error_type: str,
        error_message: str,
    ) -> None:
        await self._submit(
            "cancel_session",
            session_id,
            error_type,
            error_message,
        )

    async def cancel_pending(self) -> None:
        await self._submit("cancel_pending")

    async def evict_replica(self, replica_id: str) -> None:
        await self._submit("evict_replica", replica_id)

    async def evict_all(self) -> None:
        await self._submit("evict_all")

    async def stop_loop(self) -> None:
        await self._submit("stop_loop")

    async def force_kill_replicas(self) -> None:
        self._force_cleanup_started = True
        self._stop_requested = True
        await self._cleanup_runtime()
        if self._replica_handles:
            await self._kill_replica_handles()

    async def wait_session_terminal(self, session_id: str) -> SessionState:
        try:
            terminal = self._terminal_events[session_id]
        except KeyError:
            raise KeyError(f"unknown session: {session_id}") from None
        await terminal.wait()
        return self.core.sessions[session_id].state

    async def record_session_latency(
        self,
        session_id: str,
        latency_sec: float,
        finished_at: float,
    ) -> None:
        await self._submit(
            "record_session_latency",
            session_id,
            latency_sec,
            finished_at,
        )

    async def poll_grant(self, acquire_id: str) -> GrantInfo | None:
        return self.core.poll_grant(acquire_id)

    async def get_session_state(self, session_id: str) -> SessionState:
        return self.core._session(session_id).state

    async def get_replica_state(
        self,
        replica_id: str,
    ) -> ModelReplicaState | None:
        replica = self.core.replicas.get(replica_id)
        return replica.state if replica is not None else None

    async def drain_complete(self) -> bool:
        return (
            self.core.drain_complete()
            and self._latency_required <= self._session_latencies.keys()
            and self._commands.empty()
            and self._trace_is_idle()
        )

    async def replicas_empty(self) -> bool:
        return not self.core.replicas

    async def trace_idle(self) -> bool:
        return self._trace_is_idle()

    async def get_run_failure(self) -> str | None:
        return str(self._run_failure) if self._run_failure is not None else None

    async def _submit(
        self,
        name: str,
        *args: object,
        **kwargs: object,
    ) -> object:
        if self._run_failure is not None:
            raise self._run_failure
        if self._loop_finished:
            raise RuntimeError("scheduler run loop has stopped")
        if self._stop_requested:
            raise RuntimeError("scheduler run loop is stopping")
        future: asyncio.Future[object] = asyncio.get_running_loop().create_future()
        await self._commands.put(
            _SchedulerCommand(
                name=name,
                args=args,
                kwargs=dict(kwargs),
                future=future,
            )
        )
        return await future

    async def _next_command_batch(self) -> list[_SchedulerCommand]:
        timeout = self._next_wake_timeout()
        try:
            first = await asyncio.wait_for(self._commands.get(), timeout=timeout)
        except TimeoutError:
            return []
        commands = [first]
        while True:
            try:
                commands.append(self._commands.get_nowait())
            except asyncio.QueueEmpty:
                return commands

    def _next_wake_timeout(self) -> float:
        now = time.time()
        future_deadlines = [
            task.prefetch_at
            for task in self.core.near_ready_tasks()
            if task.prefetch_at > now
        ]
        if not future_deadlines:
            return self._config.max_tick_interval_sec
        return min(
            self._config.max_tick_interval_sec,
            max(0.0, min(future_deadlines) - now),
        )

    def _apply_command_batch(self, commands: list[_SchedulerCommand]) -> None:
        for command in commands:
            if self._run_failure is not None:
                if command.future is not None:
                    self._fail_command(command, self._run_failure)
                elif command.name in {"trace_succeeded", "trace_failed"}:
                    self._apply_command_with_ack(command)
                continue
            if self._stop_requested and command.future is not None:
                self._fail_command(
                    command,
                    RuntimeError("scheduler run loop is stopping"),
                )
                continue
            self._apply_command_with_ack(command)

    @staticmethod
    def _commands_require_scheduling(commands: list[_SchedulerCommand]) -> bool:
        return not commands or any(
            command.name not in {"trace_succeeded"} for command in commands
        )

    @staticmethod
    def _fail_command(command: _SchedulerCommand, error: Exception) -> None:
        if command.future is not None and not command.future.done():
            command.future.set_exception(error)

    def _apply_command_with_ack(self, command: _SchedulerCommand) -> None:
        try:
            result = self._apply_command(command)
        except Exception as error:
            if command.future is None:
                self._set_run_failure(error)
                return
            if not command.future.done():
                command.future.set_exception(error)
            return
        if command.future is not None and not command.future.done():
            command.future.set_result(result)

    def _apply_command(self, command: _SchedulerCommand) -> object:
        name = command.name
        args = command.args
        if name == "register_session":
            session_id = cast(str, args[0])
            submitted_at = cast(float, args[1])
            require_latency = cast(bool, args[2])
            self.core.register_session(session_id)
            self._terminal_events[session_id] = asyncio.Event()
            if require_latency:
                self._latency_required.add(session_id)
            self._record_trace(
                "session_submitted",
                ts=submitted_at,
                session_id=session_id,
            )
            return None
        if name == "record_session_latency":
            session_id = cast(str, args[0])
            latency_sec = cast(float, args[1])
            finished_at = cast(float, args[2])
            session = self.core._session(session_id)
            if session.state == SessionState.ACTIVE:
                raise ValueError(f"session is not terminal: {session_id}")
            if latency_sec < 0:
                raise ValueError("session latency must be non-negative")
            if session_id in self._session_latencies:
                raise ValueError(f"session latency is already recorded: {session_id}")
            self._session_latencies[session_id] = latency_sec
            self._record_trace(
                "session_completed"
                if session.state == SessionState.COMPLETED
                else "session_failed",
                ts=finished_at,
                session_id=session_id,
                payload={"latency_sec": latency_sec},
            )
            return None
        if name == "record_item_enqueued":
            report = cast(ItemTraceReport, args[0])
            self._validate_item_trace(
                report.session_id,
                report.source_node,
                report.target_node,
            )
            self._record_trace(
                "item_enqueued",
                ts=report.enqueued_at,
                session_id=report.session_id,
                item_id=report.item_id,
                source_node=report.source_node,
                target_node=report.target_node,
            )
            return None
        if name == "observe_input":
            report = cast(InputTraceReport, args[0])
            self._apply_input_trace(report)
            return None
        if name == "register_workers":
            self._workers = dict(cast(dict[str, object], args[0]))
            return None
        if name == "begin_node":
            session_id = cast(str, args[0])
            node_id = cast(str, args[1])
            input_item_ids = cast(list[str], args[2])
            task_id = self.core.begin_node(session_id, node_id, input_item_ids)
            self._record_trace(
                "task_started",
                session_id=session_id,
                task_id=task_id,
                node_id=node_id,
            )
            return task_id
        if name == "request_acquire":
            task_id = cast(str, args[0])
            input_tokens = cast(int, args[1])
            created_at = cast(float, args[2])
            acquire_id = self.core.request_acquire(
                task_id,
                input_tokens,
                created_at,
            )
            task = self.core.tasks[task_id]
            self._record_trace(
                "acquire_requested",
                ts=created_at,
                session_id=task.session_id,
                task_id=task_id,
                node_id=task.node_id,
                acquire_id=acquire_id,
                payload={"input_tokens": input_tokens},
            )
            return acquire_id
        if name == "cancel_acquire":
            acquire_id = cast(str, args[0])
            pending = self.core.pending_acquires.get(acquire_id)
            self.core.cancel_acquire(acquire_id)
            self._record_trace(
                "acquire_cancelled",
                task_id=pending.task_id if pending is not None else None,
                acquire_id=acquire_id,
            )
            return None
        if name == "complete":
            return self._apply_complete(
                cast(str, args[0]),
                cast(FunctionTaskRuntimeReport | AgentTaskRuntimeReport, args[1]),
                cast(str | None, args[2]),
            )
        if name == "finish_node":
            self._apply_finish(
                cast(str, args[0]),
                cast(OutputReport, args[1]),
            )
            return None
        if name == "fail_node":
            self._apply_fail_node(
                cast(str, args[0]),
                cast(str, args[1]),
                cast(str, args[2]),
            )
            return None
        if name == "cancel_session":
            self._apply_cancel_session(
                cast(str, args[0]),
                cast(str, args[1]),
                cast(str, args[2]),
            )
            return None
        if name == "cancel_pending":
            active = {
                session_id
                for session_id, session in self.core.sessions.items()
                if session.state == SessionState.ACTIVE
            }
            self.core.cancel_pending()
            for session_id in sorted(active):
                self._mark_session_terminal(session_id)
            return None
        if name == "evict_replica":
            self._manual_actions.append(self.core.request_eviction(cast(str, args[0])))
            return None
        if name == "evict_all":
            self._evict_all_requested = True
            return None
        if name == "stop_loop":
            self._stop_requested = True
            return None
        if name == "load_succeeded":
            replica_id = cast(str, args[0])
            result = cast(ReplicaLoadResult, args[1])
            handle = args[2]
            try:
                self.core.complete_load(
                    replica_id,
                    result,
                    backend_handle=handle,
                    now=time.time(),
                )
            except Exception as error:
                self._apply_load_failure(
                    replica_id,
                    type(error).__name__,
                    str(error),
                )
                return None
            self._fresh_replicas.add(replica_id)
            replica = self.core.replicas[replica_id]
            action = self._load_actions.pop(replica_id, None)
            self._record_trace(
                "model_load_finished",
                replica_id=replica_id,
                accelerator_id=replica.accelerator_ids[0],
                gpu_kind=replica.gpu_kind,
                model_key=replica.model_key,
                payload={
                    "duration_sec": result.duration_sec,
                    "vllm_version": result.vllm_version,
                    "engine_mode": result.engine_mode,
                    "attention_backend": result.attention_backend,
                    "max_num_seqs": result.max_num_seqs,
                    "block_size": result.block_size,
                    "num_gpu_blocks": result.num_gpu_blocks,
                    "gpu_kv_tokens": result.gpu_kv_tokens,
                },
            )
            if action is not None and action.reason == "near_ready_prefetch":
                self._record_trace(
                    "prefetch_finished",
                    session_id=action.session_id,
                    node_id=action.node_id,
                    replica_id=replica_id,
                    accelerator_id=replica.accelerator_ids[0],
                    gpu_kind=replica.gpu_kind,
                    model_key=replica.model_key,
                    payload={
                        "prefetch_at": action.prefetch_at,
                        "duration_sec": result.duration_sec,
                    },
                )
            return None
        if name == "load_failed":
            self._apply_load_failure(
                cast(str, args[0]),
                cast(str, args[1]),
                cast(str, args[2]),
            )
            return None
        if name == "eviction_succeeded":
            replica_id = cast(str, args[0])
            replica = self.core.replicas[replica_id]
            self.core.complete_eviction(replica_id)
            self._replica_handles.pop(replica_id, None)
            self._fresh_replicas.discard(replica_id)
            self._record_trace(
                "model_evicted",
                replica_id=replica_id,
                accelerator_id=replica.accelerator_ids[0],
                gpu_kind=replica.gpu_kind,
                model_key=replica.model_key,
            )
            return None
        if name == "eviction_failed":
            replica_id = cast(str, args[0])
            error_type = cast(str, args[1])
            error_message = cast(str, args[2])
            replica = self.core.replicas[replica_id]
            self.core.fail_eviction(replica_id, error_type, error_message)
            self._record_trace(
                "model_eviction_failed",
                replica_id=replica_id,
                accelerator_id=replica.accelerator_ids[0],
                gpu_kind=replica.gpu_kind,
                model_key=replica.model_key,
                payload={
                    "error_type": error_type,
                    "error_message": error_message,
                },
            )
            self._set_run_failure(
                RuntimeError(f"model eviction failed: {error_type}: {error_message}"),
                flush_trace=True,
            )
            return None
        if name == "worker_cancel_failed":
            raise RuntimeError(cast(str, args[0]))
        if name == "trace_succeeded":
            acknowledged = cast(int, args[0])
            expected = cast(int, args[1])
            if acknowledged != expected:
                raise RuntimeError(
                    f"trace acknowledgement mismatch: {acknowledged} != {expected}"
                )
            self._trace_last_ack = acknowledged
            self._trace_inflight = False
            return None
        if name == "trace_failed":
            self._trace_inflight = False
            raise RuntimeError(f"trace append failed: {args[0]}")
        raise ValueError(f"unknown scheduler command: {name}")

    def _apply_input_trace(self, report: InputTraceReport) -> None:
        self._validate_item_trace(
            report.session_id,
            report.source_node,
            report.target_node,
        )
        self._record_trace(
            "item_dequeued",
            ts=report.dequeued_at,
            session_id=report.session_id,
            item_id=report.item_id,
            source_node=report.source_node,
            target_node=report.target_node,
            node_id=report.target_node,
        )
        if report.fanin_state == "wait":
            self._record_trace(
                "fanin_wait",
                ts=report.dequeued_at,
                session_id=report.session_id,
                item_id=report.item_id,
                source_node=report.source_node,
                target_node=report.target_node,
                node_id=report.target_node,
                payload={
                    "waiting_for_sources": list(report.waiting_for_sources),
                },
            )
            return
        self._record_trace(
            "fanin_ready",
            ts=report.dequeued_at,
            session_id=report.session_id,
            item_id=report.item_id,
            source_node=report.source_node,
            target_node=report.target_node,
            node_id=report.target_node,
            payload={
                "source_nodes": list(report.source_nodes),
                "input_item_ids": list(report.input_item_ids),
            },
        )

    def _validate_item_trace(
        self,
        session_id: str,
        source_node: str | None,
        target_node: str,
    ) -> None:
        self.core._session(session_id)
        if source_node is None:
            if target_node != self.core.workflow.graph.entry_node:
                raise ValueError("external item must target the workflow entry")
            return
        if target_node not in self.core.workflow.graph.adjacency[source_node]:
            raise ValueError("item trace does not match a workflow edge")

    def _apply_load_failure(
        self,
        replica_id: str,
        error_type: str,
        error_message: str,
    ) -> None:
        replica = self.core.replicas[replica_id]
        accelerator_id = replica.accelerator_ids[0]
        gpu_kind = replica.gpu_kind
        model_key = replica.model_key
        self.core.fail_load(replica_id, error_type, error_message)
        self._load_actions.pop(replica_id, None)
        self._record_trace(
            "model_load_failed",
            replica_id=replica_id,
            accelerator_id=accelerator_id,
            gpu_kind=gpu_kind,
            model_key=model_key,
            payload={
                "error_type": error_type,
                "error_message": error_message,
            },
        )
        self._set_run_failure(
            RuntimeError(f"model load failed: {error_type}: {error_message}"),
            flush_trace=True,
        )

    def _apply_complete(
        self,
        task_id: str,
        report: FunctionTaskRuntimeReport | AgentTaskRuntimeReport,
        acquire_id: str | None,
    ) -> CompleteDecision:
        before_session = self.core.sessions[report.session_id].state
        grant = self.core.grants.get(acquire_id) if acquire_id is not None else None
        decision = self.core.complete(task_id, report, acquire_id)
        task = self.core.tasks[task_id]
        payload: dict[str, object] = {
            "status": report.status,
            "duration_sec": report.duration_sec,
            "started_at": report.started_at,
            "finished_at": report.finished_at,
        }
        if isinstance(report, AgentTaskRuntimeReport):
            if grant is None or self.core.predictions is None:
                raise RuntimeError("agent completion requires prediction metadata")
            prediction = self.core.predictions.lookup(grant.prediction_key)
            payload.update(
                {
                    "input_tokens": report.input_tokens,
                    "max_new_tokens": report.max_new_tokens,
                    "output_tokens": report.output_tokens,
                    "hit_token_limit": report.hit_token_limit,
                    "finish_reason": report.finish_reason,
                    "queue_time_sec": report.queue_time_sec,
                    "time_to_first_token_sec": report.time_to_first_token_sec,
                    "replica_inflight_at_start": (report.replica_inflight_at_start),
                    "engine_failed": report.engine_failed,
                    "admitted_batch_size": grant.admitted_batch_size,
                    "prediction_key": prediction.key.model_dump(mode="json"),
                    "predicted_load_sec": prediction.predicted_load_sec,
                    "predicted_run_sec": prediction.predicted_run_sec,
                    "predicted_peak_vram_mb": prediction.predicted_peak_vram_mb,
                    "predicted_power_watts": prediction.predicted_power_watts,
                }
            )
        else:
            payload["input_item_count"] = len(report.input_item_ids)
        if report.error_type is not None:
            payload["error_type"] = report.error_type
        if isinstance(report, AgentTaskRuntimeReport) and grant is not None:
            self._record_trace(
                "task_execution_finished",
                ts=report.finished_at,
                session_id=task.session_id,
                task_id=task_id,
                node_id=task.node_id,
                acquire_id=acquire_id,
                replica_id=grant.replica_id,
                accelerator_id=grant.accelerator_ids[0],
                gpu_kind=grant.gpu_kind,
                model_key=grant.model_key,
                payload=payload,
            )
        else:
            self._record_trace(
                "task_execution_finished",
                ts=report.finished_at,
                session_id=task.session_id,
                task_id=task_id,
                node_id=task.node_id,
                acquire_id=acquire_id,
                payload=payload,
            )
        if decision.retry_acquire:
            self._record_trace(
                "task_retried",
                ts=report.finished_at,
                session_id=task.session_id,
                task_id=task_id,
                node_id=task.node_id,
                acquire_id=acquire_id,
                payload={**payload, "execution_event_recorded": True},
            )
        elif task.state == NodeTaskState.FAILED:
            self._record_trace(
                "task_failed",
                ts=report.finished_at,
                session_id=task.session_id,
                task_id=task_id,
                node_id=task.node_id,
                acquire_id=acquire_id,
                payload={**payload, "execution_event_recorded": True},
            )
        if (
            isinstance(report, AgentTaskRuntimeReport)
            and grant is not None
            and self.core.replicas[grant.replica_id].state == ModelReplicaState.SUSPECT
        ):
            self._record_trace(
                "model_suspect",
                ts=report.finished_at,
                session_id=task.session_id,
                task_id=task_id,
                node_id=task.node_id,
                acquire_id=acquire_id,
                replica_id=grant.replica_id,
                accelerator_id=grant.accelerator_ids[0],
                gpu_kind=grant.gpu_kind,
                model_key=grant.model_key,
                payload={"status": report.status, "error_type": report.error_type},
            )
        if (
            before_session == SessionState.ACTIVE
            and self.core.sessions[task.session_id].state == SessionState.FAILED
        ):
            self._mark_session_terminal(task.session_id)
        return decision

    def _apply_finish(self, task_id: str, report: OutputReport) -> None:
        task = self.core.tasks[task_id]
        self.core.finish_node(task_id, report)
        for item in report.output_items:
            self._record_trace(
                "item_emitted",
                ts=item.emitted_at,
                session_id=item.session_id,
                item_id=item.item_id,
                source_node=item.source_node,
                target_node=item.target_node,
                node_id=task.node_id,
            )
            self._record_trace(
                "item_enqueued",
                ts=item.enqueued_at,
                session_id=item.session_id,
                item_id=item.item_id,
                source_node=item.source_node,
                target_node=item.target_node,
                node_id=task.node_id,
            )
        runtime = task.runtime_report
        payload = {
            "duration_sec": runtime.duration_sec if runtime is not None else 0.0,
            "output_item_count": len(report.output_item_ids),
            "persisted_terminal_result": report.persisted_terminal_result,
            "execution_event_recorded": runtime is not None,
        }
        if isinstance(runtime, AgentTaskRuntimeReport):
            self._record_trace(
                "task_completed",
                session_id=task.session_id,
                task_id=task_id,
                node_id=task.node_id,
                accelerator_id=runtime.accelerator_id,
                gpu_kind=runtime.gpu_kind,
                model_key=runtime.model_key,
                acquire_id=runtime.acquire_id,
                payload=payload,
            )
        else:
            self._record_trace(
                "task_completed",
                session_id=task.session_id,
                task_id=task_id,
                node_id=task.node_id,
                payload=payload,
            )
        if self.core.sessions[task.session_id].state == SessionState.COMPLETED:
            self._mark_session_terminal(task.session_id)

    def _apply_fail_node(
        self,
        task_id: str,
        error_type: str,
        error_message: str,
    ) -> None:
        task = self.core.tasks[task_id]
        before_session = self.core.sessions[task.session_id].state
        self.core.fail_node(task_id, error_type, error_message)
        self._record_trace(
            "task_failed",
            session_id=task.session_id,
            task_id=task_id,
            node_id=task.node_id,
            payload={"status": "failed", "error_type": error_type},
        )
        if before_session == SessionState.ACTIVE:
            self._mark_session_terminal(task.session_id)

    def _apply_cancel_session(
        self,
        session_id: str,
        error_type: str,
        error_message: str,
    ) -> None:
        before = self.core._session(session_id).state
        self.core.cancel_session(session_id, error_type, error_message)
        if before == SessionState.ACTIVE:
            self._mark_session_terminal(session_id)

    def _run_scheduling_pass(self) -> None:
        now = time.time()
        before_sessions = {
            session_id: session.state
            for session_id, session in self.core.sessions.items()
        }
        before_tasks = {
            task_id: task.state for task_id, task in self.core.tasks.items()
        }
        pending_before = dict(self.core.pending_acquires)
        near_before = self.core.near_ready_tasks()
        actions = self.core.tick_once(now)
        self._record_trace(
            "scheduler_tick",
            ts=now,
            payload={
                "pending_acquire_count": len(pending_before),
                "near_ready_count": len(near_before),
                "action_count": len(actions),
            },
        )
        self._record_tick_transitions(before_sessions, before_tasks)
        self._record_infeasible_decisions(pending_before, now)
        self._record_new_grants()
        self._record_scheduler_decisions(actions, now)
        self._record_prefetch_skips(near_before, actions, now)
        self._manual_actions.extend(actions)

    def _record_infeasible_decisions(
        self,
        pending_before: Mapping[str, PendingAcquire],
        now: float,
    ) -> None:
        for pending in pending_before.values():
            task = self.core.tasks[pending.task_id]
            if (
                task.state != NodeTaskState.FAILED
                or task.error_type != "RequestInfeasible"
            ):
                continue
            node = self.core._nodes[task.node_id]
            if not isinstance(node, AgentNodeConfig):
                raise RuntimeError("infeasible acquire belongs to a function task")
            self._record_trace(
                "request_infeasible",
                ts=now,
                session_id=task.session_id,
                task_id=task.task_id,
                node_id=task.node_id,
                payload={
                    "reason": task.error_message,
                    "input_tokens": pending.input_tokens,
                    "max_new_tokens": node.execution.max_new_tokens,
                },
            )

    def _record_scheduler_decisions(
        self,
        actions: list[LoadReplicaAction | EvictReplicaAction],
        now: float,
    ) -> None:
        for action in actions:
            if isinstance(action, LoadReplicaAction):
                self._record_trace(
                    "scheduler_decision",
                    ts=now,
                    session_id=action.session_id,
                    node_id=action.node_id,
                    replica_id=action.replica_id,
                    accelerator_id=action.accelerator.accelerator_id,
                    gpu_kind=action.accelerator.gpu_kind,
                    model_key=action.deployment.model_key,
                    payload={
                        "action_type": "load_replica",
                        "reason": action.reason,
                    },
                )
                continue
            replica = self.core.replicas[action.replica_id]
            self._record_trace(
                "scheduler_decision",
                ts=now,
                session_id=action.session_id,
                node_id=action.node_id,
                replica_id=action.replica_id,
                accelerator_id=action.accelerator_ids[0],
                gpu_kind=replica.gpu_kind,
                model_key=replica.model_key,
                payload={
                    "action_type": "evict_replica",
                    "reason": action.reason,
                },
            )

    def _record_prefetch_skips(
        self,
        near: tuple[NearReadyTask, ...],
        actions: list[LoadReplicaAction | EvictReplicaAction],
        now: float,
    ) -> None:
        started = {
            (action.session_id, action.node_id, action.prefetch_at)
            for action in actions
            if action.reason == "near_ready_prefetch"
        }
        ready_work_priority = any(action.reason == "ready_load" for action in actions)
        for candidate in near:
            identity = (
                candidate.session_id,
                candidate.node_id,
                candidate.prefetch_at,
            )
            if now < candidate.prefetch_at or identity in started:
                continue
            replica = self.core._replica_for_near(candidate)
            if replica is not None:
                reason = (
                    "already_loading"
                    if replica.state == ModelReplicaState.LOADING
                    else "replica_resident"
                )
            elif ready_work_priority:
                reason = "ready_work_priority"
            else:
                reason = "capacity_unavailable"
            skip_key = (*identity, reason)
            if skip_key in self._prefetch_skip_keys:
                continue
            self._prefetch_skip_keys.add(skip_key)
            self._record_trace(
                "prefetch_skipped",
                ts=now,
                session_id=candidate.session_id,
                node_id=candidate.node_id,
                model_key=candidate.deployment.model_key,
                payload={
                    "reason": reason,
                    "prefetch_at": candidate.prefetch_at,
                },
            )

    def _record_tick_transitions(
        self,
        before_sessions: Mapping[str, SessionState],
        before_tasks: Mapping[str, NodeTaskState],
    ) -> None:
        for task_id, task in self.core.tasks.items():
            if (
                before_tasks.get(task_id) != NodeTaskState.FAILED
                and task.state == NodeTaskState.FAILED
            ):
                self._record_trace(
                    "task_failed",
                    session_id=task.session_id,
                    task_id=task_id,
                    node_id=task.node_id,
                    payload={"status": "failed", "error_type": task.error_type},
                )
        for session_id, session in self.core.sessions.items():
            if (
                before_sessions.get(session_id) == SessionState.ACTIVE
                and session.state == SessionState.FAILED
            ):
                self._mark_session_terminal(session_id)

    def _mark_session_terminal(self, session_id: str) -> None:
        session = self.core.sessions[session_id]
        if session.state not in (SessionState.COMPLETED, SessionState.FAILED):
            raise RuntimeError(f"session is not terminal: {session_id}")
        self._terminal_events.setdefault(session_id, asyncio.Event()).set()

    def _record_new_grants(self) -> None:
        current = set(self.core.grants)
        for acquire_id in sorted(current - self._known_grants):
            grant = self.core.grants[acquire_id]
            task = self.core.tasks[grant.task_id]
            predictions = self.core.predictions
            if predictions is None:
                raise RuntimeError("granted acquire requires a prediction cache")
            prediction = predictions.lookup(grant.prediction_key)
            self._record_trace(
                "acquire_granted",
                session_id=task.session_id,
                task_id=task.task_id,
                node_id=task.node_id,
                acquire_id=acquire_id,
                replica_id=grant.replica_id,
                accelerator_id=grant.accelerator_ids[0],
                gpu_kind=grant.gpu_kind,
                model_key=grant.model_key,
                payload={
                    "admitted_batch_size": grant.admitted_batch_size,
                    "replica_inflight_at_grant": grant.admitted_batch_size,
                },
            )
            self._record_trace(
                "placement_selected",
                session_id=task.session_id,
                task_id=task.task_id,
                node_id=task.node_id,
                acquire_id=acquire_id,
                payload={
                    "input_tokens": grant.input_tokens,
                    "max_new_tokens": grant.max_new_tokens,
                    "admitted_batch_size": grant.admitted_batch_size,
                    "replica_inflight_at_grant": grant.admitted_batch_size,
                    "prediction_cache_version": predictions.version,
                    "prediction_key": prediction.key.model_dump(mode="json"),
                    "predicted_load_sec": prediction.predicted_load_sec,
                    "predicted_run_sec": prediction.predicted_run_sec,
                    "predicted_peak_vram_mb": prediction.predicted_peak_vram_mb,
                    "predicted_power_watts": prediction.predicted_power_watts,
                    "predictor_metadata": prediction.predictor_metadata,
                },
            )
            self._record_trace(
                "scheduler_decision",
                session_id=task.session_id,
                task_id=task.task_id,
                node_id=task.node_id,
                acquire_id=acquire_id,
                replica_id=grant.replica_id,
                accelerator_id=grant.accelerator_ids[0],
                gpu_kind=grant.gpu_kind,
                model_key=grant.model_key,
                payload={"action_type": "grant", "reason": "ready"},
            )
            if grant.replica_id in self._fresh_replicas:
                self._fresh_replicas.remove(grant.replica_id)
            else:
                self._record_trace(
                    "model_reused",
                    session_id=task.session_id,
                    task_id=task.task_id,
                    node_id=task.node_id,
                    acquire_id=acquire_id,
                    replica_id=grant.replica_id,
                    model_key=grant.model_key,
                    accelerator_id=grant.accelerator_ids[0],
                    gpu_kind=grant.gpu_kind,
                )
        self._known_grants = current

    def _dispatch_manual_actions(self) -> None:
        actions = self._manual_actions
        self._manual_actions = []
        for action in actions:
            if isinstance(action, LoadReplicaAction):
                self._dispatch_load(action)
            else:
                self._dispatch_eviction(action)

    def _dispatch_load(self, action: LoadReplicaAction) -> None:
        self._load_actions[action.replica_id] = action
        if self._force_cleanup_started:
            self._enqueue_internal_nowait(
                "load_failed",
                action.replica_id,
                "RuntimeStopping",
                "model load rejected during forced cleanup",
            )
            return
        try:
            handle = self._create_replica(action)
            self._replica_handles[action.replica_id] = handle
            load_ref = _call_remote(handle, "load")
        except Exception as error:
            self._enqueue_internal_nowait(
                "load_failed",
                action.replica_id,
                type(error).__name__,
                str(error),
            )
            return
        self._record_trace(
            "model_load_started",
            replica_id=action.replica_id,
            accelerator_id=action.accelerator.accelerator_id,
            gpu_kind=action.accelerator.gpu_kind,
            model_key=action.deployment.model_key,
            payload={
                "reason": action.reason,
                "max_model_len": action.deployment.serving.max_model_len,
                "max_num_seqs": action.deployment.serving.max_num_seqs,
                "max_num_batched_tokens": (
                    action.deployment.serving.max_num_batched_tokens
                ),
                "gpu_memory_utilization": (
                    action.deployment.serving.gpu_memory_utilization
                ),
            },
        )
        if action.reason == "near_ready_prefetch":
            self._record_trace(
                "prefetch_started",
                session_id=action.session_id,
                node_id=action.node_id,
                replica_id=action.replica_id,
                accelerator_id=action.accelerator.accelerator_id,
                gpu_kind=action.accelerator.gpu_kind,
                model_key=action.deployment.model_key,
                payload={
                    "prefetch_at": action.prefetch_at,
                    "expected_load_sec": action.expected_load_sec,
                },
            )
        self._start_watcher(self._watch_load(action.replica_id, handle, load_ref))

    def _create_replica(self, action: LoadReplicaAction) -> object:
        factory = self._replica_factory
        if factory is not None:
            remote = getattr(factory, "remote", None)
            if callable(remote):
                return remote(action.deployment)
            if callable(factory):
                return cast(ReplicaFactory, factory)(action)
            raise TypeError("replica factory must be callable or expose remote()")
        try:
            node_id = self._ray_node_ids[action.accelerator.hostname]
        except KeyError:
            raise KeyError(
                f"Ray node id is missing for {action.accelerator.hostname}"
            ) from None
        executable = self._config.vllm_python_executable
        if executable is None:
            raise ValueError("vllm_python_executable is required")
        source_root = str(Path(__file__).resolve().parents[1])
        current_pythonpath = os.environ.get("PYTHONPATH")
        pythonpath = (
            source_root
            if not current_pythonpath
            else os.pathsep.join((source_root, current_pythonpath))
        )
        return ModelReplicaActor.options(
            max_concurrency=action.deployment.serving.max_num_seqs + 2,
            runtime_env={
                "py_executable": executable,
                "env_vars": {
                    "PYTHONPATH": pythonpath,
                    "VLLM_ATTENTION_BACKEND": "XFORMERS",
                    "VLLM_NO_USAGE_STATS": "1",
                    "VLLM_USE_V1": "0",
                },
            },
            scheduling_strategy=NodeAffinitySchedulingStrategy(
                node_id=node_id,
                soft=False,
            ),
        ).remote(action.deployment)

    async def _watch_load(
        self,
        replica_id: str,
        handle: object,
        load_ref: object,
    ) -> None:
        try:
            value = await _await_value(load_ref)
            if not isinstance(value, ReplicaLoadResult):
                raise TypeError("replica load returned an invalid result")
        except Exception as error:
            await self._enqueue_internal(
                "load_failed",
                replica_id,
                type(error).__name__,
                str(error),
            )
            return
        await self._enqueue_internal("load_succeeded", replica_id, value, handle)

    def _dispatch_eviction(self, action: EvictReplicaAction) -> None:
        handle = self._replica_handles.get(action.replica_id)
        if handle is None:
            self._enqueue_internal_nowait(
                "eviction_failed",
                action.replica_id,
                "MissingReplicaHandle",
                "replica handle is unavailable",
            )
            return
        replica = self.core.replicas[action.replica_id]
        self._record_trace(
            "model_eviction_started",
            replica_id=action.replica_id,
            accelerator_id=action.accelerator_ids[0],
            gpu_kind=replica.gpu_kind,
            model_key=replica.model_key,
            payload={"reason": action.reason},
        )
        self._start_watcher(self._watch_eviction(action.replica_id, handle))

    async def _watch_eviction(self, replica_id: str, handle: object) -> None:
        shutdown_ref = _call_remote(handle, "shutdown")
        with suppress(Exception):
            await asyncio.wait_for(
                _await_value(shutdown_ref),
                timeout=self._config.eviction_timeout_sec,
            )
        try:
            await asyncio.wait_for(
                asyncio.to_thread(ray.kill, cast(Any, handle), no_restart=True),
                timeout=self._config.eviction_timeout_sec,
            )
        except Exception as error:
            await self._enqueue_internal(
                "eviction_failed",
                replica_id,
                type(error).__name__,
                str(error),
            )
            return
        await self._enqueue_internal("eviction_succeeded", replica_id)

    def _dispatch_cancel_actions(self) -> None:
        for action in self.core.take_actions():
            refs: list[object] = []
            try:
                refs.extend(
                    _call_remote(worker, "cancel_session", action.session_id)
                    for worker in self._workers.values()
                )
            except Exception as error:
                self._enqueue_internal_nowait(
                    "worker_cancel_failed",
                    f"worker cancellation failed: {error}",
                )
                continue
            if refs:
                self._start_watcher(
                    self._watch_worker_cancellation(action.session_id, refs)
                )

    async def _watch_worker_cancellation(
        self,
        session_id: str,
        refs: list[object],
    ) -> None:
        try:
            for ref in refs:
                await _await_value(ref)
        except Exception as error:
            await self._enqueue_internal(
                "worker_cancel_failed",
                f"worker cancellation failed for {session_id}: {error}",
            )

    def _dispatch_evict_all(self) -> None:
        for replica in tuple(self.core.replicas.values()):
            if replica.state not in (
                ModelReplicaState.IDLE,
                ModelReplicaState.SUSPECT,
            ):
                continue
            if replica.active_acquire_ids:
                continue
            self._manual_actions.append(self.core.request_eviction(replica.replica_id))
        if not self.core.replicas:
            self._evict_all_requested = False

    def _record_trace(
        self,
        event_type: str,
        *,
        ts: float | None = None,
        **fields: object,
    ) -> None:
        payload = fields.get("payload")
        if payload is not None:
            _validate_trace_payload(payload)
        event = TraceEvent.model_validate(
            {
                "run_id": self.run_id,
                "event_seq": self._next_event_seq,
                "event_type": event_type,
                "ts": time.time() if ts is None else ts,
                **fields,
            }
        )
        self._next_event_seq += 1
        self._trace_buffer.append(event)

    def _submit_trace_batch(self) -> None:
        if self._trace_inflight or not self._trace_buffer:
            return
        events = self._trace_buffer
        self._trace_buffer = []
        expected = events[-1].event_seq
        self._trace_inflight = True
        try:
            ref = _call_remote(self._trace_writer, "append_batch", events)
        except Exception as error:
            self._enqueue_internal_nowait("trace_failed", str(error))
            return
        self._start_watcher(self._watch_trace(ref, expected))

    async def _watch_trace(self, ref: object, expected: int) -> None:
        try:
            acknowledged = await _await_value(ref)
            if not isinstance(acknowledged, int):
                raise TypeError("trace writer returned an invalid acknowledgement")
        except Exception as error:
            await self._enqueue_internal("trace_failed", str(error))
            return
        await self._enqueue_internal("trace_succeeded", acknowledged, expected)

    def _start_watcher(self, awaitable: Coroutine[Any, Any, None]) -> None:
        task = asyncio.create_task(awaitable)
        self._watcher_tasks.add(task)
        task.add_done_callback(self._watcher_tasks.discard)

    async def _enqueue_internal(self, name: str, *args: object) -> None:
        await self._commands.put(
            _SchedulerCommand(name=name, args=args, kwargs={}, future=None)
        )

    def _enqueue_internal_nowait(self, name: str, *args: object) -> None:
        self._commands.put_nowait(
            _SchedulerCommand(name=name, args=args, kwargs={}, future=None)
        )

    def _set_run_failure(
        self,
        error: Exception,
        *,
        flush_trace: bool = False,
    ) -> None:
        if self._run_failure is None:
            self._run_failure = RuntimeError(str(error))
            self._flush_failure_trace = flush_trace
        elif not flush_trace:
            self._flush_failure_trace = False

    def _fail_queued_commands(self, error: Exception) -> None:
        while True:
            try:
                command = self._commands.get_nowait()
            except asyncio.QueueEmpty:
                return
            if command.future is not None and not command.future.done():
                command.future.set_exception(error)

    def _trace_is_idle(self) -> bool:
        return not self._trace_buffer and not self._trace_inflight

    def _can_stop(self) -> bool:
        return (
            self._stop_requested
            and self._run_finished_recorded
            and self._commands.empty()
            and self._trace_is_idle()
            and not any(not task.done() for task in self._watcher_tasks)
        )

    async def _cleanup_runtime(self) -> None:
        watchers = tuple(self._watcher_tasks)
        for watcher in watchers:
            watcher.cancel()
        if watchers:
            await asyncio.gather(*watchers, return_exceptions=True)
        await self._kill_replica_handles()

    async def _kill_replica_handles(self) -> None:
        handles = tuple(self._replica_handles.items())
        if not handles:
            return
        kills = [self._shutdown_and_kill(handle) for _, handle in handles]
        results = await asyncio.gather(*kills, return_exceptions=True)
        for (replica_id, handle), result in zip(handles, results, strict=True):
            if (
                not isinstance(result, BaseException)
                and self._replica_handles.get(replica_id) is handle
            ):
                del self._replica_handles[replica_id]

    async def _shutdown_and_kill(self, handle: object) -> None:
        shutdown_ref = _call_remote(handle, "shutdown")
        with suppress(Exception):
            await asyncio.wait_for(
                _await_value(shutdown_ref),
                timeout=self._config.eviction_timeout_sec,
            )
        await asyncio.wait_for(
            asyncio.to_thread(ray.kill, cast(Any, handle), no_restart=True),
            timeout=self._config.eviction_timeout_sec,
        )


def _call_remote(
    target: object,
    method_name: str,
    *args: object,
    **kwargs: object,
) -> object:
    method = getattr(target, method_name)
    remote = getattr(method, "remote", None)
    if callable(remote):
        return remote(*args, **kwargs)
    if callable(method):
        return method(*args, **kwargs)
    raise TypeError(f"method is not callable: {method_name}")


async def _await_value(value: object) -> object:
    if inspect.isawaitable(value):
        return await value
    return value


def _validate_trace_payload(value: object) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            if key in {"agent_state", "message", "messages", "output_text", "text"}:
                raise ValueError(f"trace payload contains forbidden field: {key}")
            _validate_trace_payload(item)
        return
    if isinstance(value, (list, tuple)):
        for item in value:
            _validate_trace_payload(item)


_remote_with_options = cast(Any, ray.remote)
SchedulerActor = _remote_with_options(max_concurrency=64)(_SchedulerActor)
