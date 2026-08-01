from __future__ import annotations

import asyncio
import os
import time
from collections import deque
from collections.abc import Callable, Coroutine, Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass
from math import ceil, inf, isfinite
from pathlib import Path
from typing import Any, Literal, cast
from uuid import uuid4

import ray
from pydantic import BaseModel, ConfigDict, Field, NonNegativeInt, PositiveInt
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

from common.validate import NonEmptyStr
from workflow.actor_support import await_value, dispatch
from workflow.artifacts import (
    AcceleratorConfig,
    GpuKind,
    ResourceContract,
    ResourceContractCache,
    SchedulerConfig,
)
from workflow.policy import (
    EvictionCandidate,
    PlacementDecision,
    critical_path_depth_to_node,
    critical_path_remaining_latency,
    select_eviction_victim,
    select_placement,
)
from workflow.replica import (
    ModelDeploymentConfig,
    ModelReplicaActor,
    ReplicaLoadResult,
)
from workflow.schema import (
    AgentNodeConfig,
    FunctionNodeConfig,
    FusedAgentNodeConfig,
    NodeConfig,
    Workflow,
)
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


def _finite_or_none(value: float | None) -> float | None:
    # Trace payloads go through a bare json.dumps, which emits a non-standard
    # Infinity token that strict readers reject.
    if value is None or not isfinite(value):
        return None
    return value


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


class FusedStageReport(StrictFrozenModel):
    node_id: NonEmptyStr
    input_tokens: PositiveInt
    max_new_tokens: PositiveInt
    output_tokens: NonNegativeInt
    hit_token_limit: bool
    started_at: float
    finished_at: float
    duration_sec: float = Field(ge=0)
    status: Literal["success", "failed", "oom", "cancelled"]


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
    # Populated by fused nodes only: one entry per stage, in execution order. The
    # aggregate input_tokens above is the admitted envelope, not a stage measurement.
    stage_reports: tuple[FusedStageReport, ...] = ()


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
    reason: Literal["ready_load", "near_ready_prefetch", "scale_out"]
    expected_load_sec: float = Field(gt=0)
    owner_workflow: str | None = None
    session_id: str | None = None
    node_id: str | None = None
    prefetch_at: float | None = None
    scale_out_gain_sec: float | None = None
    group_size: NonNegativeInt | None = None


class EvictReplicaAction(StrictFrozenModel):
    replica_id: NonEmptyStr
    accelerator_ids: tuple[NonEmptyStr, ...]
    reason: Literal[
        "suspect_cleanup",
        "ready_load",
        "near_ready_prefetch",
        "requested",
        "scale_out",
    ]
    reuse_distance_sec: float | None = None
    reload_cost_sec: float | None = None
    session_id: str | None = None
    node_id: str | None = None
    prefetch_at: float | None = None
    scale_out_gain_sec: float | None = None


@dataclass(frozen=True, slots=True)
class _ScaleOutBid:
    pair: tuple[str, GpuKind]
    gain_sec: float
    load_sec: float
    requirements: Mapping[tuple[str, GpuKind], float]


class DrainEvent(StrictFrozenModel):
    replica_id: NonEmptyStr
    model_key: NonEmptyStr
    gpu_kind: GpuKind
    accelerator_id: NonEmptyStr
    active_acquire_count: NonNegativeInt
    requested_by_model_key: NonEmptyStr


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
    workflow_name: NonEmptyStr
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
    # Workflow whose demand triggered this load. Replicas stay shared regardless —
    # this only scopes the lifecycle signals when cross_workflow_lifecycle is off.
    owner_workflow: str | None = None
    expected_load_sec: float = Field(gt=0)
    backend_handle: object | None = None
    physical_gpu_id: int | str | None = None
    active_acquire_ids: set[str] = Field(default_factory=set)
    created_at: float
    idle_since: float | None = None
    # Orthogonal to state: a draining replica keeps serving its in-flight leases but
    # accepts no new ones, so it reaches IDLE on its own and becomes reclaimable.
    drain_requested_at: float | None = None
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


class ReplicaGroups:
    """Live replica ids per (model_key, gpu_kind), in creation order.

    A group holds more than one member only when elastic scaling is enabled; the
    capacity check lives at the load sites and in _validate_resource_ledger.
    """

    def __init__(self) -> None:
        self._groups: dict[tuple[str, GpuKind], tuple[str, ...]] = {}

    def members(self, pair: tuple[str, GpuKind]) -> tuple[str, ...]:
        return self._groups.get(pair, ())

    def size(self, pair: tuple[str, GpuKind]) -> int:
        return len(self._groups.get(pair, ()))

    def first(self, pair: tuple[str, GpuKind]) -> str | None:
        members = self._groups.get(pair, ())
        return members[0] if members else None

    def add(self, pair: tuple[str, GpuKind], replica_id: str) -> None:
        members = self._groups.get(pair, ())
        if replica_id in members:
            raise RuntimeError(f"replica is already in its group: {replica_id}")
        self._groups[pair] = (*members, replica_id)

    def discard(self, pair: tuple[str, GpuKind], replica_id: str) -> bool:
        members = self._groups.get(pair, ())
        if replica_id not in members:
            return False
        remaining = tuple(member for member in members if member != replica_id)
        if remaining:
            self._groups[pair] = remaining
        else:
            del self._groups[pair]
        return True

    def items(self) -> tuple[tuple[tuple[str, GpuKind], tuple[str, ...]], ...]:
        return tuple(self._groups.items())


class SchedulerCore:
    def __init__(
        self,
        workflow: Workflow,
        *,
        scheduler_config: SchedulerConfig | None = None,
        predictions: ResourceContractCache | None = None,
        priority_weight: float = 1.0,
    ) -> None:
        self.scheduler_config = scheduler_config or SchedulerConfig()
        self.predictions = predictions
        self.sessions: dict[str, SessionRecord] = {}
        self.tasks: dict[str, NodeTaskRecord] = {}
        self.pending_acquires: dict[str, PendingAcquire] = {}
        self.pending_order: deque[str] = deque()
        self.grants: dict[str, GrantInfo] = {}
        # Per-lease predicted finish time, keyed like self.grants. Kept out of
        # GrantInfo because that model is frozen and crosses the Ray boundary to
        # workers; elastic routing reads it to price an occupied replica.
        self._grant_finish_at: dict[str, float] = {}
        self._inactive_acquires: dict[str, str] = {}
        self.accelerators = {
            accelerator.accelerator_id: AcceleratorRecord(config=accelerator)
            for accelerator in self.scheduler_config.accelerators
        }
        self.replicas: dict[str, ModelReplicaRecord] = {}
        self.oom_penalties: dict[WorkflowModelFeatureKey, float] = {}
        # Keyed by (workflow_name, node_id, model_key, gpu_kind): task duration
        # depends on a node's own prompt/output-length distribution, which two
        # workflows can share a node name and model_key without sharing.
        self.history: dict[tuple[str, str, str, GpuKind], RuntimeHistoryRecord] = {}
        self.duration_history: dict[
            tuple[str, str, str, GpuKind, int], DurationHistoryRecord
        ] = {}
        # Keyed by (model_key, gpu_kind) only: load time does not depend on
        # which workflow requested the model.
        self.load_history: dict[tuple[str, GpuKind], DurationHistoryRecord] = {}
        self.running_estimates: dict[tuple[str, str], RunningTaskEstimate] = {}
        self._workflows: dict[str, Workflow] = {}
        self._nodes: dict[str, dict[str, NodeConfig]] = {}
        self._node_order: dict[str, dict[str, int]] = {}
        self._workflow_weights: dict[str, float] = {}
        # Per-workflow static node -> remaining critical-path latency, used only
        # by the Kairos SRPT ordering. Recomputed on register; other policies
        # never read it.
        self._remaining_latency: dict[str, dict[str, float]] = {}
        # Static per-node run costs and earliest-start depths, both derived from the
        # prediction cache at register time. Reuse-distance eviction reads the depths.
        self._node_costs: dict[str, Mapping[str, float]] = {}
        self._node_depth: dict[str, dict[str, float]] = {}
        # Keyed by (model_key, gpu_kind) only: a replica is shareable across
        # workflows whenever their deployment configs are byte-identical.
        self._replica_groups = ReplicaGroups()
        self._actions: list[CancelSessionAction] = []
        self._drain_events: list[DrainEvent] = []
        self._next_request_seq = 0
        # select_eviction_victim tie-breaks on replica_id, so a random id makes the
        # victim run-to-run nondeterministic whenever two candidates score equally
        # (common: _future_reuse_distance returns inf for any unwanted model).
        self._next_replica_seq = 0
        self.register_workflow(workflow, priority_weight=priority_weight)

    def _new_replica_id(self) -> str:
        self._next_replica_seq += 1
        return f"r{self._next_replica_seq:04d}"

    @property
    def _replica_group_cap(self) -> int:
        if not self.scheduler_config.elastic_replicas:
            return 1
        return self.scheduler_config.max_replicas_per_model

    def register_workflow(
        self,
        workflow: Workflow,
        *,
        priority_weight: float = 1.0,
    ) -> None:
        if workflow.workflow_name in self._workflows:
            raise ValueError(f"workflow already registered: {workflow.workflow_name}")
        if priority_weight <= 0:
            raise ValueError("priority_weight must be positive")
        self._workflows[workflow.workflow_name] = workflow
        self._nodes[workflow.workflow_name] = workflow.node_map()
        self._node_order[workflow.workflow_name] = {
            node_id: index
            for index, node_id in enumerate(workflow.graph.topological_order)
        }
        self._workflow_weights[workflow.workflow_name] = priority_weight
        run_costs = self._node_run_costs(workflow)
        self._node_costs[workflow.workflow_name] = run_costs
        self._remaining_latency[workflow.workflow_name] = (
            critical_path_remaining_latency(workflow.graph, run_costs)
        )
        self._node_depth[workflow.workflow_name] = critical_path_depth_to_node(
            workflow.graph, run_costs
        )

    def deregister_workflow(self, workflow_name: str) -> None:
        if workflow_name not in self._workflows:
            raise KeyError(f"unknown workflow: {workflow_name}")
        if any(
            session.workflow_name == workflow_name
            and session.state == SessionState.ACTIVE
            for session in self.sessions.values()
        ):
            raise ValueError(f"workflow has active sessions: {workflow_name}")
        del self._workflows[workflow_name]
        del self._nodes[workflow_name]
        del self._node_order[workflow_name]
        del self._workflow_weights[workflow_name]
        del self._remaining_latency[workflow_name]

    def _node_run_costs(self, workflow: Workflow) -> dict[str, float]:
        # Static per-node point estimate of decode run time, feeding the Kairos
        # remaining-latency ranking only. Agent nodes read a representative
        # decode contract; function nodes and any node without cache coverage
        # contribute zero (a soft-decision default).
        costs: dict[str, float] = {}
        for node_id, node in workflow.node_map().items():
            if isinstance(node, AgentNodeConfig):
                costs[node_id] = self._representative_run_sec(node)
            else:
                costs[node_id] = 0.0
        return costs

    @staticmethod
    def _stage_outputs(node: AgentNodeConfig) -> tuple[int, ...]:
        """Per-stage output limits: one entry for a plain node, k for a fused chain."""
        if isinstance(node, FusedAgentNodeConfig):
            return tuple(stage.max_new_tokens for stage in node.stages)
        return (node.execution.max_new_tokens,)

    def _chain_run_sec(
        self,
        node: AgentNodeConfig,
        gpu_kind: GpuKind,
        sequence_length: int,
        batch_size: int,
    ) -> float | None:
        """Cost of running every stage of a node once, or None without coverage.

        A fused node occupies its replica for the whole chain, so its ETA is the sum
        of the stages rather than the single admission bucket, which only ever
        describes the widest one.
        """
        predictions = self.predictions
        if predictions is None:
            return None
        total = 0.0
        for max_new_tokens in self._stage_outputs(node):
            bucket = next(
                (
                    candidate
                    for candidate in predictions.decode_output_lengths(
                        node.model.name, gpu_kind, sequence_length, batch_size
                    )
                    if candidate >= max_new_tokens
                ),
                None,
            )
            if bucket is None:
                return None
            total += predictions.lookup_decode(
                model_name=node.model.name,
                gpu_kind=gpu_kind,
                batch_size=batch_size,
                sequence_length=sequence_length,
                decode_output_length=bucket,
            ).predicted_run_sec
        return total

    def _representative_run_sec(self, node: AgentNodeConfig) -> float:
        # Cheapest feasible GPU (smallest memory with cache coverage), batch 1,
        # smallest covering input bucket, output bucket >= max_new_tokens —
        # mirrors select_placement's bin-packing so the estimate tracks the GPU
        # a request would actually land on.
        predictions = self.predictions
        if predictions is None:
            return 0.0
        accelerators = sorted(
            (record.config for record in self.accelerators.values()),
            key=lambda accelerator: (
                accelerator.total_mem_mb,
                accelerator.accelerator_id,
            ),
        )
        for accelerator in accelerators:
            sequence_length = next(
                iter(
                    predictions.decode_sequence_lengths(
                        node.model.name, accelerator.gpu_kind, 1
                    )
                ),
                None,
            )
            if sequence_length is None:
                continue
            run_sec = self._chain_run_sec(
                node, accelerator.gpu_kind, sequence_length, 1
            )
            if run_sec is None:
                continue
            return run_sec
        return 0.0

    def _workflow(self, workflow_name: str) -> Workflow:
        try:
            return self._workflows[workflow_name]
        except KeyError:
            raise KeyError(f"unknown workflow: {workflow_name}") from None

    def register_session(self, session_id: str, workflow_name: str) -> None:
        if session_id in self.sessions:
            raise ValueError(f"session already registered: {session_id}")

        workflow = self._workflow(workflow_name)
        session = SessionRecord(session_id=session_id, workflow_name=workflow_name)
        for node in workflow.nodes:
            task_id = str(uuid4())
            task = NodeTaskRecord(
                task_id=task_id,
                session_id=session_id,
                node_id=node.name,
                # A fused chain is still an agent task: it acquires, holds one grant
                # and completes through _complete_agent like any other.
                node_kind=(
                    "function" if isinstance(node, FunctionNodeConfig) else "agent"
                ),
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
        nodes = self._nodes[session.workflow_name]
        if node_id not in nodes:
            raise KeyError(f"unknown node: {node_id}")

        task = self.tasks[session.task_ids[node_id]]
        if task.state != NodeTaskState.PENDING:
            raise ValueError(f"node already begun: {session_id}/{node_id}")
        if not self._dependencies_completed(session, node_id):
            raise ValueError(f"node dependencies are not completed: {node_id}")

        task.input_item_ids = input_item_ids
        node = nodes[node_id]
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
        topological_order = self._workflow(
            session.workflow_name
        ).graph.topological_order
        failed_task = next(
            (
                self.tasks[session.task_ids[node_id]]
                for node_id in topological_order
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

    def record_runtime_report(
        self,
        report: AgentTaskRuntimeReport,
        workflow_name: str,
    ) -> None:
        key = (workflow_name, report.node_id, report.model_key, report.gpu_kind)
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

        decisions = self._ready_decisions(now)
        granted_ready = False
        for acquire_id in tuple(decisions):
            pending = self.pending_acquires.get(acquire_id)
            if pending is None:
                continue
            admitted = self._admissible_replica(pending, decisions[acquire_id])
            if admitted is not None:
                replica, prediction = admitted
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
        loadable: list[tuple[PendingAcquire, PlacementDecision]] = []
        for acquire_id in tuple(decisions):
            pending = self.pending_acquires.get(acquire_id)
            if pending is None:
                continue
            decision = decisions[acquire_id]
            # A resident replica normally settles the request, but when it cannot
            # absorb the queue fast enough an extra replica is the cheaper answer.
            # Anything granted above is already out of pending_acquires, so only
            # unserved requests reach here and admission still beats loading.
            if self._replica_for_decision(pending, decision) is not None and (
                not self._scale_out_warranted(pending, decision, now)
            ):
                continue
            if self._free_accelerator(decision) is None:
                blocked_ready.append((pending, decision))
                continue
            loadable.append((pending, decision))
        near = self._collect_near_ready_tasks()
        requirements = self._demand_requirements(decisions, near)
        scaled_out = False
        for pending, decision in self._order_by_load_yield(loadable, near, now):
            accelerator = self._free_accelerator(decision)
            if accelerator is None:
                continue
            # Reordering lets two queued requests for one model reach this loop; the
            # group registry caps how many replicas a (model, gpu kind) may hold.
            gpu_kind = decision.gpu_kind
            model_key = self._pending_model_key(pending)
            if gpu_kind is None:
                raise RuntimeError("feasible placement has no GPU kind")
            if model_key is None:
                continue
            pair = (model_key, gpu_kind)
            if self._replica_groups.size(pair) >= self._replica_group_cap:
                continue
            scale_out = self._replica_groups.size(pair) > 0
            gain: float | None = None
            if scale_out:
                # Grants earlier in this tick shorten the queue, so the case for an
                # extra replica has to be re-made against the state that remains.
                gain = self._scale_out_gain(pending, decision, now)
                if gain is None or gain <= self.scheduler_config.scale_out_margin_sec:
                    continue
                # One per tick keeps the reservation snapshot valid for the whole loop.
                if scaled_out or not self._scale_out_preserves_feasibility(
                    pair, accelerator.config.accelerator_id, requirements
                ):
                    continue
                scaled_out = True
            actions.append(
                self._start_ready_load(
                    pending,
                    decision,
                    accelerator,
                    now,
                    reason="scale_out" if scale_out else "ready_load",
                    scale_out_gain_sec=gain,
                )
            )
        if actions:
            self._validate_resource_ledger()
            return actions
        if granted_ready:
            self._validate_resource_ledger()
            return []

        protected_pairs = self._ready_pairs(decisions)
        eviction_actions: list[LoadReplicaAction | EvictReplicaAction] = []
        starved: list[tuple[PendingAcquire, PlacementDecision]] = []
        for pending, decision in self._order_by_load_yield(blocked_ready, near, now):
            if pending.acquire_id not in self.pending_acquires:
                continue
            bid = self._scale_out_bid(pending, decision, requirements, now)
            if bid is not None and scaled_out:
                continue
            eviction = self._start_eviction_for_decision(
                decision,
                reason="scale_out" if bid is not None else "ready_load",
                now=now,
                protected_pairs=protected_pairs,
                near=near,
                scale_out=bid,
            )
            if eviction is None:
                if bid is None:
                    starved.append((pending, decision))
                continue
            scaled_out = scaled_out or bid is not None
            eviction_actions.append(eviction)
        if eviction_actions:
            self._validate_resource_ledger()
            return eviction_actions

        # A drain has nothing to dispatch: it only stops a spare replica taking new
        # leases so it reaches IDLE and becomes reclaimable on a later tick.
        self._start_drain_for_starved(starved, now)

        # Only the speculative loads are switched off: `near` still feeds reuse-distance
        # eviction, the scale-out reservation and the load-yield ordering above, so this
        # ablation removes prefetch alone rather than silently degrading eviction too.
        if not self.scheduler_config.enable_prefetch:
            self._validate_resource_ledger()
            return []

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
        if replica_id not in self._replica_groups.members(pair):
            raise RuntimeError("replica pair changed during model load")
        for accelerator_id in replica.accelerator_ids:
            if self.accelerators[accelerator_id].replica_id != replica_id:
                raise RuntimeError("accelerator ownership changed during model load")

        self._replica_groups.discard(pair, replica_id)
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
        if replica_id not in self._replica_groups.members(pair):
            raise RuntimeError("replica pair changed during eviction")
        accelerator_ids = replica.accelerator_ids
        for accelerator_id in accelerator_ids:
            if self.accelerators[accelerator_id].replica_id != replica_id:
                raise RuntimeError("accelerator ownership changed during eviction")

        self._replica_groups.discard(pair, replica_id)
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
        graph = self._workflow(self.sessions[task.session_id].workflow_name).graph
        if output_report.output_items:
            if [item.item_id for item in output_report.output_items] != (
                output_report.output_item_ids
            ):
                raise ValueError("output item reports do not match output ids")
            successors = set(graph.adjacency[task.node_id])
            if any(
                item.session_id != task.session_id
                or item.source_node != task.node_id
                or item.target_node not in successors
                for item in output_report.output_items
            ):
                raise ValueError("output item report does not match task routing")
        is_terminal = task.node_id == graph.terminal_node
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

    def take_drain_events(self) -> list[DrainEvent]:
        events = self._drain_events
        self._drain_events = []
        return events

    def drain_complete(self, workflow_name: str | None = None) -> bool:
        sessions = (
            self.sessions.values()
            if workflow_name is None
            else [
                session
                for session in self.sessions.values()
                if session.workflow_name == workflow_name
            ]
        )
        if any(
            session.state not in {SessionState.COMPLETED, SessionState.FAILED}
            for session in sessions
        ):
            return False
        session_ids = {session.session_id for session in sessions}
        if any(
            self.tasks[pending.task_id].session_id in session_ids
            for pending in self.pending_acquires.values()
        ):
            return False
        active_states = {
            NodeTaskState.ACQUIRING,
            NodeTaskState.RUNNING,
            NodeTaskState.EMITTING,
        }
        return not any(
            task.state in active_states and task.session_id in session_ids
            for task in self.tasks.values()
        )

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
        dependencies = self._workflow(session.workflow_name).graph.dependencies
        return all(
            self.tasks[session.task_ids[dependency]].state
            in {NodeTaskState.EMITTING, NodeTaskState.COMPLETED}
            for dependency in dependencies[node_id]
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

    def _pending_node_order(self, pending: PendingAcquire) -> int:
        task = self.tasks[pending.task_id]
        workflow_name = self.sessions[task.session_id].workflow_name
        return self._node_order[workflow_name][task.node_id]

    def _pending_workflow_name(self, pending: PendingAcquire) -> str:
        task = self.tasks[pending.task_id]
        return self.sessions[task.session_id].workflow_name

    def _demand_key(self, workflow_name: str, model_key: str) -> tuple[str | None, str]:
        """Bucket a model's demand: pooled, or one bucket per workflow."""
        if self.scheduler_config.cross_workflow_lifecycle:
            return (None, model_key)
        return (workflow_name, model_key)

    def _resident_model_keys(self) -> frozenset[str]:
        return frozenset(
            replica.model_key
            for replica in self.replicas.values()
            if replica.state in (ModelReplicaState.IDLE, ModelReplicaState.BUSY)
        )

    def _needs_load(
        self,
        pending: PendingAcquire,
        resident: frozenset[str],
        now: float,
    ) -> bool:
        """Whether serving this request costs a load, once aging stops protecting it."""
        starvation_sec = 0.5 * self.scheduler_config.acquire_timeout_sec
        return (
            now - pending.created_at <= starvation_sec
            and self._pending_model_key(pending) not in resident
        )

    def _intra_workflow_key(
        self,
        pending: PendingAcquire,
        *,
        resident: frozenset[str],
        now: float,
    ) -> tuple[int, int]:
        if self.scheduler_config.policy == "fifo":
            return (0, pending.request_seq)
        if self.scheduler_config.policy == "cache":
            # Locality-first, but only over this workflow's own queue. Reached when
            # cross_workflow_lifecycle is off: each workflow still drains a residency
            # before giving it up, it just no longer yields to another workflow's.
            return (
                int(self._needs_load(pending, resident, now)),
                pending.request_seq,
            )
        return (self._pending_node_order(pending), pending.request_seq)

    def _interleave_by_workflow(
        self,
        pending_values: Sequence[PendingAcquire],
        now: float,
    ) -> list[PendingAcquire]:
        # Weighted fair queueing across workflows: each workflow's own pending
        # acquires keep today's single-workflow ordering rule internally (via
        # _intra_workflow_key), then a workflow's k-th ranked item is assigned
        # a virtual rank of k / weight before merging across workflows. Equal
        # weights (the default) reduce this to plain global request order, so
        # a single registered workflow's ordering is unchanged.
        resident = self._resident_model_keys()
        grouped: dict[str, list[PendingAcquire]] = {}
        for pending in pending_values:
            grouped.setdefault(self._pending_workflow_name(pending), []).append(pending)
        for group in grouped.values():
            group.sort(
                key=lambda pending: self._intra_workflow_key(
                    pending, resident=resident, now=now
                )
            )

        ranked: list[tuple[float, int, str, PendingAcquire]] = []
        for workflow_name, group in grouped.items():
            weight = self._workflow_weights.get(workflow_name, 1.0)
            for rank, pending in enumerate(group, start=1):
                ranked.append(
                    (rank / weight, pending.request_seq, workflow_name, pending)
                )
        ranked.sort(key=lambda item: (item[0], item[1], item[2]))
        return [item[3] for item in ranked]

    def _order_pending(
        self, pending_values: Sequence[PendingAcquire], now: float
    ) -> list[PendingAcquire]:
        if self.scheduler_config.policy == "kairos":
            # Kairos ignores workflow identity and priority weight: rank every
            # ready request globally by shortest remaining critical-path latency
            # (SRPT), tie-broken by arrival order.
            return sorted(
                pending_values,
                key=lambda pending: (
                    self._pending_remaining_latency(pending),
                    pending.request_seq,
                ),
            )
        if self.scheduler_config.policy == "cache" and (
            self.scheduler_config.cross_workflow_lifecycle
        ):
            # Weighted-fair interleaving spreads admission across workflows, which
            # fragments model locality: consecutive grants keep landing on different
            # models and each switch costs a full load. Serve work whose model is
            # already resident first, oldest-first within each class, so a residency
            # is drained before it is given up. Aging bounds the delay a request can
            # accumulate from being on the unlucky side of that split.
            resident = self._resident_model_keys()
            return sorted(
                pending_values,
                key=lambda pending: (
                    self._needs_load(pending, resident, now),
                    pending.request_seq,
                ),
            )
        return self._interleave_by_workflow(pending_values, now)

    def _pending_remaining_latency(self, pending: PendingAcquire) -> float:
        task = self.tasks[pending.task_id]
        workflow_name = self.sessions[task.session_id].workflow_name
        return self._remaining_latency.get(workflow_name, {}).get(task.node_id, 0.0)

    def _ready_decisions(self, now: float) -> dict[str, PlacementDecision]:
        if not self.pending_acquires:
            return {}
        predictions = self.predictions
        if predictions is None:
            raise RuntimeError("agent scheduling requires a prediction cache")

        pending_values = tuple(self.pending_acquires.values())
        ordered = self._order_pending(pending_values, now)
        decisions: dict[str, PlacementDecision] = {}
        for snapshot in ordered:
            pending = self.pending_acquires.get(snapshot.acquire_id)
            if pending is None:
                continue
            task = self.tasks[pending.task_id]
            session = self.sessions[task.session_id]
            if session.state != SessionState.ACTIVE:
                continue
            node = self._nodes[session.workflow_name][task.node_id]
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
        predictions: ResourceContractCache,
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
        if self.scheduler_config.policy in ("fifo", "kairos"):
            return []
        predictions = self.predictions
        if predictions is None:
            return []
        near: list[NearReadyTask] = []
        for session in self.sessions.values():
            if session.state != SessionState.ACTIVE:
                continue
            workflow = self._workflow(session.workflow_name)
            nodes = self._nodes[session.workflow_name]
            for node_id in workflow.graph.topological_order:
                node = nodes[node_id]
                task = self.tasks[session.task_ids[node_id]]
                if not isinstance(node, AgentNodeConfig):
                    continue
                if task.state != NodeTaskState.PENDING:
                    continue
                dependencies = workflow.graph.dependencies[node_id]
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
                input_tokens = self._estimate_near_input_tokens(
                    session.workflow_name, node, dependencies
                )
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
                self._node_order[self.sessions[candidate.session_id].workflow_name][
                    candidate.node_id
                ],
                candidate.session_id,
            ),
        )

    def _estimate_near_input_tokens(
        self,
        workflow_name: str,
        node: AgentNodeConfig,
        dependencies: tuple[str, ...],
    ) -> int:
        estimated_outputs = 0
        for dependency in dependencies:
            samples = (
                [
                    history.output_tokens_p90
                    for (
                        history_workflow_name,
                        history_node_id,
                        _,
                        _,
                    ), history in self.history.items()
                    if history_workflow_name == workflow_name
                    and history_node_id == dependency
                    and history.samples
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
        workflow_name = self.sessions[task.session_id].workflow_name
        node = self._nodes[workflow_name][task.node_id]
        if not isinstance(node, AgentNodeConfig):
            raise RuntimeError("pending acquire belongs to a function task")
        model_key = ModelDeploymentConfig.from_node(node).model_key
        replica_id = self._replica_groups.first((model_key, gpu_kind))
        return self.replicas.get(replica_id) if replica_id is not None else None

    def _group_replicas(
        self,
        pending: PendingAcquire,
        decision: PlacementDecision,
    ) -> tuple[ModelReplicaRecord, ...]:
        gpu_kind = decision.gpu_kind
        model_key = self._pending_model_key(pending)
        if gpu_kind is None or model_key is None:
            return ()
        return tuple(
            self.replicas[replica_id]
            for replica_id in self._replica_groups.members((model_key, gpu_kind))
            if replica_id in self.replicas
        )

    def _admissible_replica(
        self,
        pending: PendingAcquire,
        decision: PlacementDecision,
    ) -> tuple[ModelReplicaRecord, ResourceContract] | None:
        # Non-elastic groups hold one member, so this is the historical
        # "look up the replica, then price the batch" path verbatim.
        if not self.scheduler_config.elastic_replicas:
            replica = self._replica_for_decision(pending, decision)
            if replica is None:
                return None
            prediction = self._admission_prediction(decision, replica)
            return None if prediction is None else (replica, prediction)
        scored: list[tuple[float, str, ModelReplicaRecord, ResourceContract]] = []
        for replica in self._group_replicas(pending, decision):
            if replica.drain_requested_at is not None:
                continue
            prediction = self._admission_prediction(decision, replica)
            if prediction is None:
                continue
            scored.append(
                (prediction.predicted_run_sec, replica.replica_id, replica, prediction)
            )
        if not scored:
            return None
        # A busier replica prices into a higher batch bucket, so the batched run
        # estimate is itself the load signal. Ties break on the deterministic id.
        best = min(scored, key=lambda entry: (entry[0], entry[1]))
        return best[2], best[3]

    def _is_redundant(self, replica: ModelReplicaRecord, now: float) -> bool:
        """Whether reclaiming this replica leaves its model still resident somewhere.

        Evicting one member drops the group to a single copy, which stops the rest of
        the group qualifying, so this cannot cascade a model out of residency.
        """
        if not self.scheduler_config.elastic_replicas:
            return False
        if replica.drain_requested_at is not None:
            return True
        if replica.state != ModelReplicaState.IDLE:
            return False
        if self._replica_groups.size((replica.model_key, replica.gpu_kind)) < 2:
            return False
        idle_since = replica.idle_since
        if idle_since is None:
            return False
        return now - idle_since >= self.scheduler_config.scale_in_idle_sec

    def _lease_finish_times(self, replica: ModelReplicaRecord) -> list[float]:
        return [
            self._grant_finish_at[acquire_id]
            for acquire_id in replica.active_acquire_ids
            if acquire_id in self._grant_finish_at
        ]

    def _admissible_slots(
        self,
        replica: ModelReplicaRecord,
        decision: PlacementDecision,
    ) -> int:
        """Requests this replica can serve at once, as the cache actually covers them.

        max_num_seqs is only an upper bound: admission needs an exact batch-k contract
        and the memory to hold it, so a model whose cache stops at batch 1 for this
        prompt length really does serve one request at a time.
        """
        predictions = self.predictions
        key = decision.prediction_key
        gpu_kind = decision.gpu_kind
        if predictions is None or key is None or gpu_kind is None:
            return 1
        accelerator = self.accelerators[replica.accelerator_ids[0]].config
        slots = 0
        for batch_size in range(1, replica.deployment.serving.max_num_seqs + 1):
            try:
                prediction = predictions.lookup_decode(
                    model_name=replica.deployment.model_name,
                    gpu_kind=gpu_kind,
                    batch_size=batch_size,
                    sequence_length=key.sequence_length,
                    decode_output_length=key.decode_output_length,
                )
            except KeyError:
                break
            effective_vram_mb = (
                prediction.peak_vram_mb_upper_bound
                + self.scheduler_config.eps_mem_mb
                + self.oom_penalties.get(prediction.key, 0.0)
            )
            if effective_vram_mb > accelerator.total_mem_mb:
                break
            slots = batch_size
        return max(1, slots)

    def _pair_queue_depth(
        self,
        pending: PendingAcquire,
        decision: PlacementDecision,
        replicas: Sequence[ModelReplicaRecord],
    ) -> int:
        """Requests contending for this model ahead of, and including, this one."""
        model_key = self._pending_model_key(pending)
        waiting = sum(
            1
            for other in self.pending_acquires.values()
            if other.request_seq <= pending.request_seq
            and self._pending_model_key(other) == model_key
        )
        inflight = sum(len(replica.active_acquire_ids) for replica in replicas)
        return waiting + inflight

    def _best_existing_ect(
        self,
        pending: PendingAcquire,
        decision: PlacementDecision,
        now: float,
    ) -> float | None:
        """When this request would finish if it waits for the resident replicas.

        The queue ahead of it is the term that justifies an extra replica: a fan-out
        node puts many requests on one model at once, and they drain a service round
        at a time. Rounds are priced with the cache's own run estimate, and the first
        turnover uses the live lease estimates rather than a full round.
        """
        run_sec = decision.predicted_run_sec
        if run_sec is None:
            return None
        replicas = [
            replica
            for replica in self._group_replicas(pending, decision)
            if replica.drain_requested_at is None
            and replica.state
            not in (ModelReplicaState.EVICTING, ModelReplicaState.SUSPECT)
        ]
        if not replicas:
            return None
        slots = sum(self._admissible_slots(replica, decision) for replica in replicas)
        depth = self._pair_queue_depth(pending, decision, replicas)
        rounds_ahead = max(0, (depth - 1) // max(1, slots))
        if rounds_ahead == 0:
            return now + run_sec
        finishes = [
            finish
            for replica in replicas
            for finish in self._lease_finish_times(replica)
        ]
        first_turnover = max(now, min(finishes)) if finishes else now + run_sec
        return first_turnover + (rounds_ahead - 1) * run_sec + run_sec

    def _scale_out_gain(
        self,
        pending: PendingAcquire,
        decision: PlacementDecision,
        now: float,
    ) -> float | None:
        load_sec = decision.predicted_load_sec
        run_sec = decision.predicted_run_sec
        if load_sec is None or run_sec is None:
            return None
        best_existing = self._best_existing_ect(pending, decision, now)
        if best_existing is None:
            return None
        return best_existing - (now + load_sec + run_sec)

    def _scale_out_warranted(
        self,
        pending: PendingAcquire,
        decision: PlacementDecision,
        now: float,
    ) -> bool:
        if not self.scheduler_config.elastic_replicas:
            return False
        gain = self._scale_out_gain(pending, decision, now)
        return gain is not None and gain > self.scheduler_config.scale_out_margin_sec

    def _demand_requirements(
        self,
        decisions: Mapping[str, PlacementDecision],
        near: Sequence[NearReadyTask],
    ) -> dict[tuple[str, GpuKind], float]:
        """Memory each model with queued or imminent work needs, keyed by pair."""
        requirements: dict[tuple[str, GpuKind], float] = {}
        for acquire_id, decision in decisions.items():
            pending = self.pending_acquires.get(acquire_id)
            if pending is None:
                continue
            model_key = self._pending_model_key(pending)
            gpu_kind = decision.gpu_kind
            vram = decision.effective_vram_mb
            if model_key is None or gpu_kind is None or vram is None:
                continue
            pair = (model_key, gpu_kind)
            requirements[pair] = max(requirements.get(pair, 0.0), vram)
        for candidate in near:
            gpu_kind = candidate.decision.gpu_kind
            vram = candidate.decision.effective_vram_mb
            if gpu_kind is None or vram is None:
                continue
            pair = (candidate.deployment.model_key, gpu_kind)
            requirements[pair] = max(requirements.get(pair, 0.0), vram)
        return requirements

    def _scale_out_preserves_feasibility(
        self,
        scaling_pair: tuple[str, GpuKind],
        consumed_accelerator_id: str,
        requirements: Mapping[tuple[str, GpuKind], float],
    ) -> bool:
        """Reject a scale-out that would leave some demanding model with nowhere to go.

        Only models that are not resident anywhere need a slot reserved: a resident
        model already has a home this scale-out cannot take (its replica is not an
        eviction candidate while it holds leases, and if it is idle it still counts).
        """
        for pair, vram_mb in requirements.items():
            if pair == scaling_pair:
                continue
            model_key, gpu_kind = pair
            if self._replica_groups.size(pair) > 0:
                continue
            usable = 0
            for accelerator in self.accelerators.values():
                if accelerator.config.gpu_kind != gpu_kind:
                    continue
                if accelerator.config.total_mem_mb < vram_mb:
                    continue
                if accelerator.config.accelerator_id == consumed_accelerator_id:
                    continue
                if accelerator.replica_id is None:
                    usable += 1
                    continue
                resident = self.replicas[accelerator.replica_id]
                if resident.model_key != model_key and resident.state in (
                    ModelReplicaState.IDLE,
                    ModelReplicaState.SUSPECT,
                ):
                    usable += 1
            if usable == 0:
                return False
        return True

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
        *,
        reason: Literal["ready_load", "scale_out"] = "ready_load",
        scale_out_gain_sec: float | None = None,
    ) -> LoadReplicaAction:
        gpu_kind = decision.gpu_kind
        if gpu_kind is None:
            raise RuntimeError("feasible placement has no GPU kind")
        task = self.tasks[pending.task_id]
        workflow_name = self.sessions[task.session_id].workflow_name
        node = self._nodes[workflow_name][task.node_id]
        if not isinstance(node, AgentNodeConfig):
            raise RuntimeError("pending acquire belongs to a function task")
        deployment = ModelDeploymentConfig.from_node(node)
        pair = (deployment.model_key, gpu_kind)
        if self._replica_groups.size(pair) >= self._replica_group_cap:
            raise RuntimeError(f"replica group is at capacity: {pair}")
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

        replica_id = self._new_replica_id()
        replica = ModelReplicaRecord(
            replica_id=replica_id,
            deployment=deployment,
            model_key=deployment.model_key,
            gpu_kind=gpu_kind,
            accelerator_ids=(accelerator.config.accelerator_id,),
            state=ModelReplicaState.LOADING,
            owner_workflow=workflow_name,
            expected_load_sec=expected_load_sec,
            created_at=now,
        )
        accelerator.replica_id = replica_id
        self.replicas[replica_id] = replica
        self._replica_groups.add(pair, replica_id)
        self._validate_resource_ledger()
        return LoadReplicaAction(
            replica_id=replica_id,
            deployment=deployment,
            accelerator=accelerator.config,
            reason=reason,
            expected_load_sec=expected_load_sec,
            owner_workflow=workflow_name,
            scale_out_gain_sec=_finite_or_none(scale_out_gain_sec),
            group_size=self._replica_groups.size(pair),
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
        if self._replica_groups.size(pair) >= self._replica_group_cap:
            raise RuntimeError(f"replica group is at capacity: {pair}")
        if accelerator.replica_id is not None:
            raise RuntimeError("accelerator is already reserved")

        workflow_name = self.sessions[near.session_id].workflow_name
        replica_id = self._new_replica_id()
        replica = ModelReplicaRecord(
            replica_id=replica_id,
            deployment=near.deployment,
            model_key=near.deployment.model_key,
            gpu_kind=gpu_kind,
            accelerator_ids=(accelerator.config.accelerator_id,),
            state=ModelReplicaState.LOADING,
            owner_workflow=workflow_name,
            expected_load_sec=near.load_sec,
            created_at=now,
        )
        accelerator.replica_id = replica_id
        self.replicas[replica_id] = replica
        self._replica_groups.add(pair, replica_id)
        self._validate_resource_ledger()
        return LoadReplicaAction(
            replica_id=replica_id,
            deployment=near.deployment,
            accelerator=accelerator.config,
            reason="near_ready_prefetch",
            expected_load_sec=near.load_sec,
            owner_workflow=workflow_name,
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
        replica_id = self._replica_groups.first((near.deployment.model_key, gpu_kind))
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
            workflow_name = self.sessions[task.session_id].workflow_name
            node = self._nodes[workflow_name][task.node_id]
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

    def _min_residency_sec(self, replica: ModelReplicaRecord) -> float:
        """How long this replica must hold its device before it may be displaced.

        Priced in what the replica itself cost to load, so it scales with the model
        rather than needing a constant: a 6s load earns 6s of tenure, a 257s load earns
        257s. A replica with no measured load duration is not protected, because
        blocking eviction on a missing field would strand the device indefinitely.
        """
        multiple = self.scheduler_config.min_residency_load_multiple
        load_duration_sec = replica.load_duration_sec
        if multiple <= 0 or load_duration_sec is None:
            return 0.0
        return multiple * load_duration_sec

    def _within_min_residency(self, replica: ModelReplicaRecord, now: float) -> bool:
        # created_at is when the device was committed to this model, so the load time is
        # inside the window: tenure is counted from the moment the card stopped being
        # available to anyone else. A zero minimum short-circuits rather than comparing,
        # so a disabled guard cannot react to a replica whose clock reads ahead of now.
        minimum_sec = self._min_residency_sec(replica)
        return minimum_sec > 0 and now - replica.created_at < minimum_sec

    def _start_eviction_for_decision(
        self,
        decision: PlacementDecision,
        *,
        reason: Literal["ready_load", "near_ready_prefetch", "scale_out"],
        now: float,
        protected_pairs: set[tuple[str, GpuKind]],
        near: list[NearReadyTask],
        prefetch_candidate: NearReadyTask | None = None,
        scale_out: _ScaleOutBid | None = None,
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
            redundant = self._is_redundant(replica, now)
            # Protection covers a pair's residency, not every copy of it: a scaled-out
            # group that still has queued work would otherwise shield its own spare
            # replicas and make them unreclaimable.
            if pair in protected_pairs and not redundant:
                continue
            if replica.state not in (ModelReplicaState.IDLE, ModelReplicaState.SUSPECT):
                continue
            if replica.active_acquire_ids:
                continue
            if not redundant and self._within_min_residency(replica, now):
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
                    redundant=redundant,
                )
            )
        victim = select_eviction_victim(candidates)
        if victim is None:
            return None
        if scale_out is not None and not self._scale_out_clears_gates(
            scale_out, victim
        ):
            return None
        return self._mark_evicting(
            self.replicas[victim.replica_id],
            reason=reason,
            reuse_distance_sec=victim.reuse_distance_sec,
            reload_cost_sec=victim.reload_cost_sec,
            prefetch_candidate=prefetch_candidate,
            scale_out_gain_sec=None if scale_out is None else scale_out.gain_sec,
        )

    def _scale_out_clears_gates(
        self,
        bid: _ScaleOutBid,
        victim: EvictionCandidate,
    ) -> bool:
        """Charge a contended scale-out for the disruption it causes.

        Gate 2 makes the extra replica save more time than the victim spends coming
        back; a redundant sibling costs nothing to displace. Marginal gain shrinks with
        every copy of a model while reload cost does not, so replica counts settle
        without a quota doing the work. Gate 3 then refuses to take the last card a
        model with waiting work could have run on.
        """
        if not victim.redundant:
            if victim.reload_cost_sec is None:
                return False
            if bid.gain_sec <= victim.reload_cost_sec:
                return False
            # The victim must not be wanted back before the new replica has even
            # finished loading. Pricing the gain alone let a scale-out take the card
            # of a model whose reuse distance was zero, which relocates the queue
            # instead of draining it and leaves the pool permanently over-subscribed.
            horizon = self.scheduler_config.min_residency_load_multiple * bid.load_sec
            if horizon > 0 and (
                victim.reuse_distance_sec is None or victim.reuse_distance_sec < horizon
            ):
                return False
        accelerator_id = self.replicas[victim.replica_id].accelerator_ids[0]
        return self._scale_out_preserves_feasibility(
            bid.pair, accelerator_id, bid.requirements
        )

    def _start_drain_for_starved(
        self,
        starved: Sequence[tuple[PendingAcquire, PlacementDecision]],
        now: float,
    ) -> None:
        """Free a card for a model that has queued work but no legal eviction victim.

        Every candidate card is busy, so nothing can be evicted outright. A model
        holding more than one replica can give one up instead: stop feeding it new
        leases and it drains to IDLE, where normal eviction reclaims it.
        """
        if not self.scheduler_config.elastic_replicas:
            return
        for pending, decision in starved:
            gpu_kind = decision.gpu_kind
            effective_vram_mb = decision.effective_vram_mb
            model_key = self._pending_model_key(pending)
            if gpu_kind is None or effective_vram_mb is None or model_key is None:
                continue
            candidates: list[ModelReplicaRecord] = []
            for accelerator in self.accelerators.values():
                if accelerator.config.gpu_kind != gpu_kind:
                    continue
                if accelerator.config.total_mem_mb < effective_vram_mb:
                    continue
                if accelerator.replica_id is None:
                    continue
                replica = self.replicas[accelerator.replica_id]
                if replica.model_key == model_key:
                    continue
                if replica.drain_requested_at is not None:
                    continue
                if replica.state != ModelReplicaState.BUSY:
                    continue
                if self._replica_groups.size((replica.model_key, replica.gpu_kind)) < 2:
                    continue
                candidates.append(replica)
            if not candidates:
                continue
            victim = min(
                candidates,
                key=lambda replica: (
                    len(replica.active_acquire_ids),
                    replica.replica_id,
                ),
            )
            victim.drain_requested_at = now
            self._drain_events.append(
                DrainEvent(
                    replica_id=victim.replica_id,
                    model_key=victim.model_key,
                    gpu_kind=victim.gpu_kind,
                    accelerator_id=victim.accelerator_ids[0],
                    active_acquire_count=len(victim.active_acquire_ids),
                    requested_by_model_key=model_key,
                )
            )
            return

    def _scale_out_bid(
        self,
        pending: PendingAcquire,
        decision: PlacementDecision,
        requirements: Mapping[tuple[str, GpuKind], float],
        now: float,
    ) -> _ScaleOutBid | None:
        """The case for giving this request a second replica, or None if it has none."""
        gpu_kind = decision.gpu_kind
        model_key = self._pending_model_key(pending)
        if gpu_kind is None or model_key is None:
            return None
        pair = (model_key, gpu_kind)
        if not 0 < self._replica_groups.size(pair) < self._replica_group_cap:
            return None
        if not self._scale_out_warranted(pending, decision, now):
            return None
        gain = self._scale_out_gain(pending, decision, now)
        load_sec = decision.predicted_load_sec
        if gain is None or load_sec is None:
            return None
        return _ScaleOutBid(
            pair=pair,
            gain_sec=gain,
            load_sec=load_sec,
            requirements=requirements,
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
            "scale_out",
        ],
        reuse_distance_sec: float | None,
        reload_cost_sec: float | None,
        prefetch_candidate: NearReadyTask | None = None,
        scale_out_gain_sec: float | None = None,
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
            scale_out_gain_sec=_finite_or_none(scale_out_gain_sec),
        )

    def _future_reuse_distance(
        self,
        replica: ModelReplicaRecord,
        now: float,
        near: list[NearReadyTask],
    ) -> float | None:
        # A high-weight workflow's demand shrinks the effective distance (more
        # protected from eviction); a low-weight workflow's demand shrinks it
        # less. Equal weights (the default) leave raw distances unchanged.
        # Under the scoped ablation only the workflow that loaded this replica can
        # keep it alive, so another workflow wanting the same model no longer
        # extends its residency.
        scope = (
            None
            if self.scheduler_config.cross_workflow_lifecycle
            else replica.owner_workflow
        )
        distances = [
            max(0.0, candidate.upstream_eta - now)
            / self._workflow_weights.get(
                self.sessions[candidate.session_id].workflow_name, 1.0
            )
            for candidate in near
            if candidate.deployment.model_key == replica.model_key
            and candidate.decision.gpu_kind == replica.gpu_kind
            and (
                scope is None
                or self.sessions[candidate.session_id].workflow_name == scope
            )
        ]
        if distances:
            return min(distances)
        # No upstream is running yet, so there is no ETA to read. Returning None here
        # would drop the replica out of the comparable set and silently degrade the
        # whole decision to LRU, which is what it must not do: fall back to the static
        # earliest-start depth of the nearest node that still wants this model, offset
        # by how far its session has already progressed.
        for session in self.sessions.values():
            if session.state != SessionState.ACTIVE:
                continue
            if scope is not None and session.workflow_name != scope:
                continue
            nodes = self._nodes[session.workflow_name]
            depth = self._node_depth.get(session.workflow_name, {})
            weight = self._workflow_weights.get(session.workflow_name, 1.0)
            progress = self._session_progress(session)
            for node_id, task_id in session.task_ids.items():
                task = self.tasks[task_id]
                node = nodes[node_id]
                if task.state != NodeTaskState.PENDING:
                    continue
                if not isinstance(node, AgentNodeConfig):
                    continue
                if ModelDeploymentConfig.from_node(node).model_key != replica.model_key:
                    continue
                distances.append(max(0.0, depth.get(node_id, 0.0) - progress) / weight)
        if distances:
            return min(distances)
        return inf

    def _session_progress(self, session: SessionRecord) -> float:
        # How far this session has advanced along the static cost model: the latest
        # finish depth among its completed nodes.
        depth = self._node_depth.get(session.workflow_name, {})
        costs = self._node_costs.get(session.workflow_name, {})
        return max(
            (
                depth.get(node_id, 0.0) + costs.get(node_id, 0.0)
                for node_id, task_id in session.task_ids.items()
                if self.tasks[task_id].state == NodeTaskState.COMPLETED
            ),
            default=0.0,
        )

    def _reload_cost(self, replica: ModelReplicaRecord) -> float | None:
        if self.scheduler_config.policy in ("fifo", "kairos"):
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

    def _pending_model_key(self, pending: PendingAcquire) -> str | None:
        task = self.tasks[pending.task_id]
        workflow_name = self.sessions[task.session_id].workflow_name
        node = self._nodes[workflow_name][task.node_id]
        if not isinstance(node, AgentNodeConfig):
            return None
        return ModelDeploymentConfig.from_node(node).model_key

    def _order_by_load_yield(
        self,
        blocked: Sequence[tuple[PendingAcquire, PlacementDecision]],
        near: Sequence[NearReadyTask],
        now: float,
    ) -> list[tuple[PendingAcquire, PlacementDecision]]:
        # A load is the dominant cost in this system, so when one has to be paid the
        # queue head is the wrong thing to serve: pick the model whose transfer unlocks
        # the most already-queued and imminent work per second spent loading it. Ties
        # and unknown load costs fall back to the policy's request order, and the
        # oldest waiting request always outranks yield so nothing starves.
        if self.scheduler_config.policy != "cache" or len(blocked) < 2:
            return list(blocked)
        # Scoping the tally makes a load pay for itself out of one workflow's queue
        # only, so a model two workflows both want stops outranking a model one
        # workflow wants just as badly.
        demand: dict[tuple[str | None, str], int] = {}
        for pending in self.pending_acquires.values():
            model_key = self._pending_model_key(pending)
            if model_key is not None:
                key = self._demand_key(self._pending_workflow_name(pending), model_key)
                demand[key] = demand.get(key, 0) + 1
        for candidate in near:
            key = self._demand_key(
                self.sessions[candidate.session_id].workflow_name,
                candidate.deployment.model_key,
            )
            demand[key] = demand.get(key, 0) + 1
        starvation_sec = 0.5 * self.scheduler_config.acquire_timeout_sec
        ranked: list[tuple[float, int, tuple[PendingAcquire, PlacementDecision]]] = []
        for order, entry in enumerate(blocked):
            pending, decision = entry
            model_key = self._pending_model_key(pending)
            load_sec = decision.predicted_load_sec
            if model_key is None or load_sec is None:
                yield_score = 0.0
            else:
                key = self._demand_key(self._pending_workflow_name(pending), model_key)
                yield_score = demand.get(key, 1) / load_sec
            if now - pending.created_at > starvation_sec:
                yield_score = inf
            ranked.append((-yield_score, order, entry))
        ranked.sort(key=lambda item: (item[0], item[1]))
        return [item[2] for item in ranked]

    def _expected_run_sec(
        self,
        *,
        workflow_name: str,
        node: AgentNodeConfig,
        prediction: ResourceContract,
        model_key: str,
        gpu_kind: GpuKind,
        batch_size: int,
    ) -> float:
        # Admission prices a forced full-length generation because VRAM must cover
        # the worst case, but a request stops at its own EOS. Charging that bucket to
        # the timing path overstates decode several-fold and desynchronises every ETA
        # derived from it, so re-price at the node's observed output length instead.
        predictions = self.predictions
        if predictions is None:
            return prediction.predicted_run_sec
        sequence_length = prediction.key.sequence_length
        # A fused node holds its lease across every stage, so the admission bucket
        # (the widest single stage) is not the occupancy; the stage sum is.
        cold = prediction.predicted_run_sec
        if isinstance(node, FusedAgentNodeConfig):
            cold = (
                self._chain_run_sec(node, gpu_kind, sequence_length, batch_size) or cold
            )
        history = self.history.get((workflow_name, node.name, model_key, gpu_kind))
        if history is None or not history.samples:
            return cold
        # Observed output tokens already sum across a fused chain, so re-pricing them
        # as one generation tracks the whole occupancy.
        chain_limit = sum(self._stage_outputs(node))
        expected_tokens = min(history.output_tokens_p90, chain_limit)
        if expected_tokens <= 0:
            return cold
        bucket = next(
            (
                candidate
                for candidate in predictions.decode_output_lengths(
                    node.model.name, gpu_kind, sequence_length, batch_size
                )
                if candidate >= expected_tokens
            ),
            None,
        )
        if bucket is None:
            return cold
        return predictions.lookup_decode(
            model_name=node.model.name,
            gpu_kind=gpu_kind,
            batch_size=batch_size,
            sequence_length=sequence_length,
            decode_output_length=bucket,
        ).predicted_run_sec

    def _grant(
        self,
        pending: PendingAcquire,
        decision: PlacementDecision,
        prediction: ResourceContract,
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
        workflow_name = self.sessions[task.session_id].workflow_name
        node = self._nodes[workflow_name][task.node_id]
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
            duration = self._expected_run_sec(
                workflow_name=workflow_name,
                node=node,
                prediction=prediction,
                model_key=replica.model_key,
                gpu_kind=gpu_kind,
                batch_size=admitted_batch_size,
            )
        elif self.scheduler_config.policy == "history":
            history = self.duration_history.get(
                (
                    workflow_name,
                    task.node_id,
                    replica.model_key,
                    gpu_kind,
                    admitted_batch_size,
                )
            )
            if history is not None:
                duration = history.duration_sec_ema
        if duration is not None:
            self._grant_finish_at[pending.acquire_id] = now + duration
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
    ) -> ResourceContract | None:
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
            prediction.peak_vram_mb_upper_bound
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
        self.record_runtime_report(report, self.sessions[task.session_id].workflow_name)
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
            node = self._nodes[session.workflow_name][task.node_id]
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
        self._grant_finish_at.pop(grant.acquire_id, None)
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
        active_acquires: set[str] = set()
        for pair, members in self._replica_groups.items():
            if len(set(members)) != len(members):
                raise RuntimeError("model replica pairs index is inconsistent")
            for replica_id in members:
                replica = self.replicas.get(replica_id)
                if replica is None or (replica.model_key, replica.gpu_kind) != pair:
                    raise RuntimeError("model replica pairs index is inconsistent")
            if len(members) > self._replica_group_cap:
                raise RuntimeError("model replica pairs must be unique")
        for replica in self.replicas.values():
            pair = (replica.model_key, replica.gpu_kind)
            if replica.replica_id not in self._replica_groups.members(pair):
                raise RuntimeError("model replica pairs must be unique")
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
        if not set(self._grant_finish_at) <= set(self.grants):
            raise RuntimeError("grant finish estimates outlive their grants")
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
        predictions: ResourceContractCache | None,
        trace_writer: object,
        replica_factory: object | None = None,
        ray_node_ids: Mapping[str, str] | None = None,
        run_id: str = "workflow-run",
        priority_weight: float = 1.0,
    ) -> None:
        self.core = SchedulerCore(
            workflow,
            scheduler_config=scheduler_config,
            predictions=predictions,
            priority_weight=priority_weight,
        )
        self.run_id = run_id
        self._config = scheduler_config
        self._trace_writer = trace_writer
        self._replica_factory = replica_factory
        self._ray_node_ids = dict(ray_node_ids or {})
        self._commands: asyncio.Queue[_SchedulerCommand] = asyncio.Queue()
        self._workers: dict[str, dict[str, object]] = {workflow.workflow_name: {}}
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
        workflow_name: str,
        submitted_at: float,
        require_latency: bool = False,
    ) -> None:
        await self._submit(
            "register_session",
            session_id,
            workflow_name,
            submitted_at,
            require_latency,
        )

    async def register_workflow(
        self,
        workflow: Workflow,
        *,
        priority_weight: float = 1.0,
    ) -> None:
        await self._submit("register_workflow", workflow, priority_weight)

    async def deregister_workflow(self, workflow_name: str) -> None:
        await self._submit("deregister_workflow", workflow_name)

    async def register_workers(
        self,
        workflow_name: str,
        workers: Mapping[str, object],
    ) -> None:
        await self._submit("register_workers", workflow_name, dict(workers))

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

    async def drain_complete(self, workflow_name: str | None = None) -> bool:
        latency_required = self._latency_required
        if workflow_name is not None:
            workflow_sessions = {
                session.session_id
                for session in self.core.sessions.values()
                if session.workflow_name == workflow_name
            }
            latency_required = latency_required & workflow_sessions
        return (
            self.core.drain_complete(workflow_name)
            and latency_required <= self._session_latencies.keys()
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
            workflow_name = cast(str, args[1])
            submitted_at = cast(float, args[2])
            require_latency = cast(bool, args[3])
            self.core.register_session(session_id, workflow_name)
            self._terminal_events[session_id] = asyncio.Event()
            if require_latency:
                self._latency_required.add(session_id)
            self._record_trace(
                "session_submitted",
                ts=submitted_at,
                session_id=session_id,
            )
            return None
        if name == "register_workflow":
            workflow = cast(Workflow, args[0])
            priority_weight = cast(float, args[1])
            self.core.register_workflow(workflow, priority_weight=priority_weight)
            self._workers.setdefault(workflow.workflow_name, {})
            self._record_trace(
                "workflow_registered",
                workflow_name=workflow.workflow_name,
                payload={
                    "workflow_name": workflow.workflow_name,
                    "priority_weight": priority_weight,
                },
            )
            return None
        if name == "deregister_workflow":
            workflow_name = cast(str, args[0])
            self.core.deregister_workflow(workflow_name)
            self._workers.pop(workflow_name, None)
            self._record_trace(
                "workflow_deregistered",
                workflow_name=workflow_name,
                payload={"workflow_name": workflow_name},
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
            workflow_name = cast(str, args[0])
            workers = cast(dict[str, object], args[1])
            if workflow_name not in self._workers:
                raise KeyError(f"unknown workflow: {workflow_name}")
            self._workers[workflow_name] = dict(workers)
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
        session = self.core._session(session_id)
        graph = self.core._workflow(session.workflow_name).graph
        if source_node is None:
            if target_node != graph.entry_node:
                raise ValueError("external item must target the workflow entry")
            return
        if target_node not in graph.adjacency[source_node]:
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
            if report.stage_reports:
                # Fused runs report one task per chain; keeping the per-stage rows
                # lets analysis bucket them under their original node ids so a fused
                # arm stays comparable with an unfused one.
                payload["stages"] = [
                    stage.model_dump(mode="json") for stage in report.stage_reports
                ]
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
        self._record_drain_decisions(now)
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
            workflow_name = self.core.sessions[task.session_id].workflow_name
            node = self.core._nodes[workflow_name][task.node_id]
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
                        "group_size": action.group_size,
                        "scale_out_gain_sec": action.scale_out_gain_sec,
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
                    "scale_out_gain_sec": action.scale_out_gain_sec,
                    # The score the victim actually lost on. Without it, scoping the
                    # reuse distance to one workflow leaves no trace evidence at all,
                    # and the cross_workflow_lifecycle arm cannot be audited.
                    # inf means "nobody in scope wants this model" and is not JSON.
                    "owner_workflow": replica.owner_workflow,
                    "reuse_distance_sec": _finite_or_none(action.reuse_distance_sec),
                    "reload_cost_sec": _finite_or_none(action.reload_cost_sec),
                },
            )

    def _record_drain_decisions(self, now: float) -> None:
        for event in self.core.take_drain_events():
            self._record_trace(
                "scheduler_decision",
                ts=now,
                replica_id=event.replica_id,
                accelerator_id=event.accelerator_id,
                gpu_kind=event.gpu_kind,
                model_key=event.model_key,
                payload={
                    "action_type": "drain_replica",
                    "reason": "reclaim_for_starved_model",
                    "active_acquire_count": event.active_acquire_count,
                    "requested_by_model_key": event.requested_by_model_key,
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
        ready_work_priority = any(
            action.reason in ("ready_load", "scale_out") for action in actions
        )
        for candidate in near:
            identity = (
                candidate.session_id,
                candidate.node_id,
                candidate.prefetch_at,
            )
            if now < candidate.prefetch_at or identity in started:
                continue
            replica = self.core._replica_for_near(candidate)
            if not self._config.enable_prefetch:
                # Counts the prefetches the full system would have issued here, which
                # is what the ablation is measured against.
                reason = "disabled"
            elif replica is not None:
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
            load_ref = dispatch(handle, "load")
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
                # model_key is a hash of the whole deployment; carrying the name keeps
                # residency tables groupable by model without reversing it.
                "model_name": action.deployment.model_name,
                "owner_workflow": action.owner_workflow,
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
            value = await await_value(load_ref)
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
        shutdown_ref = dispatch(handle, "shutdown")
        with suppress(Exception):
            await asyncio.wait_for(
                await_value(shutdown_ref),
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
            workflow_name = self.core.sessions[action.session_id].workflow_name
            refs: list[object] = []
            try:
                refs.extend(
                    dispatch(worker, "cancel_session", action.session_id)
                    for worker in self._workers.get(workflow_name, {}).values()
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
                await await_value(ref)
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
        if "workflow_name" not in fields:
            session_id = fields.get("session_id")
            if isinstance(session_id, str):
                session = self.core.sessions.get(session_id)
                if session is not None:
                    fields["workflow_name"] = session.workflow_name
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
            ref = dispatch(self._trace_writer, "append_batch", events)
        except Exception as error:
            self._enqueue_internal_nowait("trace_failed", str(error))
            return
        self._start_watcher(self._watch_trace(ref, expected))

    async def _watch_trace(self, ref: object, expected: int) -> None:
        try:
            acknowledged = await await_value(ref)
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
        shutdown_ref = dispatch(handle, "shutdown")
        with suppress(Exception):
            await asyncio.wait_for(
                await_value(shutdown_ref),
                timeout=self._config.eviction_timeout_sec,
            )
        await asyncio.wait_for(
            asyncio.to_thread(ray.kill, cast(Any, handle), no_restart=True),
            timeout=self._config.eviction_timeout_sec,
        )


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
SchedulerActor = _remote_with_options(max_concurrency=1024)(_SchedulerActor)
