from __future__ import annotations

import hashlib
from collections.abc import Mapping
from enum import StrEnum
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, NonNegativeInt, PositiveInt

from common.validate import NonEmptyStr
from workflow.schema import (
    AgentNodeConfig,
    EdgeConfig,
    ExecutionConfig,
    FunctionNodeConfig,
    NodeConfig,
    RetryConfig,
    Workflow,
    WorkflowGraph,
    WorkflowModelConfig,
)


class FailureRecord(BaseModel):
    node_name: NonEmptyStr
    session_id: NonEmptyStr
    item_id: NonEmptyStr
    attempt: PositiveInt
    error_type: NonEmptyStr
    error_message: str


class WorkflowDataItem(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    session_id: NonEmptyStr
    item_id: NonEmptyStr = Field(default_factory=lambda: str(uuid4()))
    source_node: NonEmptyStr | None
    target_node: NonEmptyStr
    message: Mapping[str, Any]
    session_input_ref: Any


class SessionState(StrEnum):
    ACTIVE = "active"
    COMPLETED = "completed"
    FAILED = "failed"


class NodeTaskState(StrEnum):
    PENDING = "pending"
    ACQUIRING = "acquiring"
    RUNNING = "running"
    EMITTING = "emitting"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class NodeWorkerState(StrEnum):
    IDLE = "idle"
    RUNNING = "running"
    STOPPED = "stopped"


class ModelReplicaState(StrEnum):
    LOADING = "loading"
    IDLE = "idle"
    BUSY = "busy"
    EVICTING = "evicting"
    SUSPECT = "suspect"


class WorkerQueueItem(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    worker_state: NodeWorkerState | None = None
    data: WorkflowDataItem | None = None


class WorkflowStatus(BaseModel):
    node_states: dict[str, NodeWorkerState]
    queue_sizes: dict[str, int]
    failures: list[FailureRecord]


class TraceEvent(BaseModel):
    run_id: str
    event_id: str = Field(default_factory=lambda: str(uuid4()))
    event_seq: NonNegativeInt
    event_type: str
    ts: float
    session_id: str | None = None
    item_id: str | None = None
    task_id: str | None = None
    node_id: str | None = None
    source_node: str | None = None
    target_node: str | None = None
    model_key: str | None = None
    replica_id: str | None = None
    accelerator_id: str | None = None
    gpu_kind: str | None = None
    acquire_id: str | None = None
    payload: dict[str, object] = Field(default_factory=dict)


class WorkflowModelFeatureKey(BaseModel):
    model_name: NonEmptyStr
    phase: Literal["prefill", "decode"]
    gpu_name: NonEmptyStr
    batch_size: PositiveInt
    sequence_length: PositiveInt
    decode_output_length: int  # 0 for prefill, decode_max_output_length for decode

    @property
    def stable_digest(self) -> str:
        payload = "|".join(
            (
                self.model_name,
                self.phase,
                self.gpu_name,
                str(self.batch_size),
                str(self.sequence_length),
                str(self.decode_output_length),
            )
        )
        return hashlib.sha256(payload.encode()).hexdigest()[:16]

    def __hash__(self) -> int:
        return int(self.stable_digest, 16)
