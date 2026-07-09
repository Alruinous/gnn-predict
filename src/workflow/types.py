from __future__ import annotations

import hashlib
from enum import Enum
from typing import Any, Literal, Self
from uuid import uuid4

from langchain.agents import AgentState
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from common.validate import NonEmptyStr, NonNegativeInt, PositiveInt

DEFAULT_QUEUE_CAPACITY = 16


class NodeType(Enum):
    INPUT = "input"
    OUTPUT = "output"
    AGENT = "agent"
    TOOL = "tool"
    EVALUATOR = "evaluator"


class RuntimeConfig(BaseModel):
    batch_size: NonNegativeInt
    input_shape: list[PositiveInt] | None = None
    sequence_length: PositiveInt | None = None
    decode_max_output_length: PositiveInt | None = None


class WorkflowModelConfig(BaseModel):
    name: NonEmptyStr
    parameters: dict[str, Any]


class ExecutionConfig(BaseModel):
    model_name: NonEmptyStr
    model_path: NonEmptyStr
    devices: list[NonEmptyStr]
    dtype: NonEmptyStr = "float16"
    max_new_tokens: PositiveInt | None = None
    do_sample: bool = False
    temperature: float | None = None
    use_chat_template: bool = True
    enable_thinking: bool = False
    truncation_side: Literal["left", "right"] | None = None

    @field_validator("devices")
    @classmethod
    def validate_devices(cls, value: list[str]) -> list[str]:
        if not value:
            raise ValueError("execution devices must not be empty")
        for device in value:
            if not device.startswith("cuda:"):
                raise ValueError("execution devices must use cuda:N syntax")
            index_text = device.split(":", maxsplit=1)[1]
            if not index_text.isdigit():
                raise ValueError("execution devices must use cuda:N syntax")
        if len(value) != len(set(value)):
            raise ValueError("execution devices must be unique")
        return value


class RetryConfig(BaseModel):
    max_attempts: PositiveInt = 1
    retry_delay_sec: NonNegativeInt = 0
    on_exhausted: Literal["fail_workflow", "skip_item"] = "fail_workflow"


class FailureRecord(BaseModel):
    node_name: NonEmptyStr
    session_id: NonEmptyStr
    item_id: NonEmptyStr
    attempt: PositiveInt
    error_type: NonEmptyStr
    error_message: str


class NodeConfig(BaseModel):
    name: NonEmptyStr
    type: NodeType
    model: WorkflowModelConfig
    runtime: RuntimeConfig
    execution: ExecutionConfig
    task: NonEmptyStr | None = None
    description: NonEmptyStr | None = None
    prompt_template: NonEmptyStr | None = None
    system_prompt: NonEmptyStr | None = None
    queue_capacity: PositiveInt = DEFAULT_QUEUE_CAPACITY
    retry: RetryConfig = Field(default_factory=RetryConfig)

    @model_validator(mode="after")
    def validate_execution_presence(self) -> Self:
        if self.type in (NodeType.AGENT, NodeType.TOOL) and self.execution is None:
            raise ValueError(
                f"node {self.name} of type {self.type.value} requires execution config"
            )
        return self


class EdgeConfig(BaseModel):
    source: NonEmptyStr
    target: NonEmptyStr
    attributes: dict[str, Any] = Field(default_factory=dict)


class Workflow(BaseModel):
    """尽可能减少 validate 有问题优先检查 YAML 文件 而不是让代码适配配置文件。"""

    nodes: list[NodeConfig]
    edges: list[EdgeConfig]

    def node_map(self) -> dict[str, NodeConfig]:
        return {node.name: node for node in self.nodes}

    def node_names(self) -> list[str]:
        return [node.name for node in self.nodes]

    @field_validator("nodes")
    @classmethod
    def validate_unique_node_names(cls, value: list[NodeConfig]) -> list[NodeConfig]:
        names = [node.name for node in value]
        if len(names) != len(set(names)):
            raise ValueError("node names must be unique")
        return value

    @model_validator(mode="after")
    def validate_edges(self) -> Self:
        names = set(self.node_names())
        pairs = [(edge.source, edge.target) for edge in self.edges]
        for source, target in pairs:
            if source not in names or target not in names:
                raise ValueError(f"edge {source} -> {target} references unknown node")
        if len(pairs) != len(set(pairs)):
            raise ValueError("edges must be unique per (source, target) pair")
        return self


class WorkflowDataItem:
    session_id: NonEmptyStr
    item_id: NonEmptyStr
    source_node: NonEmptyStr
    target_node: NonEmptyStr
    message: AgentState

    def __init__(
        self,
        session_id: str,
        source_node: str,
        target_node: str,
        message: AgentState,
    ):
        self.session_id = session_id
        self.item_id = str(uuid4())  # 唯一标识
        self.source_node = source_node
        self.target_node = target_node
        self.message = message


class WorkerState(Enum):
    IDLE = "idle"
    RUNNING = "running"
    STOPPED = "stopped"


class WorkerQueueItem(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    worker_state: WorkerState | None = None
    data: WorkflowDataItem | None = None


class WorkflowStatus(BaseModel):
    node_states: dict[str, WorkerState]
    queue_sizes: dict[str, int]
    failures: list[FailureRecord]


class WorkflowModelFeatureKey(BaseModel):
    model_name: NonEmptyStr
    phase: Literal["prefill", "decode"]
    gpu_name: Literal["v100", "a100"]
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
