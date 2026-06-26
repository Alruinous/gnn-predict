from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from common.validate import NonEmptyStr, NonNegativeInt, PositiveInt
from workflow.types import (
    NodeType,
    WorkflowNodeStatus,
    WorkflowPhase,
)


class WorkflowRuntimeConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    batch_size: NonNegativeInt
    phase: WorkflowPhase
    input_shape: list[PositiveInt] | None = None
    sequence_length: PositiveInt | None = None
    decode_max_output_length: PositiveInt | None = None


class WorkflowModelConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: NonEmptyStr
    parameters: dict[str, Any]


class WorkflowExecutionConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model_path: NonEmptyStr
    devices: list[NonEmptyStr]
    dtype: NonEmptyStr = "float16"
    max_new_tokens: PositiveInt | None = None
    do_sample: bool = False
    temperature: float | None = None
    use_chat_template: bool = True
    enable_thinking: bool = False

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


class WorkflowNodeConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: NonEmptyStr
    type: NodeType
    task: NonEmptyStr | None = None
    description: NonEmptyStr | None = None
    prompt_template: NonEmptyStr | None = None
    model: WorkflowModelConfig | None = None
    runtime: WorkflowRuntimeConfig | None = None
    execution: WorkflowExecutionConfig | None = None


class WorkflowEdgeConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source: str
    target: str
    attributes: dict[str, Any] = Field(default_factory=dict)


class Workflow(BaseModel):
    model_config = ConfigDict(extra="forbid")

    nodes: list[WorkflowNodeConfig]
    edges: list[WorkflowEdgeConfig]

    def node_map(self) -> dict[str, WorkflowNodeConfig]:
        return {node.name: node for node in self.nodes}

    def node_names(self) -> list[str]:
        return [node.name for node in self.nodes]


class WorkflowNodeResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    node_name: NonEmptyStr
    status: WorkflowNodeStatus
    started_at: float | None = None
    ended_at: float | None = None
    duration_sec: float | None = None
    error_type: str | None = None
    error_message: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
