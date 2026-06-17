from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

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


class WorkflowNodeConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: NonEmptyStr
    type: NodeType
    task: NonEmptyStr | None = None
    description: NonEmptyStr | None = None
    model: WorkflowModelConfig | None = None
    runtime: WorkflowRuntimeConfig | None = None


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
