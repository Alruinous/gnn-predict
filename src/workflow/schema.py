from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from workflow.types import (
    NodeType,
    WorkflowNodeStatus,
    WorkflowPhase,
)


class WorkflowRuntimeConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    batch_size: int
    phase: WorkflowPhase
    input_shape: list[int] | None = None
    sequence_length: int | None = None
    decode_max_output_length: int | None = None


class WorkflowNodeConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    type: NodeType
    description: str | None = None
    model: dict[str, Any] | None = None
    runtime: WorkflowRuntimeConfig | None = None

    @field_validator("description")
    @classmethod
    def validate_description(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized_value = value.strip()
        if not normalized_value:
            raise ValueError("description must not be empty")
        return normalized_value


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

    node_name: str
    status: WorkflowNodeStatus
    started_at: float | None = None
    ended_at: float | None = None
    duration_sec: float | None = None
    error_type: str | None = None
    error_message: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

