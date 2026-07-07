from __future__ import annotations

from typing import Any, Literal, Self

from pydantic import BaseModel, Field, field_validator, model_validator

from common.validate import NonEmptyStr, PositiveInt

NodeType = Literal["input", "tool", "evaluator", "output"]


class BaselineModelConfig(BaseModel):
    name: NonEmptyStr
    parameters: dict[str, Any] = Field(default_factory=dict)


class BaselineRuntimeConfig(BaseModel):
    batch_size: PositiveInt
    sequence_length: PositiveInt | None = None
    decode_max_output_length: PositiveInt | None = None
    phase: Literal["prefill", "decode"] | None = None


class BaselineExecutionConfig(BaseModel):
    model_path: NonEmptyStr
    devices: list[NonEmptyStr]
    dtype: NonEmptyStr = "float16"
    max_new_tokens: PositiveInt | None = None
    do_sample: bool = False
    use_chat_template: bool = True
    enable_thinking: bool = False
    temperature: float | None = None
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


class BaselineNodeConfig(BaseModel):
    name: NonEmptyStr
    type: NodeType
    task: NonEmptyStr | None = None
    description: NonEmptyStr | None = None
    prompt_template: NonEmptyStr | None = None
    system_prompt: NonEmptyStr | None = None
    model: BaselineModelConfig | None = None
    runtime: BaselineRuntimeConfig | None = None
    execution: BaselineExecutionConfig | None = None

    @model_validator(mode="after")
    def validate_tool_fields(self) -> Self:
        if self.type == "tool":
            if self.prompt_template is None:
                raise ValueError(f"tool node {self.name} requires prompt_template")
            if self.execution is None:
                raise ValueError(f"tool node {self.name} requires execution config")
        return self


class BaselineEdgeConfig(BaseModel):
    source: NonEmptyStr
    target: NonEmptyStr
    attributes: dict[str, Any] = Field(default_factory=dict)


class BaselineWorkflow(BaseModel):
    nodes: list[BaselineNodeConfig]
    edges: list[BaselineEdgeConfig]

    def node_map(self) -> dict[str, BaselineNodeConfig]:
        return {node.name: node for node in self.nodes}

    def node_names(self) -> list[str]:
        return [node.name for node in self.nodes]

    def dependencies(self) -> dict[str, list[str]]:
        dependencies: dict[str, list[str]] = {name: [] for name in self.node_names()}
        for edge in self.edges:
            dependencies[edge.target].append(edge.source)
        return dependencies

    def adjacency(self) -> dict[str, list[str]]:
        adjacency: dict[str, list[str]] = {name: [] for name in self.node_names()}
        for edge in self.edges:
            adjacency[edge.source].append(edge.target)
        return adjacency

    @field_validator("nodes")
    @classmethod
    def validate_unique_node_names(
        cls, value: list[BaselineNodeConfig]
    ) -> list[BaselineNodeConfig]:
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


class BaselineTraceEvent(BaseModel):
    node_name: NonEmptyStr
    node_type: NodeType
    started_at: float
    ended_at: float
    duration_sec: float
    output_text: str
