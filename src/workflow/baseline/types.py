from __future__ import annotations

from collections import deque
from typing import Any, Literal, Self

from pydantic import BaseModel, Field, PositiveInt, field_validator, model_validator

from common.validate import NonEmptyStr

NodeType = Literal["input", "agent", "tool", "evaluator", "output"]


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
        if self.type in ("agent", "tool"):
            if self.prompt_template is None:
                raise ValueError(
                    f"{self.type} node {self.name} requires prompt_template"
                )
            if self.execution is None:
                raise ValueError(
                    f"{self.type} node {self.name} requires execution config"
                )
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

    def entry_names(self) -> list[str]:
        dependencies = self.dependencies()
        return [name for name in self.node_names() if not dependencies[name]]

    def terminal_name(self) -> str:
        adjacency = self.adjacency()
        terminal_names = [name for name in self.node_names() if not adjacency[name]]
        assert len(terminal_names) == 1, terminal_names
        return terminal_names[0]

    def topological_order(self) -> list[str]:
        node_names = self.node_names()
        adjacency = self.adjacency()
        dependencies = self.dependencies()
        indegree = {name: len(dependencies[name]) for name in node_names}
        ready = deque(name for name in node_names if indegree[name] == 0)
        order: list[str] = []
        while ready:
            source = ready.popleft()
            order.append(source)
            for target in adjacency[source]:
                indegree[target] -= 1
                if indegree[target] == 0:
                    ready.append(target)
        if len(order) != len(node_names):
            raise ValueError("workflow graph must be acyclic")
        return order

    @field_validator("nodes")
    @classmethod
    def validate_unique_node_names(
        cls, value: list[BaselineNodeConfig]
    ) -> list[BaselineNodeConfig]:
        if not value:
            raise ValueError("workflow graph must be non-empty")
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
            if source == target:
                raise ValueError(f"self-edge is not allowed: {source}")
        if len(pairs) != len(set(pairs)):
            raise ValueError("edges must be unique per (source, target) pair")
        self.topological_order()
        terminal_names = [
            name for name, targets in self.adjacency().items() if not targets
        ]
        if len(terminal_names) != 1:
            raise ValueError("workflow graph must have exactly one terminal node")
        return self


class BaselineSessionRequest(BaseModel):
    session_id: NonEmptyStr
    inputs: dict[str, str]


class BaselineTraceEvent(BaseModel):
    session_id: NonEmptyStr
    node_name: NonEmptyStr
    node_type: NodeType
    started_at: float
    ended_at: float
    duration_sec: float
    output_text: str
