from __future__ import annotations

from collections import deque
from collections.abc import Mapping
from types import MappingProxyType
from typing import Annotated, Literal, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    NonNegativeInt,
    PositiveInt,
    PrivateAttr,
    field_validator,
    model_validator,
)

from common.validate import NonEmptyStr

DEFAULT_QUEUE_CAPACITY = 16
OpenUnitInterval = Annotated[float, Field(gt=0, le=1)]


class SchemaModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class WorkflowModelConfig(SchemaModel):
    name: NonEmptyStr
    parameters: dict[str, JsonValue] = Field(default_factory=dict)


class ServingConfig(SchemaModel):
    max_model_len: PositiveInt
    max_num_seqs: PositiveInt
    max_num_batched_tokens: PositiveInt
    gpu_memory_utilization: OpenUnitInterval = 0.98

    @model_validator(mode="after")
    def validate_token_capacity(self) -> Self:
        if self.max_num_batched_tokens < self.max_model_len:
            raise ValueError("max_num_batched_tokens must cover max_model_len")
        return self


class ExecutionConfig(SchemaModel):
    model_path: NonEmptyStr
    max_new_tokens: PositiveInt
    dtype: Literal["float16"] = "float16"
    do_sample: bool = False
    temperature: float | None = None
    use_chat_template: bool = True
    enable_thinking: bool = False
    truncation_side: Literal["left", "right"] | None = None
    serving: ServingConfig

    @model_validator(mode="after")
    def validate_sampling(self) -> Self:
        if self.do_sample and self.temperature is not None and self.temperature <= 0:
            raise ValueError("sampling temperature must be positive")
        return self


class RetryConfig(SchemaModel):
    max_attempts: PositiveInt = 1
    retry_delay_sec: NonNegativeInt = 0


class NodeConfigBase(SchemaModel):
    name: NonEmptyStr
    task: NonEmptyStr | None = None
    description: NonEmptyStr | None = None
    queue_capacity: PositiveInt = DEFAULT_QUEUE_CAPACITY
    retry: RetryConfig = Field(default_factory=RetryConfig)


class AgentNodeConfig(NodeConfigBase):
    type: Literal["agent"]
    model: WorkflowModelConfig
    execution: ExecutionConfig
    prompt_template: NonEmptyStr
    system_prompt: NonEmptyStr | None = None


class FunctionNodeConfig(NodeConfigBase):
    type: Literal["function"]
    function: NonEmptyStr
    parameters: dict[str, JsonValue] = Field(default_factory=dict)
    routing: Literal["broadcast", "targeted"] = "broadcast"
    max_concurrency: PositiveInt = 1


NodeConfig = Annotated[
    AgentNodeConfig | FunctionNodeConfig,
    Field(discriminator="type"),
]


class EdgeConfig(SchemaModel):
    source: NonEmptyStr
    target: NonEmptyStr


class WorkflowGraph(SchemaModel):
    adjacency: Mapping[str, tuple[str, ...]]
    dependencies: Mapping[str, tuple[str, ...]]
    topological_order: tuple[str, ...]
    entry_node: NonEmptyStr
    terminal_node: NonEmptyStr

    @field_validator("adjacency", "dependencies", mode="after")
    @classmethod
    def freeze_mapping(
        cls, value: Mapping[str, tuple[str, ...]]
    ) -> Mapping[str, tuple[str, ...]]:
        return MappingProxyType(dict(value))


class Workflow(SchemaModel):
    nodes: tuple[NodeConfig, ...]
    edges: tuple[EdgeConfig, ...]
    _graph: WorkflowGraph = PrivateAttr()

    @field_validator("nodes")
    @classmethod
    def validate_nodes(cls, value: tuple[NodeConfig, ...]) -> tuple[NodeConfig, ...]:
        if not value:
            raise ValueError("workflow graph must be non-empty")
        names = [node.name for node in value]
        if len(names) != len(set(names)):
            raise ValueError("node names must be unique")
        return value

    @model_validator(mode="after")
    def build_graph(self) -> Self:
        graph = _build_graph(self.nodes, self.edges)
        terminal = next(node for node in self.nodes if node.name == graph.terminal_node)
        if isinstance(terminal, FunctionNodeConfig) and terminal.routing == "targeted":
            raise ValueError("terminal function routing must be broadcast")
        self._graph = graph
        return self

    @property
    def graph(self) -> WorkflowGraph:
        return self._graph

    def node_map(self) -> dict[str, NodeConfig]:
        return {node.name: node for node in self.nodes}

    def node_names(self) -> list[str]:
        return [node.name for node in self.nodes]


def _build_graph(
    nodes: tuple[NodeConfig, ...], edges: tuple[EdgeConfig, ...]
) -> WorkflowGraph:
    names = tuple(node.name for node in nodes)
    name_set = set(names)
    adjacency: dict[str, list[str]] = {name: [] for name in names}
    dependencies: dict[str, list[str]] = {name: [] for name in names}
    edge_pairs: set[tuple[str, str]] = set()

    for edge in edges:
        pair = (edge.source, edge.target)
        if edge.source not in name_set or edge.target not in name_set:
            raise ValueError(
                f"edge {edge.source} -> {edge.target} references unknown node"
            )
        if edge.source == edge.target:
            raise ValueError(f"self-edge is not allowed: {edge.source}")
        if pair in edge_pairs:
            raise ValueError(f"duplicate edge: {edge.source} -> {edge.target}")
        edge_pairs.add(pair)
        adjacency[edge.source].append(edge.target)
        dependencies[edge.target].append(edge.source)

    indegree = {name: len(dependencies[name]) for name in names}
    ready = deque(name for name in names if indegree[name] == 0)
    topological_order: list[str] = []
    while ready:
        source = ready.popleft()
        topological_order.append(source)
        for target in adjacency[source]:
            indegree[target] -= 1
            if indegree[target] == 0:
                ready.append(target)

    if len(topological_order) != len(names):
        raise ValueError("workflow graph must be acyclic")

    entry_nodes = [name for name in names if not dependencies[name]]
    if len(entry_nodes) != 1:
        raise ValueError("workflow graph must have exactly one entry node")
    terminal_nodes = [name for name in names if not adjacency[name]]
    if len(terminal_nodes) != 1:
        raise ValueError("workflow graph must have exactly one terminal node")

    return WorkflowGraph(
        adjacency={name: tuple(targets) for name, targets in adjacency.items()},
        dependencies={name: tuple(sources) for name, sources in dependencies.items()},
        topological_order=tuple(topological_order),
        entry_node=entry_nodes[0],
        terminal_node=terminal_nodes[0],
    )
