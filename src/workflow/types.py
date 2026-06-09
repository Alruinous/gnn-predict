from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Literal

NodeType = Literal[
    "input",
    "output",
    "main_agent",
    "object_detection",
    "image_classification",
    "vision_transformer",
    "text_encoder",
    "text_generation",
]
WorkflowPhase = Literal["training", "inference", "prefill", "decode"]
WorkflowRunMode = Literal["serial", "parallel", "adaptive"]
WorkflowNodeStatus = Literal["pending", "running", "completed", "failed"]
WorkflowWorkerSignal = Literal["done"]


@dataclass
class WorkflowQueueItem:
    node: WorkflowNode | None = None
    signal: WorkflowWorkerSignal | None = None


class NodeRuntime:
    batch_size: int
    phase: WorkflowPhase
    input_shape: list[int] | None = None
    sequence_length: int | None = None
    decode_max_output_length: int | None = None


class WorkflowNode(ABC):
    name: str
    type: NodeType
    model: dict[str, Any] | None = None
    runtime: NodeRuntime | None = None
    status: WorkflowNodeStatus = "pending"
    previous_nodes: list[WorkflowNode]
    next_nodes: list[WorkflowNode]

    @abstractmethod
    def run(self): ...


class WorkflowEdge:
    source: str
    target: str
    attributes: dict[str, Any]


class Workflow:
    nodes: list[WorkflowNode]
    edges: list[WorkflowEdge]

    def __post_init__(self):
        assert (
            self.input_node is not None and len(self.input_node.previous_nodes) == 0
        ), "input node must not have previous nodes"
        assert self.output_node is not None and len(self.output_node.next_nodes) == 0, (
            "output node must not have next nodes"
        )

    def get_topological_node_list(self) -> list[WorkflowNode]:
        nodes = [self.input_node]
        i = 0
        while i < len(nodes):
            for next_node in nodes[i].next_nodes:
                nodes.append(next_node)
            i += 1
        return nodes

    @property
    def input_node(self) -> WorkflowNode:
        return next(node for node in self.nodes if node.type == "input")

    @property
    def output_node(self) -> WorkflowNode:
        return next(node for node in self.nodes if node.type == "output")


class WorkflowExperimentResult:
    # TODO
    pass
