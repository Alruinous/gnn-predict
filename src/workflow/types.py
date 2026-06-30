from __future__ import annotations

from collections import deque
from enum import Enum
from typing import Any, Literal, Self

from langchain.agents import AgentState, create_agent
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from ray.util.queue import Queue

from common.validate import NonEmptyStr, NonNegativeInt, PositiveInt

BOUNDARY_NODE_TYPES = {"input", "output"}
MODEL_NODE_TYPES = {"agent", "tool"}
EVALUATOR_TASKS = {
    "gsm8k_numeric_exact_match",
    "mbpp_pass_at_1",
    "summary_rouge",
    "summary_llm_judge",
}
EDGE_CONDITIONS = {"passed", "failed", "default"}


class NodeType(Enum):
    INPUT = "input"
    OUTPUT = "output"
    AGENT = "agent"
    TOOL = "tool"
    EVALUATOR = "evaluator"


class RuntimeConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

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


class NodeConfig(BaseModel):
    name: NonEmptyStr
    type: NodeType
    task: NonEmptyStr | None = None
    description: NonEmptyStr | None = None
    prompt_template: NonEmptyStr | None = None
    model: WorkflowModelConfig | None = None
    runtime: RuntimeConfig | None = None
    execution: ExecutionConfig | None = None

    @model_validator(mode="after")
    def validate_node_contract(self) -> Self:
        if self.type in BOUNDARY_NODE_TYPES:
            if (
                self.task is not None
                or self.model is not None
                or self.runtime is not None
                or self.execution is not None
                or self.prompt_template is not None
            ):
                raise ValueError("boundary nodes must omit task, model, and runtime")
            return self
        if self.type == "evaluator":
            if self.task not in EVALUATOR_TASKS:
                raise ValueError(f"evaluator node task is unsupported: {self.name}")
            if (
                self.model is not None
                or self.runtime is not None
                or self.execution is not None
                or self.prompt_template is not None
            ):
                raise ValueError("evaluator nodes must omit model and runtime")
            return self
        if self.task is None:
            raise ValueError(f"agent/tool node must define task: {self.name}")
        if self.model is None:
            raise ValueError(f"agent/tool node must define model: {self.name}")
        if self.runtime is None:
            raise ValueError(f"agent/tool node must define runtime: {self.name}")
        return self


class EdgeConfig(BaseModel):
    source: NonEmptyStr
    target: NonEmptyStr
    attributes: dict[str, Any] = Field(default_factory=dict)


class Workflow(BaseModel):
    """
    尽可能减少 validate，有问题也是必须优先检查 YAML 文件，而不是让代码适配配置文件。
    """

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


class WorkflowDataItem(BaseModel):
    session_id: NonEmptyStr
    item_id: NonEmptyStr
    source_node: NonEmptyStr
    target_node: NonEmptyStr
    message: AgentState
