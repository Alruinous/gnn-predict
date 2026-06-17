from __future__ import annotations

import time
from typing import Any

import yaml
from langchain_core.tools import BaseTool, StructuredTool
from pydantic import BaseModel, ConfigDict, Field

from workflow.handlers import NodeHandler
from workflow.schema import Workflow, WorkflowNodeConfig, WorkflowNodeResult
from workflow.types import WorkflowContext


class ToolNodeInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task: str
    input_path: str | None = None
    payload: dict[str, Any] = Field(default_factory=dict)


class BaseToolNode:
    def __init__(
        self,
        node: WorkflowNodeConfig,
        handler: NodeHandler,
        context: WorkflowContext | None = None,
    ) -> None:
        self.node = node
        self.handler = handler
        self.context = context or {}

    def as_tool(self) -> BaseTool:
        return StructuredTool.from_function(
            func=self.run,
            name=self.node.name,
            description=self.description(),
            args_schema=ToolNodeInput,
        )

    def description(self) -> str:
        if self.node.description:
            return self.node.description
        model_name = self.node.model.name if self.node.model else "unknown"
        model_task = self.node.task or "unspecified"
        return (
            f"Run workflow tool node '{self.node.name}' for task '{model_task}'. "
            f"The configured model name is '{model_name}'."
        )

    def run(
        self,
        task: str,
        input_path: str | None = None,
        payload: dict[str, Any] | None = None,
    ) -> str:
        node_input = self.prepare_node_input(task, input_path, payload or {})
        started_at = time.time()
        try:
            metadata = self.handler.run(self.node, node_input)
        except Exception as exc:
            ended_at = time.time()
            node_result = WorkflowNodeResult(
                node_name=self.node.name,
                status="failed",
                started_at=started_at,
                ended_at=ended_at,
                duration_sec=ended_at - started_at,
                error_type=type(exc).__name__,
                error_message=str(exc),
            )
            return self.format_node_output(node_result)

        ended_at = time.time()
        node_result = WorkflowNodeResult(
            node_name=self.node.name,
            status="succeeded",
            started_at=started_at,
            ended_at=ended_at,
            duration_sec=ended_at - started_at,
            metadata=metadata,
        )
        return self.format_node_output(node_result)

    def prepare_node_input(
        self,
        task: str,
        input_path: str | None,
        payload: dict[str, Any],
    ) -> WorkflowContext:
        return {
            **self.context,
            "task": task,
            "input_path": input_path,
            "payload": payload,
        }

    def format_node_output(self, node_result: WorkflowNodeResult) -> str:
        return yaml.safe_dump(
            node_result.model_dump(),
            sort_keys=False,
            allow_unicode=True,
        )


def build_tool_nodes(
    workflow: Workflow,
    handler: NodeHandler,
    context: WorkflowContext | None = None,
) -> list[BaseToolNode]:
    tool_nodes: list[BaseToolNode] = []
    for node in workflow.nodes:
        if node.type != "tool":
            continue
        tool_nodes.append(
            BaseToolNode(
                node=node,
                handler=handler,
                context=context,
            )
        )
    return tool_nodes
