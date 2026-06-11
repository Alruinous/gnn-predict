from __future__ import annotations

from typing import Any

from workflow.handlers import NodeHandler
from workflow.loader import load_workflow, load_workflows
from workflow.schema import (
    Workflow,
    WorkflowEdgeConfig,
    WorkflowNodeConfig,
    WorkflowNodeResult,
    WorkflowRuntimeConfig,
)
from workflow.tool_nodes import BaseToolNode, ToolNodeInput, build_tool_nodes
from workflow.validation import validate_workflow


def build_workflow_agent(*args: Any, **kwargs: Any) -> Any:
    from workflow.agent import build_workflow_agent as agent_builder

    return agent_builder(*args, **kwargs)


def build_workflow_model() -> Any:
    from workflow.agent import build_workflow_model as model_builder

    return model_builder()


def run_workflow_agent(*args: Any, **kwargs: Any) -> str:
    from workflow.agent import run_workflow_agent as agent_runner

    return agent_runner(*args, **kwargs)


__all__ = [
    "BaseToolNode",
    "NodeHandler",
    "ToolNodeInput",
    "Workflow",
    "WorkflowEdgeConfig",
    "WorkflowNodeConfig",
    "WorkflowNodeResult",
    "WorkflowRuntimeConfig",
    "build_tool_nodes",
    "build_workflow_agent",
    "build_workflow_model",
    "load_workflow",
    "load_workflows",
    "run_workflow_agent",
    "validate_workflow",
]
