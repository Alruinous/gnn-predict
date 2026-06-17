from __future__ import annotations

from importlib import import_module
from typing import Any

from workflow.handlers import NodeHandler
from workflow.loader import load_workflow, load_workflows
from workflow.schema import (
    Workflow,
    WorkflowEdgeConfig,
    WorkflowModelConfig,
    WorkflowNodeConfig,
    WorkflowNodeResult,
    WorkflowRuntimeConfig,
)
from workflow.tool_nodes import BaseToolNode, ToolNodeInput, build_tool_nodes
from workflow.validation import validate_workflow

_LAZY_EXPORTS = {
    "ScheduledToolNode": ("workflow.scheduled_tool_nodes", "ScheduledToolNode"),
    "ToolPrediction": ("workflow.gnn_predictor", "ToolPrediction"),
    "ToolScheduleDecision": ("workflow.scheduler", "ToolScheduleDecision"),
    "WorkflowCachedModel": ("workflow.scheduler", "WorkflowCachedModel"),
    "WorkflowDeviceState": ("workflow.scheduler", "WorkflowDeviceState"),
    "WorkflowGnnPredictor": ("workflow.gnn_predictor", "WorkflowGnnPredictor"),
    "WorkflowGnnPredictorConfig": (
        "workflow.gnn_predictor",
        "WorkflowGnnPredictorConfig",
    ),
    "WorkflowScheduler": ("workflow.scheduler", "WorkflowScheduler"),
    "WorkflowSchedulerConfig": ("workflow.scheduler", "WorkflowSchedulerConfig"),
    "WorkflowSchedulingError": ("workflow.scheduler", "WorkflowSchedulingError"),
    "build_scheduled_tool_nodes": (
        "workflow.scheduled_tool_nodes",
        "build_scheduled_tool_nodes",
    ),
    "load_scheduler_config": ("workflow.scheduler", "load_scheduler_config"),
}


def __getattr__(name: str) -> Any:
    if name not in _LAZY_EXPORTS:
        raise AttributeError(f"module 'workflow' has no attribute {name!r}")
    module_name, attribute_name = _LAZY_EXPORTS[name]
    value = getattr(import_module(module_name), attribute_name)
    globals()[name] = value
    return value


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
    "ScheduledToolNode",
    "ToolNodeInput",
    "ToolPrediction",
    "ToolScheduleDecision",
    "Workflow",
    "WorkflowCachedModel",
    "WorkflowDeviceState",
    "WorkflowEdgeConfig",
    "WorkflowGnnPredictor",
    "WorkflowGnnPredictorConfig",
    "WorkflowModelConfig",
    "WorkflowNodeConfig",
    "WorkflowNodeResult",
    "WorkflowRuntimeConfig",
    "WorkflowScheduler",
    "WorkflowSchedulerConfig",
    "WorkflowSchedulingError",
    "build_scheduled_tool_nodes",
    "build_tool_nodes",
    "build_workflow_agent",
    "build_workflow_model",
    "load_scheduler_config",
    "load_workflow",
    "load_workflows",
    "run_workflow_agent",
    "validate_workflow",
]
