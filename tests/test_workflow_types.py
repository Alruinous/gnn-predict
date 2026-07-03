from __future__ import annotations

import pytest
from pydantic import ValidationError

from workflow.types import DEFAULT_QUEUE_CAPACITY, NodeConfig, Workflow


def make_execution() -> dict:
    return {
        "model_name": "qwen3-4b",
        "model_path": "/models/qwen3-4b",
        "devices": ["cuda:0"],
    }


def make_agent_node(name: str) -> dict:
    return {"name": name, "type": "agent", "execution": make_execution()}


def test_workflow_rejects_duplicate_node_names():
    with pytest.raises(ValidationError, match="unique"):
        Workflow.model_validate(
            {"nodes": [make_agent_node("a"), make_agent_node("a")], "edges": []}
        )


def test_node_config_requires_execution_for_agent_type():
    with pytest.raises(ValidationError, match="requires execution"):
        NodeConfig.model_validate({"name": "a", "type": "agent"})


def test_node_config_allows_missing_execution_for_input_type():
    node = NodeConfig.model_validate({"name": "reader", "type": "input"})
    assert node.execution is None
    assert node.queue_capacity == DEFAULT_QUEUE_CAPACITY
    assert node.retry.max_attempts == 1
    assert node.retry.on_exhausted == "fail_workflow"


def test_workflow_rejects_edge_referencing_unknown_node():
    with pytest.raises(ValidationError, match="unknown node"):
        Workflow.model_validate(
            {
                "nodes": [{"name": "reader", "type": "input"}],
                "edges": [{"source": "reader", "target": "ghost"}],
            }
        )


def test_workflow_rejects_duplicate_edges():
    with pytest.raises(ValidationError, match="unique"):
        Workflow.model_validate(
            {
                "nodes": [{"name": "reader", "type": "input"}, make_agent_node("a")],
                "edges": [
                    {"source": "reader", "target": "a"},
                    {"source": "reader", "target": "a"},
                ],
            }
        )
