from __future__ import annotations

import pytest
from pydantic import ValidationError

from workflow.schema import Workflow, WorkflowNodeConfig
from workflow.validation import validate_workflow


def build_valid_workflow_payload() -> dict[str, object]:
    return {
        "nodes": [
            {"name": "input", "type": "input"},
            {
                "name": "planner",
                "type": "agent",
                "description": "Plan which tool nodes should handle the user request.",
                "model": {
                    "task": "react_agent",
                    "name": "qwen3",
                    "hidden_size": 1536,
                    "intermediate_size": 5120,
                    "num_hidden_layers": 28,
                    "num_attention_heads": 24,
                    "num_key_value_heads": 4,
                    "vocab_size": 151936,
                    "max_position_embeddings": 40960,
                },
                "runtime": {
                    "batch_size": 1,
                    "sequence_length": 512,
                    "phase": "prefill",
                },
            },
            {"name": "output", "type": "output"},
        ],
        "edges": [
            {"source": "input", "target": "planner", "attributes": {}},
            {"source": "planner", "target": "output", "attributes": {}},
        ],
    }


def node_by_name(workflow: Workflow, name: str) -> WorkflowNodeConfig:
    return workflow.node_map()[name]


def replace_node(workflow: Workflow, node: WorkflowNodeConfig) -> Workflow:
    nodes = [node if item.name == node.name else item for item in workflow.nodes]
    return workflow.model_copy(update={"nodes": nodes})


def test_workflow_schema_loads_valid_minimal_workflow() -> None:
    workflow = Workflow.model_validate(build_valid_workflow_payload())

    validated = validate_workflow(workflow)

    assert validated.node_names() == ["input", "planner", "output"]
    assert node_by_name(validated, "input").model is None
    assert node_by_name(validated, "output").runtime is None
    planner = node_by_name(validated, "planner")
    assert planner.description == "Plan which tool nodes should handle the user request."
    assert planner.runtime is not None
    assert planner.runtime.sequence_length == 512


def test_workflow_schema_strips_node_description() -> None:
    payload = build_valid_workflow_payload()
    nodes = payload["nodes"]
    assert isinstance(nodes, list)
    nodes[1]["description"] = "  Route image and text work to tools.  "

    workflow = Workflow.model_validate(payload)

    assert node_by_name(workflow, "planner").description == (
        "Route image and text work to tools."
    )


def test_workflow_schema_rejects_empty_node_description() -> None:
    payload = build_valid_workflow_payload()
    nodes = payload["nodes"]
    assert isinstance(nodes, list)
    nodes[1]["description"] = "  "

    with pytest.raises(ValidationError, match="description"):
        Workflow.model_validate(payload)


def test_workflow_schema_rejects_invalid_node_type() -> None:
    payload = build_valid_workflow_payload()
    nodes = payload["nodes"]
    assert isinstance(nodes, list)
    nodes[1]["type"] = "unknown"

    with pytest.raises(ValidationError, match="unknown"):
        Workflow.model_validate(payload)


def test_workflow_schema_rejects_extra_top_level_fields() -> None:
    payload = build_valid_workflow_payload()
    payload["training"] = {"enabled": True}

    with pytest.raises(ValidationError, match="training"):
        Workflow.model_validate(payload)


def test_workflow_schema_rejects_duplicate_node_name() -> None:
    payload = build_valid_workflow_payload()
    nodes = payload["nodes"]
    assert isinstance(nodes, list)
    nodes.append({"name": "planner", "type": "input"})
    workflow = Workflow.model_validate(payload)

    with pytest.raises(ValueError, match="node names must be unique"):
        validate_workflow(workflow)


def test_workflow_schema_rejects_executable_node_without_model() -> None:
    workflow = Workflow.model_validate(build_valid_workflow_payload())
    planner = node_by_name(workflow, "planner").model_copy(update={"model": None})
    workflow = replace_node(workflow, planner)

    with pytest.raises(ValueError, match="must define model"):
        validate_workflow(workflow)


def test_workflow_schema_rejects_executable_node_without_runtime() -> None:
    workflow = Workflow.model_validate(build_valid_workflow_payload())
    planner = node_by_name(workflow, "planner").model_copy(update={"runtime": None})
    workflow = replace_node(workflow, planner)

    with pytest.raises(ValueError, match="must define runtime"):
        validate_workflow(workflow)


def test_workflow_schema_rejects_boundary_node_model_config() -> None:
    workflow = Workflow.model_validate(build_valid_workflow_payload())
    input_node = node_by_name(workflow, "input").model_copy(
        update={"model": {"name": "x"}}
    )
    workflow = replace_node(workflow, input_node)

    with pytest.raises(ValueError, match="boundary nodes must omit"):
        validate_workflow(workflow)
