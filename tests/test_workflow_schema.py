from __future__ import annotations

import pytest
from pydantic import ValidationError

from workflow.schema import Workflow
from workflow.validation import validate_workflow


def build_valid_workflow_payload() -> dict[str, object]:
    return {
        "nodes": {
            "input": {"type": "input"},
            "planner": {
                "type": "main_agent",
                "model": {
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
            "output": {"type": "output"},
        },
        "edges": [
            {"source": "input", "target": "planner", "attributes": {}},
            {"source": "planner", "target": "output", "attributes": {}},
        ],
    }


def test_workflow_schema_loads_valid_minimal_workflow() -> None:
    workflow = Workflow.model_validate(build_valid_workflow_payload())

    validated = validate_workflow(workflow)

    assert set(validated.nodes) == {"input", "planner", "output"}
    assert validated.nodes["input"].model is None
    assert validated.nodes["output"].runtime is None
    assert validated.nodes["planner"].runtime is not None
    assert validated.nodes["planner"].runtime.sequence_length == 512


def test_workflow_schema_rejects_invalid_node_type() -> None:
    payload = build_valid_workflow_payload()
    nodes = payload["nodes"]
    assert isinstance(nodes, dict)
    nodes["planner"]["type"] = "unknown"

    with pytest.raises(ValidationError, match="unknown"):
        Workflow.model_validate(payload)


def test_workflow_schema_rejects_extra_top_level_fields() -> None:
    payload = build_valid_workflow_payload()
    payload["training"] = {"enabled": True}

    with pytest.raises(ValidationError, match="training"):
        Workflow.model_validate(payload)


def test_workflow_schema_rejects_model_node_without_model() -> None:
    workflow = Workflow.model_validate(build_valid_workflow_payload())
    planner = workflow.nodes["planner"].model_copy(update={"model": None})
    workflow = workflow.model_copy(update={"nodes": {**workflow.nodes, "planner": planner}})

    with pytest.raises(ValueError, match="must define model"):
        validate_workflow(workflow)


def test_workflow_schema_rejects_model_node_without_runtime() -> None:
    workflow = Workflow.model_validate(build_valid_workflow_payload())
    planner = workflow.nodes["planner"].model_copy(update={"runtime": None})
    workflow = workflow.model_copy(update={"nodes": {**workflow.nodes, "planner": planner}})

    with pytest.raises(ValueError, match="must define runtime"):
        validate_workflow(workflow)


def test_workflow_schema_rejects_boundary_node_model_config() -> None:
    workflow = Workflow.model_validate(build_valid_workflow_payload())
    input_node = workflow.nodes["input"].model_copy(update={"model": {"name": "x"}})
    workflow = workflow.model_copy(update={"nodes": {**workflow.nodes, "input": input_node}})

    with pytest.raises(ValueError, match="boundary nodes must omit"):
        validate_workflow(workflow)
