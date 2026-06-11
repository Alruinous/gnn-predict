from __future__ import annotations

import pytest

from workflow.schema import Workflow
from workflow.validation import validate_workflow


def build_parallel_workflow_payload() -> dict[str, object]:
    agent_model = {
        "task": "react_agent",
        "name": "qwen3",
        "hidden_size": 1536,
        "intermediate_size": 5120,
        "num_hidden_layers": 28,
        "num_attention_heads": 24,
        "num_key_value_heads": 4,
        "vocab_size": 151936,
        "max_position_embeddings": 40960,
    }
    agent_runtime = {
        "batch_size": 1,
        "sequence_length": 512,
        "phase": "prefill",
    }
    return {
        "nodes": [
            {"name": "input", "type": "input"},
            {
                "name": "planner",
                "type": "agent",
                "model": dict(agent_model),
                "runtime": dict(agent_runtime),
            },
            {
                "name": "summarizer",
                "type": "tool",
                "model": {**agent_model, "task": "text_generation"},
                "runtime": dict(agent_runtime),
            },
            {"name": "output", "type": "output"},
        ],
        "edges": [
            {"source": "input", "target": "planner", "attributes": {}},
            {"source": "input", "target": "summarizer", "attributes": {}},
            {"source": "planner", "target": "output", "attributes": {}},
            {"source": "summarizer", "target": "output", "attributes": {}},
        ],
    }


def build_workflow(payload: dict[str, object]) -> Workflow:
    return Workflow.model_validate(payload)


def get_node_payload(payload: dict[str, object], name: str) -> dict[str, object]:
    nodes = payload["nodes"]
    assert isinstance(nodes, list)
    return next(node for node in nodes if node["name"] == name)


def test_workflow_validation_rejects_undefined_edge_reference() -> None:
    payload = build_parallel_workflow_payload()
    edges = payload["edges"]
    assert isinstance(edges, list)
    edges[0]["source"] = "missing"
    workflow = build_workflow(payload)

    with pytest.raises(ValueError, match="source is undefined"):
        validate_workflow(workflow)


def test_workflow_validation_rejects_non_empty_edge_attributes() -> None:
    payload = build_parallel_workflow_payload()
    edges = payload["edges"]
    assert isinstance(edges, list)
    edges[0]["attributes"] = {"weight": 1}
    workflow = build_workflow(payload)

    with pytest.raises(ValueError, match="attributes must be an empty map"):
        validate_workflow(workflow)


def test_workflow_validation_rejects_cycle() -> None:
    payload = build_parallel_workflow_payload()
    edges = payload["edges"]
    assert isinstance(edges, list)
    edges.append({"source": "planner", "target": "input", "attributes": {}})
    workflow = build_workflow(payload)

    with pytest.raises(ValueError, match="acyclic"):
        validate_workflow(workflow)


def test_workflow_validation_rejects_attention_head_mismatch() -> None:
    payload = build_parallel_workflow_payload()
    planner = get_node_payload(payload, "planner")
    model = planner["model"]
    assert isinstance(model, dict)
    model["hidden_size"] = 1537
    workflow = build_workflow(payload)

    with pytest.raises(ValueError, match="divisible by num_attention_heads"):
        validate_workflow(workflow)


def test_workflow_validation_rejects_kv_head_mismatch() -> None:
    payload = build_parallel_workflow_payload()
    planner = get_node_payload(payload, "planner")
    model = planner["model"]
    assert isinstance(model, dict)
    model["num_key_value_heads"] = 5
    workflow = build_workflow(payload)

    with pytest.raises(ValueError, match="divisible by num_key_value_heads"):
        validate_workflow(workflow)


def test_workflow_validation_rejects_decode_context_overflow() -> None:
    payload = build_parallel_workflow_payload()
    planner = get_node_payload(payload, "planner")
    planner["runtime"] = {
        "batch_size": 1,
        "sequence_length": 40900,
        "decode_max_output_length": 128,
        "phase": "decode",
    }
    workflow = build_workflow(payload)

    with pytest.raises(ValueError, match="exceed max_position_embeddings"):
        validate_workflow(workflow)


def test_workflow_validation_rejects_input_shape_batch_mismatch() -> None:
    payload = build_parallel_workflow_payload()
    planner = get_node_payload(payload, "planner")
    planner["runtime"] = {
        "batch_size": 4,
        "input_shape": [2, 3, 224, 224],
        "phase": "inference",
    }
    workflow = build_workflow(payload)

    with pytest.raises(ValueError, match="input_shape must match batch_size"):
        validate_workflow(workflow)
