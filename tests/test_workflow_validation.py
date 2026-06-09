from __future__ import annotations

import pytest

from workflow.execution import run_experiment, run_workflow
from workflow.schema import Workflow
from workflow.validation import validate_workflow


def build_parallel_workflow_payload() -> dict[str, object]:
    agent_model = {
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
        "nodes": {
            "input": {"type": "input"},
            "planner": {
                "type": "main_agent",
                "model": dict(agent_model),
                "runtime": dict(agent_runtime),
            },
            "summarizer": {
                "type": "main_agent",
                "model": dict(agent_model),
                "runtime": dict(agent_runtime),
            },
            "output": {"type": "output"},
        },
        "edges": [
            {"source": "input", "target": "planner", "attributes": {}},
            {"source": "input", "target": "summarizer", "attributes": {}},
            {"source": "planner", "target": "output", "attributes": {}},
            {"source": "summarizer", "target": "output", "attributes": {}},
        ],
    }


def build_workflow(payload: dict[str, object]) -> Workflow:
    return Workflow.model_validate(payload)


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
    nodes = payload["nodes"]
    assert isinstance(nodes, dict)
    nodes["planner"]["model"]["hidden_size"] = 1537
    workflow = build_workflow(payload)

    with pytest.raises(ValueError, match="divisible by num_attention_heads"):
        validate_workflow(workflow)


def test_workflow_validation_rejects_kv_head_mismatch() -> None:
    payload = build_parallel_workflow_payload()
    nodes = payload["nodes"]
    assert isinstance(nodes, dict)
    nodes["planner"]["model"]["num_key_value_heads"] = 5
    workflow = build_workflow(payload)

    with pytest.raises(ValueError, match="divisible by num_key_value_heads"):
        validate_workflow(workflow)


def test_workflow_validation_rejects_decode_context_overflow() -> None:
    payload = build_parallel_workflow_payload()
    nodes = payload["nodes"]
    assert isinstance(nodes, dict)
    nodes["planner"]["runtime"] = {
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
    nodes = payload["nodes"]
    assert isinstance(nodes, dict)
    nodes["planner"]["runtime"] = {
        "batch_size": 4,
        "input_shape": [2, 3, 224, 224],
        "phase": "inference",
    }
    workflow = build_workflow(payload)

    with pytest.raises(ValueError, match="input_shape must match batch_size"):
        validate_workflow(workflow)


def test_workflow_validation_runs_serial_and_parallel_order() -> None:
    workflow = build_workflow(build_parallel_workflow_payload())

    serial_plan = run_workflow(workflow, mode="serial")
    parallel_plan = run_workflow(workflow, mode="parallel")
    result = run_experiment(workflow)

    assert serial_plan.levels == [
        ["input"],
        ["planner"],
        ["summarizer"],
        ["output"],
    ]
    assert parallel_plan.levels == [
        ["input"],
        ["planner", "summarizer"],
        ["output"],
    ]
    assert result.serial == serial_plan
    assert result.parallel == parallel_plan


def test_workflow_validation_rejects_adaptive_mode() -> None:
    workflow = build_workflow(build_parallel_workflow_payload())

    with pytest.raises(ValueError, match="unsupported workflow run mode"):
        run_workflow(workflow, mode="adaptive")
