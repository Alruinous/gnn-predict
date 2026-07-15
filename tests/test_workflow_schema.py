from __future__ import annotations

import copy
import pickle
from typing import Any, cast

import pytest
from langchain.agents import AgentState
from pydantic import ValidationError

from workflow.schema import (
    AgentNodeConfig,
    ExecutionConfig,
    FunctionNodeConfig,
    Workflow,
)
from workflow.types import (
    ModelReplicaState,
    NodeTaskState,
    NodeWorkerState,
    SessionState,
    WorkflowDataItem,
    WorkflowModelFeatureKey,
)

WORKFLOW_PAYLOAD: dict[str, Any] = {
    "nodes": [
        {
            "name": "split",
            "type": "function",
            "function": "split_input",
            "parameters": {"parts": 2},
            "routing": "targeted",
        },
        {
            "name": "left",
            "type": "agent",
            "model": {"name": "test-model", "parameters": {}},
            "execution": {
                "model_path": "/models/test-model",
                "max_new_tokens": 16,
                "serving": {
                    "max_model_len": 1024,
                    "max_num_seqs": 3,
                    "max_num_batched_tokens": 1536,
                },
            },
            "prompt_template": "Summarize {content}",
        },
        {
            "name": "right",
            "type": "agent",
            "model": {"name": "test-model", "parameters": {}},
            "execution": {
                "model_path": "/models/test-model",
                "max_new_tokens": 16,
                "serving": {
                    "max_model_len": 1024,
                    "max_num_seqs": 3,
                    "max_num_batched_tokens": 1536,
                },
            },
            "prompt_template": "Summarize {content}",
        },
        {
            "name": "merge",
            "type": "function",
            "function": "merge_outputs",
        },
    ],
    "edges": [
        {"source": "split", "target": "left"},
        {"source": "split", "target": "right"},
        {"source": "left", "target": "merge"},
        {"source": "right", "target": "merge"},
    ],
}


def test_workflow_computes_one_static_graph() -> None:
    workflow = Workflow.model_validate(WORKFLOW_PAYLOAD)

    assert workflow.graph.entry_node == "split"
    assert workflow.graph.terminal_node == "merge"
    assert workflow.graph.topological_order == ("split", "left", "right", "merge")
    assert workflow.graph.adjacency == {
        "split": ("left", "right"),
        "left": ("merge",),
        "right": ("merge",),
        "merge": (),
    }
    assert workflow.graph.dependencies == {
        "split": (),
        "left": ("split",),
        "right": ("split",),
        "merge": ("left", "right"),
    }
    assert isinstance(workflow.nodes[0], FunctionNodeConfig)
    assert isinstance(workflow.nodes[1], AgentNodeConfig)


def test_execution_requires_explicit_serving_capacity() -> None:
    with pytest.raises(ValidationError, match="serving"):
        ExecutionConfig.model_validate({"model_path": "/models/test-model"})


def test_execution_rejects_incompatible_vllm_capacity_and_dtype() -> None:
    with pytest.raises(ValidationError, match="cover max_model_len"):
        ExecutionConfig.model_validate(
            {
                "model_path": "/models/test-model",
                "max_new_tokens": 16,
                "serving": {
                    "max_model_len": 2048,
                    "max_num_seqs": 3,
                    "max_num_batched_tokens": 1024,
                },
            }
        )
    with pytest.raises(ValidationError, match="float16"):
        ExecutionConfig.model_validate(
            {
                "model_path": "/models/test-model",
                "max_new_tokens": 16,
                "dtype": "bfloat16",
                "serving": {
                    "max_model_len": 1024,
                    "max_num_seqs": 3,
                    "max_num_batched_tokens": 1024,
                },
            }
        )


def test_workflow_graph_is_immutable() -> None:
    graph = Workflow.model_validate(WORKFLOW_PAYLOAD).graph

    with pytest.raises(TypeError):
        cast(Any, graph.adjacency)["split"] = ()
    with pytest.raises(ValidationError):
        graph.entry_node = "left"


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ({"nodes": [], "edges": []}, "non-empty"),
        (
            {
                "nodes": WORKFLOW_PAYLOAD["nodes"],
                "edges": [{"source": "missing", "target": "merge"}],
            },
            "unknown node",
        ),
        (
            {
                "nodes": WORKFLOW_PAYLOAD["nodes"],
                "edges": [{"source": "split", "target": "split"}],
            },
            "self-edge",
        ),
        (
            {
                "nodes": WORKFLOW_PAYLOAD["nodes"],
                "edges": [
                    {"source": "split", "target": "left"},
                    {"source": "split", "target": "left"},
                ],
            },
            "duplicate edge",
        ),
        (
            {
                "nodes": WORKFLOW_PAYLOAD["nodes"][:2],
                "edges": [
                    {"source": "split", "target": "left"},
                    {"source": "left", "target": "split"},
                ],
            },
            "acyclic",
        ),
        (
            {
                "nodes": WORKFLOW_PAYLOAD["nodes"][:3],
                "edges": [
                    {"source": "split", "target": "left"},
                    {"source": "split", "target": "right"},
                ],
            },
            "exactly one terminal",
        ),
        (
            {
                "nodes": WORKFLOW_PAYLOAD["nodes"][1:],
                "edges": [
                    {"source": "left", "target": "merge"},
                    {"source": "right", "target": "merge"},
                ],
            },
            "exactly one entry",
        ),
    ],
)
def test_workflow_rejects_invalid_graph(
    payload: dict[str, object], message: str
) -> None:
    with pytest.raises(ValidationError, match=message):
        Workflow.model_validate(payload)


def test_workflow_rejects_cycle() -> None:
    cyclic_payload = copy.deepcopy(WORKFLOW_PAYLOAD)
    cyclic_payload["edges"].append({"source": "merge", "target": "split"})

    with pytest.raises(ValidationError, match="acyclic"):
        Workflow.model_validate(cyclic_payload)


def test_workflow_rejects_duplicate_node_names() -> None:
    payload = copy.deepcopy(WORKFLOW_PAYLOAD)
    payload["nodes"][1]["name"] = "split"

    with pytest.raises(ValidationError, match="node names must be unique"):
        Workflow.model_validate(payload)


def test_workflow_rejects_unsupported_and_mixed_node_kinds() -> None:
    unsupported = copy.deepcopy(WORKFLOW_PAYLOAD)
    unsupported["nodes"][0]["type"] = "tool"
    mixed = copy.deepcopy(WORKFLOW_PAYLOAD)
    mixed["nodes"][0]["model"] = {"name": "invalid", "parameters": {}}

    with pytest.raises(ValidationError):
        Workflow.model_validate(unsupported)
    with pytest.raises(ValidationError):
        Workflow.model_validate(mixed)


def test_workflow_rejects_targeted_terminal_function() -> None:
    payload = copy.deepcopy(WORKFLOW_PAYLOAD)
    payload["nodes"][-1]["routing"] = "targeted"

    with pytest.raises(ValidationError, match=r"terminal function.*broadcast"):
        Workflow.model_validate(payload)


def test_execution_requires_positive_max_new_tokens() -> None:
    payload = copy.deepcopy(WORKFLOW_PAYLOAD)
    payload["nodes"][1]["execution"]["max_new_tokens"] = 0

    with pytest.raises(ValidationError, match="max_new_tokens"):
        Workflow.model_validate(payload)


def test_workflow_data_item_generates_identity_and_preserves_input_ref() -> None:
    session_input_ref = object()
    message = AgentState(messages=[])

    item = WorkflowDataItem(
        session_id="session-1",
        source_node=None,
        target_node="split",
        message=message,
        session_input_ref=session_input_ref,
    )

    assert item.item_id
    assert item.message == message
    assert item.session_input_ref is session_input_ref


def test_runtime_state_enums_match_the_managed_lifecycle() -> None:
    assert [state.value for state in SessionState] == [
        "active",
        "completed",
        "failed",
    ]
    assert [state.value for state in NodeTaskState] == [
        "pending",
        "acquiring",
        "running",
        "emitting",
        "completed",
        "failed",
        "cancelled",
    ]
    assert [state.value for state in NodeWorkerState] == [
        "idle",
        "running",
        "stopped",
    ]
    assert [state.value for state in ModelReplicaState] == [
        "loading",
        "idle",
        "busy",
        "evicting",
        "suspect",
    ]


def test_workflow_model_feature_key_remains_cache_compatible() -> None:
    payload = {
        "model_name": "Qwen3-4B",
        "phase": "decode",
        "gpu_name": "a100",
        "batch_size": 1,
        "sequence_length": 8192,
        "decode_output_length": 384,
    }
    key = WorkflowModelFeatureKey.model_validate(payload)

    assert key.model_dump(mode="python") == payload
    assert key.stable_digest == "ed8b56de663b781b"
    assert pickle.loads(pickle.dumps(key)) == key
