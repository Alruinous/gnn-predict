from __future__ import annotations

from collections import deque

from workflow.schema import Workflow, WorkflowNodeConfig

BOUNDARY_NODE_TYPES = {"input", "output"}


def validate_workflow(workflow: Workflow) -> Workflow:
    node_map = workflow.node_map()
    validate_node_names(workflow)
    validate_node_configs(workflow)
    validate_edges(workflow, node_map)
    validate_acyclic(workflow, node_map)
    return workflow


def validate_node_names(workflow: Workflow) -> None:
    names = workflow.node_names()
    if len(names) != len(set(names)):
        raise ValueError("workflow node names must be unique")


def validate_node_configs(workflow: Workflow) -> None:
    for node in workflow.nodes:
        if node.type in BOUNDARY_NODE_TYPES:
            if node.model is not None or node.runtime is not None:
                raise ValueError(
                    f"boundary nodes must omit model and runtime: {node.name}"
                )
            continue
        if node.model is None:
            raise ValueError(f"agent/tool node must define model: {node.name}")
        if node.runtime is None:
            raise ValueError(f"agent/tool node must define runtime: {node.name}")
        validate_runtime(node)
        validate_model(node)


def validate_runtime(node: WorkflowNodeConfig) -> None:
    assert node.runtime is not None
    if (
        node.runtime.input_shape is not None
        and node.runtime.input_shape
        and node.runtime.input_shape[0] != node.runtime.batch_size
    ):
        raise ValueError(f"input_shape must match batch_size: {node.name}")
    if node.runtime.decode_max_output_length is not None:
        sequence_length = node.runtime.sequence_length or 0
        max_positions = 0
        if node.model:
            max_positions = int(node.model.get("max_position_embeddings", 0))
        if (
            max_positions
            and sequence_length + node.runtime.decode_max_output_length > max_positions
        ):
            message = "decode length and sequence length exceed max_position_embeddings"
            raise ValueError(f"{message}: {node.name}")


def validate_model(node: WorkflowNodeConfig) -> None:
    if not node.model:
        return
    hidden_size = node.model.get("hidden_size")
    num_attention_heads = node.model.get("num_attention_heads")
    num_key_value_heads = node.model.get("num_key_value_heads")
    if (
        hidden_size is not None
        and num_attention_heads is not None
        and int(hidden_size) % int(num_attention_heads) != 0
    ):
        message = "hidden_size must be divisible by num_attention_heads"
        raise ValueError(f"{message}: {node.name}")
    if (
        hidden_size is not None
        and num_key_value_heads is not None
        and int(hidden_size) % int(num_key_value_heads) != 0
    ):
        raise ValueError(
            f"hidden_size must be divisible by num_key_value_heads: {node.name}"
        )


def validate_edges(
    workflow: Workflow,
    node_map: dict[str, WorkflowNodeConfig],
) -> None:
    for edge in workflow.edges:
        if edge.source not in node_map:
            raise ValueError(f"edge source is undefined: {edge.source}")
        if edge.target not in node_map:
            raise ValueError(f"edge target is undefined: {edge.target}")
        if edge.attributes:
            raise ValueError("edge attributes must be an empty map")


def validate_acyclic(
    workflow: Workflow,
    node_map: dict[str, WorkflowNodeConfig],
) -> None:
    indegree = {name: 0 for name in node_map}
    successors = {name: [] for name in node_map}
    for edge in workflow.edges:
        indegree[edge.target] += 1
        successors[edge.source].append(edge.target)

    ready = deque(name for name in workflow.node_names() if indegree[name] == 0)
    visited = 0
    while ready:
        name = ready.popleft()
        visited += 1
        for target in successors[name]:
            indegree[target] -= 1
            if indegree[target] == 0:
                ready.append(target)

    if visited != len(workflow.nodes):
        raise ValueError("workflow must be acyclic")
