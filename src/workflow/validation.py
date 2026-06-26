from __future__ import annotations

from collections import deque
from typing import Any

from gnn_archs.config import (
    DCNConfigOverride,
    DCNv2ConfigOverride,
    DeepFMConfigOverride,
    EDCNConfigOverride,
    Gpt2ConfigOverride,
    Qwen3ConfigOverride,
    T5ConfigOverride,
    is_causal_lm_model_name,
    is_detection_model_name,
    is_recommender_model_name,
    is_text_model_name,
    normalize_model_identifier,
)
from workflow.schema import Workflow, WorkflowEdgeConfig, WorkflowNodeConfig

BOUNDARY_NODE_TYPES = {"input", "output"}
MODEL_NODE_TYPES = {"agent", "tool"}
EVALUATOR_TASKS = {
    "gsm8k_numeric_exact_match",
    "mbpp_pass_at_1",
    "summary_rouge",
    "summary_llm_judge",
}
EDGE_CONDITIONS = {"passed", "failed", "default"}
FORBIDDEN_MODEL_PARAMETER_FIELDS = {"mutations", "pretrained", "task"}
RECOMMENDER_CONFIG_BY_NAME = {
    "deepfm": ("deepfm_config", DeepFMConfigOverride),
    "dcn": ("dcn_config", DCNConfigOverride),
    "dcnv2": ("dcnv2_config", DCNv2ConfigOverride),
    "edcn": ("edcn_config", EDCNConfigOverride),
}


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
            if (
                node.task is not None
                or node.model is not None
                or node.runtime is not None
                or node.execution is not None
                or node.prompt_template is not None
            ):
                message = "boundary nodes must omit task, model, runtime, and execution"
                raise ValueError(f"{message}: {node.name}")
            continue
        if node.type == "evaluator":
            validate_evaluator_node(node)
            continue
        if node.task is None:
            raise ValueError(f"agent/tool node must define task: {node.name}")
        if node.model is None:
            raise ValueError(f"agent/tool node must define model: {node.name}")
        if node.runtime is None:
            raise ValueError(f"agent/tool node must define runtime: {node.name}")
        validate_runtime(node)
        validate_model(node)


def validate_evaluator_node(node: WorkflowNodeConfig) -> None:
    if node.task not in EVALUATOR_TASKS:
        raise ValueError(f"evaluator node task is unsupported: {node.name}")
    if node.model is not None or node.runtime is not None or node.execution is not None:
        raise ValueError(
            f"evaluator nodes must omit model, runtime, and execution: {node.name}"
        )
    if node.prompt_template is not None:
        raise ValueError(f"evaluator nodes must omit prompt_template: {node.name}")


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
            parameters = node.model.parameters
            max_positions = int(
                parameters.get("max_position_embeddings")
                or parameters.get("n_positions")
                or 0
            )
        if (
            max_positions
            and sequence_length + node.runtime.decode_max_output_length > max_positions
        ):
            message = "decode length and sequence length exceed max_position_embeddings"
            raise ValueError(f"{message}: {node.name}")


def validate_model(node: WorkflowNodeConfig) -> None:
    if not node.model:
        return
    parameters = node.model.parameters
    forbidden_fields = sorted(set(parameters) & FORBIDDEN_MODEL_PARAMETER_FIELDS)
    if forbidden_fields:
        raise ValueError(
            f"model parameters contain workflow-only fields: {forbidden_fields}"
        )
    if node.type != "tool":
        return

    model_name = node.model.name
    normalized_name = normalize_model_identifier(model_name)
    if normalized_name == "gpt2":
        validate_attention_parameters(parameters, node.name)
        validate_gpt2_parameters(parameters, node.name)
        return
    if normalized_name == "t5":
        validate_attention_parameters(parameters, node.name)
        validate_t5_parameters(parameters, node.name)
        return
    if normalized_name.startswith("qwen"):
        validate_attention_parameters(parameters, node.name)
        Qwen3ConfigOverride.model_validate(parameters)
        return
    if is_recommender_model_name(model_name):
        validate_recommender_parameters(normalized_name, parameters, node.name)
        return
    if is_detection_model_name(model_name):
        validate_positive_int_parameter(parameters, "input_channels", node.name)
        validate_positive_int_parameter(parameters, "output_classes", node.name)
        return
    if is_causal_lm_model_name(model_name):
        return
    if is_text_model_name(model_name):
        validate_positive_int_parameter(parameters, "output_classes", node.name)
        return

    validate_positive_int_parameter(parameters, "input_channels", node.name)
    validate_positive_int_parameter(parameters, "output_classes", node.name)


def validate_gpt2_parameters(parameters: dict[str, Any], node_name: str) -> None:
    validate_positive_int_parameter(parameters, "output_classes", node_name)
    model_parameters = {
        key: value for key, value in parameters.items() if key != "output_classes"
    }
    Gpt2ConfigOverride.model_validate(model_parameters)


def validate_t5_parameters(parameters: dict[str, Any], node_name: str) -> None:
    validate_positive_int_parameter(parameters, "output_classes", node_name)
    model_parameters = {
        key: value for key, value in parameters.items() if key != "output_classes"
    }
    T5ConfigOverride.model_validate(model_parameters)


def validate_recommender_parameters(
    normalized_name: str,
    parameters: dict[str, Any],
    node_name: str,
) -> None:
    config_field, config_model = RECOMMENDER_CONFIG_BY_NAME[normalized_name]
    config_payload = parameters.get(config_field)
    if config_payload is None:
        raise ValueError(f"{config_field} is required: {node_name}")
    config_model.model_validate(config_payload)


def validate_attention_parameters(
    parameters: dict[str, Any],
    node_name: str,
) -> None:
    hidden_size = (
        parameters.get("hidden_size")
        or parameters.get("n_embd")
        or parameters.get("d_model")
    )
    num_attention_heads = (
        parameters.get("num_attention_heads")
        or parameters.get("n_head")
        or parameters.get("num_heads")
    )
    num_key_value_heads = parameters.get("num_key_value_heads")
    if (
        isinstance(hidden_size, int)
        and isinstance(num_attention_heads, int)
        and hidden_size % num_attention_heads != 0
    ):
        raise ValueError(
            f"hidden_size must be divisible by attention heads: {node_name}"
        )
    if (
        isinstance(num_attention_heads, int)
        and isinstance(num_key_value_heads, int)
        and num_attention_heads % num_key_value_heads != 0
    ):
        raise ValueError(
            f"num_attention_heads must be divisible by num_key_value_heads: {node_name}"
        )


def validate_positive_int_parameter(
    parameters: dict[str, Any],
    field_name: str,
    node_name: str,
) -> int:
    value = parameters.get(field_name)
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{field_name} must be a positive integer: {node_name}")
    return value


def validate_edges(
    workflow: Workflow,
    node_map: dict[str, WorkflowNodeConfig],
) -> None:
    for edge in workflow.edges:
        if edge.source not in node_map:
            raise ValueError(f"edge source is undefined: {edge.source}")
        if edge.target not in node_map:
            raise ValueError(f"edge target is undefined: {edge.target}")
        validate_edge_attributes(edge, node_map[edge.source])


def validate_edge_attributes(
    edge: WorkflowEdgeConfig,
    source_node: WorkflowNodeConfig,
) -> None:
    if not edge.attributes:
        return
    if set(edge.attributes) != {"condition"}:
        raise ValueError("edge attributes only support condition")
    condition = edge.attributes["condition"]
    if condition not in EDGE_CONDITIONS:
        raise ValueError(f"edge condition is unsupported: {condition}")
    if source_node.type != "evaluator":
        raise ValueError("edge condition is only allowed from evaluator nodes")


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
