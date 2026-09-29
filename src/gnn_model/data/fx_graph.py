from __future__ import annotations

import math
import operator
from collections import deque
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import torch
from torch.export import ExportedProgram
from torch.export.graph_signature import InputKind
from torch.fx import Node
from torch_geometric.data import Data

from common.graph_artifact import load_graph_artifact, validate_zero_storage_state
from gnn_model.data.constants import (
    EDGE_FEATURE_DIM,
    GPU_SPECS,
    GRAPH_FEATURE_DIM,
    MAX_SHAPE_RANK,
    NODE_FEATURE_DIM,
    OP_TYPE_TO_INDEX,
    PHASE_TO_INDEX,
    normalize_gpu_name,
)

CONV_OPS = {
    "conv1d",
    "conv2d",
    "conv3d",
    "convolution",
    "conv_transpose1d",
    "conv_transpose2d",
    "conv_transpose3d",
}
DENSE_OPS = {"linear", "mm", "matmul", "bmm", "addmm"}
EMBEDDING_OPS = {
    "embedding",
    "gather",
    "index",
    "index_select",
    "take",
    "take_along_dim",
}
ATTENTION_OPS = {
    "scaled_dot_product_attention",
    "_scaled_dot_product_attention_math",
    "_scaled_dot_product_flash_attention",
    "softmax",
    "_softmax",
    "log_softmax",
    "_log_softmax",
}
NORM_OPS = {
    "batch_norm",
    "_native_batch_norm_legit_no_training",
    "native_batch_norm",
    "instance_norm",
    "group_norm",
    "native_group_norm",
    "layer_norm",
    "native_layer_norm",
    "rms_norm",
}
POOL_OPS = {
    "adaptive_avg_pool1d",
    "adaptive_avg_pool2d",
    "adaptive_avg_pool3d",
    "adaptive_max_pool1d",
    "adaptive_max_pool2d",
    "adaptive_max_pool3d",
    "avg_pool1d",
    "avg_pool2d",
    "avg_pool3d",
    "max_pool1d",
    "max_pool2d",
    "max_pool3d",
    "max_pool2d_with_indices",
}
ACTIVATION_OPS = {
    "clamp",
    "clamp_max",
    "clamp_min",
    "elu",
    "erf",
    "gelu",
    "hardsigmoid",
    "hardswish",
    "leaky_relu",
    "mish",
    "relu",
    "relu_",
    "selu",
    "sigmoid",
    "silu",
    "softplus",
    "tanh",
}
ELEMENTWISE_OPS = {
    "__and__",
    "abs",
    "add",
    "bitwise_and",
    "bitwise_not",
    "bitwise_or",
    "bitwise_xor",
    "copy",
    "cos",
    "div",
    "eq",
    "exp",
    "fill",
    "fmod",
    "ge",
    "gt",
    "isinf",
    "isnan",
    "le",
    "logical_and",
    "logical_not",
    "logical_or",
    "logical_xor",
    "log",
    "lt",
    "masked_fill",
    "maximum",
    "minimum",
    "mul",
    "ne",
    "neg",
    "pow",
    "reciprocal",
    "remainder",
    "round",
    "rsqrt",
    "sign",
    "sin",
    "sqrt",
    "sub",
    "tril",
    "triu",
    "where",
}
REDUCE_OPS = {
    "all",
    "amax",
    "amin",
    "any",
    "argmax",
    "argmin",
    "cumsum",
    "max",
    "mean",
    "min",
    "prod",
    "sum",
}
SHAPE_OPS = {
    "arange",
    "empty",
    "empty_like",
    "full",
    "full_like",
    "new_empty",
    "new_full",
    "new_ones",
    "new_zeros",
    "ones",
    "ones_like",
    "scalar_tensor",
    "zeros",
    "zeros_like",
}
LAYOUT_OPS = {
    "_unsafe_view",
    "contiguous",
    "flatten",
    "pad",
    "permute",
    "reshape",
    "resize",
    "roll",
    "squeeze",
    "t",
    "transpose",
    "unsqueeze",
    "upsample_bicubic2d",
    "upsample_bilinear2d",
    "upsample_nearest2d",
    "view",
}
JOIN_SPLIT_OPS = {
    "cat",
    "chunk",
    "expand",
    "index_put",
    "narrow",
    "repeat",
    "repeat_interleave",
    "scatter",
    "scatter_add",
    "select",
    "select_scatter",
    "slice",
    "slice_scatter",
    "split",
    "split_with_sizes",
    "stack",
    "tile",
    "unbind",
}
CAST_OPS = {"_to_copy", "to", "type_as"}
CONSTANT_OPS = {"lift_fresh_copy"}
IDENTITY_OPS = {"alias", "clone", "detach"}


@dataclass(frozen=True)
class TensorSpec:
    shape: tuple[int, ...]
    itemsize: int

    @property
    def elements(self) -> int:
        return count_elements(self.shape)

    @property
    def bytes(self) -> int:
        return self.elements * self.itemsize


@dataclass(frozen=True)
class FxNodeInfo:
    node: Node
    op_name: str
    category: str
    input_nodes: tuple[Node, ...]
    input_specs: tuple[TensorSpec, ...]
    output_specs: tuple[TensorSpec, ...]
    attr_count: int
    macs: int

    @property
    def memory_bytes(self) -> int:
        return sum(spec.bytes for spec in self.output_specs)


@dataclass(frozen=True)
class FxEdge:
    source_index: int
    target_index: int
    tensor: TensorSpec


def build_graph_data_from_fx(
    graph_path: str | Path,
    *,
    batch_size: int = 1,
    gpu_name: str = "v100",
    phase: str = "training",
    decode_output_length: int = 0,
) -> Data:
    exported_program, metadata = load_graph_artifact(graph_path)
    return build_graph_data_from_exported_program(
        exported_program,
        runtime_input_names=metadata.runtime_input_names,
        batch_size=batch_size,
        gpu_name=gpu_name,
        phase=phase,
        decode_output_length=decode_output_length,
        graph_path=graph_path,
    )


def build_graph_data_from_exported_program(
    exported_program: ExportedProgram,
    *,
    runtime_input_names: list[str],
    batch_size: int = 1,
    gpu_name: str = "v100",
    phase: str = "training",
    decode_output_length: int = 0,
    graph_path: str | Path | None = None,
) -> Data:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if decode_output_length < 0:
        raise ValueError("decode_output_length must be non-negative")
    validate_zero_storage_state(exported_program)

    node_infos = build_node_infos(exported_program)
    if not node_infos:
        raise ValueError("exported graph has no tensor operator nodes")
    node_to_index = {info.node: index for index, info in enumerate(node_infos)}
    edges = build_edges(node_infos, node_to_index)
    in_degree, out_degree = build_node_degrees(len(node_infos), edges)
    node_features = [
        build_node_feature_vector(
            info,
            in_degree=in_degree[index],
            out_degree=out_degree[index],
        )
        for index, info in enumerate(node_infos)
    ]
    edge_features = [
        build_edge_feature_vector(edge, in_degree, out_degree) for edge in edges
    ]
    graph_features, graph_capture_batch_size = build_graph_feature_vector(
        exported_program,
        node_infos=node_infos,
        edges=edges,
        runtime_input_names=runtime_input_names,
        phase=phase,
        batch_size=batch_size,
        decode_output_length=decode_output_length,
        gpu_name=gpu_name,
    )

    edge_index = (
        torch.tensor(
            [(edge.source_index, edge.target_index) for edge in edges],
            dtype=torch.long,
        )
        .t()
        .contiguous()
        if edges
        else torch.empty((2, 0), dtype=torch.long)
    )
    edge_attr = (
        torch.tensor(edge_features, dtype=torch.float32)
        if edge_features
        else torch.empty((0, EDGE_FEATURE_DIM), dtype=torch.float32)
    )
    return Data(
        x=torch.tensor(node_features, dtype=torch.float32),
        edge_index=edge_index,
        edge_attr=edge_attr,
        graph_features=torch.tensor(graph_features, dtype=torch.float32).unsqueeze(0),
        op_type_ids=torch.tensor(
            [OP_TYPE_TO_INDEX[info.category] for info in node_infos],
            dtype=torch.long,
        ),
        graph_capture_batch_size=graph_capture_batch_size,
        graph_path=str(graph_path) if graph_path is not None else "",
    )


def build_node_infos(exported_program: ExportedProgram) -> list[FxNodeInfo]:
    constant_placeholders = {
        str(getattr(spec.arg, "name", ""))
        for spec in exported_program.graph_signature.input_specs
        if spec.kind == InputKind.CONSTANT_TENSOR
    }
    infos: list[FxNodeInfo] = []
    for node in exported_program.graph.nodes:
        if node.op != "call_function":
            continue
        output_specs = tuple(tensor_specs_from_value(node.meta.get("val")))
        if not output_specs:
            continue
        input_nodes = tuple(iter_fx_nodes((node.args, node.kwargs)))
        input_specs = build_input_tensor_specs(node, input_nodes)
        op_name = resolve_op_name(node.target)
        category = resolve_op_category(
            node,
            op_name,
            constant_placeholders=constant_placeholders,
        )
        attr_count = count_explicit_attributes(node)
        partial = FxNodeInfo(
            node=node,
            op_name=op_name,
            category=category,
            input_nodes=input_nodes,
            input_specs=input_specs,
            output_specs=output_specs,
            attr_count=attr_count,
            macs=0,
        )
        infos.append(
            FxNodeInfo(
                node=node,
                op_name=op_name,
                category=category,
                input_nodes=input_nodes,
                input_specs=input_specs,
                output_specs=output_specs,
                attr_count=attr_count,
                macs=estimate_node_macs(partial),
            )
        )
    return infos


def build_edges(
    node_infos: list[FxNodeInfo],
    node_to_index: dict[Node, int],
) -> list[FxEdge]:
    edges: list[FxEdge] = []
    for target_index, info in enumerate(node_infos):
        for source_node in info.input_nodes:
            source_index = node_to_index.get(source_node)
            if source_index is None:
                continue
            source_info = node_infos[source_index]
            edges.append(
                FxEdge(
                    source_index=source_index,
                    target_index=target_index,
                    tensor=resolve_edge_tensor_spec(source_info, info),
                )
            )
    return edges


def resolve_edge_tensor_spec(source: FxNodeInfo, target: FxNodeInfo) -> TensorSpec:
    selected = resolve_getitem_tensor_specs(target.node, source.node)
    if selected is not None and len(selected) == 1:
        return selected[0]
    if len(source.output_specs) != 1:
        raise ValueError(
            f"multi-output FX node must be consumed through getitem: {source.node.name}"
        )
    return source.output_specs[0]


def build_input_tensor_specs(
    node: Node,
    input_nodes: tuple[Node, ...],
) -> tuple[TensorSpec, ...]:
    specs: list[TensorSpec] = []
    for input_node in input_nodes:
        selected = resolve_getitem_tensor_specs(node, input_node)
        if selected is not None:
            specs.extend(selected)
        else:
            specs.extend(tensor_specs_from_value(input_node.meta.get("val")))
    return tuple(specs)


def resolve_getitem_tensor_specs(
    target_node: Node,
    source_node: Node,
) -> tuple[TensorSpec, ...] | None:
    if (
        target_node.target is not operator.getitem
        or target_node.args[0] is not source_node
    ):
        return None
    index = target_node.args[1]
    source_value = source_node.meta.get("val")
    if not isinstance(index, int) or not isinstance(source_value, (tuple, list)):
        return None
    return tuple(tensor_specs_from_value(source_value[index]))


def build_node_degrees(
    node_count: int,
    edges: list[FxEdge],
) -> tuple[list[int], list[int]]:
    in_degree = [0] * node_count
    out_degree = [0] * node_count
    for edge in edges:
        out_degree[edge.source_index] += 1
        in_degree[edge.target_index] += 1
    return in_degree, out_degree


def build_node_feature_vector(
    info: FxNodeInfo,
    *,
    in_degree: int,
    out_degree: int,
) -> list[float]:
    output_shape = info.output_specs[0].shape
    feature_vector = [
        float(info.macs),
        float(info.memory_bytes),
        0.0,
        float(len(info.input_nodes)),
        float(len(info.output_specs)),
        float(info.attr_count),
        float(in_degree),
        float(out_degree),
        safe_log1p(sum(spec.bytes for spec in info.input_specs)),
        safe_log1p(info.memory_bytes),
        safe_log1p(sum(spec.elements for spec in info.input_specs)),
        safe_log1p(sum(spec.elements for spec in info.output_specs)),
        float(max((len(spec.shape) for spec in info.input_specs), default=0)),
        float(max((len(spec.shape) for spec in info.output_specs), default=0)),
        safe_log1p(
            sum(count_nonbatch_elements(spec.shape) for spec in info.output_specs)
        ),
        *build_shape_dimension_logs(output_shape),
        float(info.output_specs[0].itemsize),
    ]
    if len(feature_vector) != NODE_FEATURE_DIM:
        raise ValueError(f"invalid node feature dimension: {len(feature_vector)}")
    return feature_vector


def build_edge_feature_vector(
    edge: FxEdge,
    in_degree: list[int],
    out_degree: list[int],
) -> list[float]:
    tensor = edge.tensor
    feature_vector = [
        float(tensor.bytes),
        float(len(tensor.shape)),
        float(tensor.elements),
        float(out_degree[edge.source_index]),
        float(in_degree[edge.target_index]),
        *build_shape_dimension_logs(tensor.shape),
        safe_log1p(count_nonbatch_elements(tensor.shape)),
        float(tensor.itemsize),
        float(not tensor.shape),
        float(any(dimension == 0 for dimension in tensor.shape)),
    ]
    if len(feature_vector) != EDGE_FEATURE_DIM:
        raise ValueError(f"invalid edge feature dimension: {len(feature_vector)}")
    return feature_vector


def build_graph_feature_vector(
    exported_program: ExportedProgram,
    *,
    node_infos: list[FxNodeInfo],
    edges: list[FxEdge],
    runtime_input_names: list[str],
    phase: str,
    batch_size: int,
    decode_output_length: int,
    gpu_name: str,
) -> tuple[list[float], int]:
    normalized_phase = phase.strip().lower()
    if normalized_phase not in PHASE_TO_INDEX:
        raise ValueError(f"unsupported phase: {phase}")
    normalized_gpu_name = normalize_gpu_name(gpu_name)
    signature_specs = exported_program.graph_signature.input_specs
    placeholders = {
        node.name: node
        for node in exported_program.graph.nodes
        if node.op == "placeholder"
    }
    parameter_specs = [
        tensor_spec_for_input(spec, placeholders)
        for spec in signature_specs
        if spec.kind in {InputKind.PARAMETER, InputKind.BUFFER}
    ]
    runtime_specs = [
        tensor_spec_for_input(spec, placeholders)
        for spec in signature_specs
        if spec.kind == InputKind.USER_INPUT
    ]
    if len(runtime_input_names) != len(runtime_specs):
        raise ValueError(
            "runtime input name count does not match graph signature: "
            f"{len(runtime_input_names)} != {len(runtime_specs)}"
        )
    graph_capture_batch_size = infer_graph_capture_batch_size(runtime_specs)
    all_tensor_specs = [
        *(
            tensor_spec
            for node in exported_program.graph.nodes
            for tensor_spec in tensor_specs_from_value(node.meta.get("val"))
        ),
    ]
    activation_bytes = sum(info.memory_bytes for info in node_infos)
    activation_elements = sum(
        spec.elements for info in node_infos for spec in info.output_specs
    )
    graph_output_indices = resolve_graph_output_indices(node_infos)
    feature_vector = [
        float(PHASE_TO_INDEX[normalized_phase]),
        float(batch_size),
        float(decode_output_length),
        *GPU_SPECS[normalized_gpu_name],
        float(len(parameter_specs)),
        float(sum(spec.elements for spec in parameter_specs)),
        float(sum(spec.bytes for spec in parameter_specs)),
        float(sum(info.macs for info in node_infos)),
        float(activation_bytes),
        0.0,
        safe_log1p(len(node_infos)),
        safe_log1p(len(edges)),
        safe_log1p(activation_bytes),
        safe_log1p(
            estimate_peak_live_activation_bytes(
                node_infos=node_infos,
                edges=edges,
                graph_output_indices=graph_output_indices,
            )
        ),
        safe_log1p(activation_elements),
        float(max((len(spec.shape) for spec in all_tensor_specs), default=0)),
        safe_log1p(
            max(
                (dimension for spec in all_tensor_specs for dimension in spec.shape),
                default=0,
            )
        ),
        float(len(runtime_specs)),
        safe_log1p(sum(spec.elements for spec in runtime_specs)),
        safe_log1p(sum(count_nonbatch_elements(spec.shape) for spec in runtime_specs)),
    ]
    if len(feature_vector) != GRAPH_FEATURE_DIM:
        raise ValueError(f"invalid graph feature dimension: {len(feature_vector)}")
    return feature_vector, graph_capture_batch_size


def infer_graph_capture_batch_size(
    runtime_specs: list[TensorSpec],
) -> int:
    if not runtime_specs:
        raise ValueError("exported graph has no runtime tensor inputs")
    capture_batch_size: int | None = None
    for tensor_spec in runtime_specs:
        if not tensor_spec.shape:
            raise ValueError("runtime inputs must include a batch dimension")
        input_batch_size = tensor_spec.shape[0]
        if input_batch_size <= 0:
            raise ValueError("runtime input batch dimension must be positive")
        if capture_batch_size is None:
            capture_batch_size = input_batch_size
        elif input_batch_size != capture_batch_size:
            raise ValueError(
                "inconsistent static runtime input batch sizes: "
                f"{capture_batch_size} != {input_batch_size}"
            )
    assert capture_batch_size is not None
    return capture_batch_size


def tensor_spec_for_input(spec: Any, placeholders: dict[str, Node]) -> TensorSpec:
    argument_name = getattr(spec.arg, "name", None)
    if not isinstance(argument_name, str) or argument_name not in placeholders:
        raise ValueError(f"graph input is not a tensor placeholder: {spec.arg}")
    tensor_specs = tuple(
        tensor_specs_from_value(placeholders[argument_name].meta.get("val"))
    )
    if len(tensor_specs) != 1:
        raise ValueError(f"graph input must contain one tensor: {argument_name}")
    return tensor_specs[0]


def resolve_graph_output_indices(node_infos: list[FxNodeInfo]) -> set[int]:
    output_node = list(node_infos[0].node.graph.nodes)[-1]
    if output_node.op != "output":
        raise ValueError("FX graph is missing output node")
    node_to_index = {info.node: index for index, info in enumerate(node_infos)}
    return {
        node_to_index[node]
        for node in iter_fx_nodes(output_node.args)
        if node in node_to_index
    }


def estimate_peak_live_activation_bytes(
    *,
    node_infos: list[FxNodeInfo],
    edges: list[FxEdge],
    graph_output_indices: set[int],
) -> int:
    node_count = len(node_infos)
    execution_order = validate_or_build_execution_order(node_count, edges)
    consumers: list[list[int]] = [[] for _ in range(node_count)]
    for edge in edges:
        consumers[edge.source_index].append(edge.target_index)
    releases: list[list[int]] = [[] for _ in range(node_count)]
    for index in range(node_count):
        if index in graph_output_indices:
            continue
        release_index = max(consumers[index]) if consumers[index] else index
        releases[release_index].append(index)

    live_bytes = 0
    peak_bytes = 0
    for node_index in execution_order:
        live_bytes += node_infos[node_index].memory_bytes
        peak_bytes = max(peak_bytes, live_bytes)
        for released_index in releases[node_index]:
            live_bytes -= node_infos[released_index].memory_bytes
        if live_bytes < 0:
            raise ValueError("peak live activation scan produced negative bytes")
    return peak_bytes


def validate_or_build_execution_order(
    node_count: int,
    edges: list[FxEdge],
) -> list[int]:
    if all(edge.source_index < edge.target_index for edge in edges):
        return list(range(node_count))
    outgoing: list[list[int]] = [[] for _ in range(node_count)]
    indegree = [0] * node_count
    for edge in edges:
        if not 0 <= edge.source_index < node_count:
            raise ValueError("FX edge source index is out of range")
        if not 0 <= edge.target_index < node_count:
            raise ValueError("FX edge target index is out of range")
        outgoing[edge.source_index].append(edge.target_index)
        indegree[edge.target_index] += 1
    queue = deque(index for index, degree in enumerate(indegree) if degree == 0)
    order: list[int] = []
    while queue:
        node_index = queue.popleft()
        order.append(node_index)
        for target_index in outgoing[node_index]:
            indegree[target_index] -= 1
            if indegree[target_index] == 0:
                queue.append(target_index)
    if len(order) != node_count:
        raise ValueError("FX graph contains cyclic dependencies")
    return order


def estimate_node_macs(info: FxNodeInfo) -> int:
    output_elements = sum(spec.elements for spec in info.output_specs)
    input_elements = info.input_specs[0].elements if info.input_specs else 0
    if info.op_name in CONV_OPS:
        return estimate_conv_macs(info, output_elements)
    if info.op_name in DENSE_OPS:
        return estimate_dense_macs(info, output_elements)
    if info.op_name in ATTENTION_OPS:
        return estimate_attention_macs(info, output_elements)
    if info.op_name in NORM_OPS:
        return input_elements * 5
    if info.op_name in POOL_OPS:
        return input_elements + info.memory_bytes
    if info.op_name in ACTIVATION_OPS:
        return output_elements * activation_instruction_weight(info.op_name)
    if info.op_name in ELEMENTWISE_OPS or info.op_name in {"max", "min"}:
        return output_elements * elementwise_instruction_weight(info.op_name)
    if info.op_name in REDUCE_OPS:
        return input_elements
    return 0


def estimate_conv_macs(info: FxNodeInfo, output_elements: int) -> int:
    if len(info.input_specs) < 2:
        raise ValueError(f"convolution is missing weight input: {info.node.name}")
    weight_shape = info.input_specs[1].shape
    if len(weight_shape) < 3:
        raise ValueError(f"invalid convolution weight shape: {weight_shape}")
    kernel_macs = math.prod(weight_shape[1:])
    bias_macs = output_elements if len(info.input_nodes) >= 3 else 0
    return output_elements * kernel_macs + bias_macs


def estimate_dense_macs(info: FxNodeInfo, output_elements: int) -> int:
    if info.op_name == "linear":
        if len(info.input_specs) < 2:
            raise ValueError(f"linear is missing weight input: {info.node.name}")
        inner_dim = info.input_specs[1].shape[-1]
        bias_macs = output_elements if len(info.input_nodes) >= 3 else 0
        return output_elements * inner_dim + bias_macs
    if info.op_name == "addmm":
        if len(info.input_specs) < 3:
            raise ValueError(f"addmm is missing matrix input: {info.node.name}")
        return output_elements * (info.input_specs[1].shape[-1] + 1)
    if len(info.input_specs) < 2:
        raise ValueError(f"matrix multiplication is missing input: {info.node.name}")
    return output_elements * info.input_specs[0].shape[-1]


def estimate_attention_macs(info: FxNodeInfo, output_elements: int) -> int:
    if info.op_name not in {
        "scaled_dot_product_attention",
        "_scaled_dot_product_attention_math",
        "_scaled_dot_product_flash_attention",
    }:
        return output_elements * 5
    if len(info.input_specs) < 2:
        raise ValueError(f"attention is missing key input: {info.node.name}")
    query_shape = info.input_specs[0].shape
    key_shape = info.input_specs[1].shape
    if len(query_shape) < 2 or len(key_shape) < 2:
        raise ValueError("attention inputs must have rank >= 2")
    batch_heads = math.prod(query_shape[:-2])
    query_length = query_shape[-2]
    key_length = key_shape[-2]
    head_dim = query_shape[-1]
    attention_elements = batch_heads * query_length * key_length
    return 2 * attention_elements * head_dim + 5 * attention_elements


def activation_instruction_weight(op_name: str) -> int:
    return {
        "erf": 32,
        "gelu": 8,
        "sigmoid": 4,
        "silu": 5,
        "softplus": 43,
        "tanh": 20,
    }.get(op_name, 1)


def elementwise_instruction_weight(op_name: str) -> int:
    return {
        "cos": 39,
        "div": 4,
        "exp": 32,
        "log": 43,
        "pow": 32,
        "rsqrt": 28,
        "sin": 39,
        "sqrt": 24,
    }.get(op_name, 1)


def resolve_op_name(target: object) -> str:
    if target is operator.getitem:
        return "getitem"
    schema = getattr(target, "_schema", None)
    schema_name = getattr(schema, "name", None)
    if isinstance(schema_name, str) and "::" in schema_name:
        return schema_name.split("::", maxsplit=1)[1]
    raise ValueError(f"unsupported FX call target: {target}")


def resolve_op_category(
    node: Node,
    op_name: str,
    *,
    constant_placeholders: set[str],
) -> str:
    if op_name == "getitem":
        source = node.args[0]
        source_value = source.meta.get("val") if isinstance(source, Node) else source
        return (
            "op_identity" if isinstance(source_value, (tuple, list)) else "op_embedding"
        )
    if op_name in CONV_OPS:
        return "op_conv"
    if op_name in DENSE_OPS:
        return "op_dense"
    if op_name in EMBEDDING_OPS:
        return "op_embedding"
    if op_name in ATTENTION_OPS:
        return "op_attention"
    if op_name in NORM_OPS:
        return "op_norm"
    if op_name in POOL_OPS:
        return "op_pool"
    if op_name in ACTIVATION_OPS:
        return "op_activation"
    if op_name in ELEMENTWISE_OPS:
        return "op_elementwise"
    if op_name in {"max", "min"} and str(node.target).endswith(".other"):
        return "op_elementwise"
    if op_name in REDUCE_OPS:
        return "op_reduce"
    if op_name in SHAPE_OPS:
        return "op_shape"
    if op_name in LAYOUT_OPS:
        return "op_layout"
    if op_name in JOIN_SPLIT_OPS:
        return "op_join_split"
    if op_name in CAST_OPS:
        return "op_cast"
    if op_name in CONSTANT_OPS or consumes_lifted_tensor_constant(
        node,
        constant_placeholders,
    ):
        return "op_constant"
    if op_name in IDENTITY_OPS:
        return "op_identity"
    raise ValueError(f"unsupported FX tensor operator: {node.target}")


def consumes_lifted_tensor_constant(
    node: Node,
    constant_placeholders: set[str],
) -> bool:
    return any(
        input_node.op == "placeholder" and input_node.name in constant_placeholders
        for input_node in iter_fx_nodes((node.args, node.kwargs))
    )


def tensor_specs_from_value(value: object) -> Iterator[TensorSpec]:
    if isinstance(value, torch.Tensor):
        shape = tuple(resolve_static_dimension(dimension) for dimension in value.shape)
        yield TensorSpec(shape=shape, itemsize=value.element_size())
        return
    if isinstance(value, (tuple, list)):
        for item in value:
            yield from tensor_specs_from_value(item)
        return
    if isinstance(value, dict):
        for item in value.values():
            yield from tensor_specs_from_value(item)


def resolve_static_dimension(value: object) -> int:
    if isinstance(value, torch.SymInt):
        raise ValueError(f"FX graph contains a symbolic dimension: {value}")
    if isinstance(value, int):
        dimension = value
    else:
        try:
            dimension = int(cast(Any, value))
        except (TypeError, ValueError, RuntimeError) as error:
            raise ValueError(
                f"FX graph contains a symbolic dimension: {value}"
            ) from error
    if dimension < 0:
        raise ValueError(f"tensor dimension must be non-negative: {dimension}")
    return dimension


def iter_fx_nodes(value: object) -> Iterator[Node]:
    if isinstance(value, Node):
        yield value
        return
    if isinstance(value, dict):
        for item in value.values():
            yield from iter_fx_nodes(item)
        return
    if isinstance(value, (tuple, list)):
        for item in value:
            yield from iter_fx_nodes(item)
        return
    if isinstance(value, slice):
        yield from iter_fx_nodes((value.start, value.stop, value.step))


def count_explicit_attributes(node: Node) -> int:
    values = [*node.args, *node.kwargs.values()]
    return sum(not any(iter_fx_nodes(value)) for value in values)


def build_shape_dimension_logs(shape: tuple[int, ...]) -> list[float]:
    dimensions = list(shape[:MAX_SHAPE_RANK])
    dimensions.extend([0] * (MAX_SHAPE_RANK - len(dimensions)))
    return [safe_log1p(dimension) for dimension in dimensions]


def count_elements(shape: tuple[int, ...]) -> int:
    return math.prod(shape) if shape else 1


def count_nonbatch_elements(shape: tuple[int, ...]) -> int:
    if not shape:
        return 0
    if len(shape) == 1:
        return 1
    return count_elements(shape[1:])


def safe_log1p(value: int | float) -> float:
    parsed = float(value)
    if parsed < 0.0 or not math.isfinite(parsed):
        raise ValueError(f"log1p input must be finite and non-negative: {value}")
    return math.log1p(parsed)
