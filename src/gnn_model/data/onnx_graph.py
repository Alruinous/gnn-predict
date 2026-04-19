from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import onnx
import torch
from onnx import numpy_helper
from torch_geometric.data import Data

from gnn_model.data.constants import (
    COMMON_OP_TYPES,
    EDGE_FEATURE_DIM,
    GPU_SPECS,
    GRAPH_FEATURE_DIM,
    GRAPH_METRIC_DIM,
    NODE_FEATURE_DIM,
    OP_TYPE_TO_INDEX,
)


@dataclass(frozen=True)
class TensorStats:
    shape: tuple[int, ...]
    elem_type: int
    element_count: int
    byte_size: int

    @property
    def rank(self) -> int:
        return len(self.shape)


@dataclass(frozen=True)
class NodeStats:
    op_type: str
    parameter_count: int
    parameter_bytes: int
    input_count: int
    output_count: int
    attr_count: int
    input_rank: float
    output_rank: float
    input_elements: int
    output_elements: int
    estimated_flops: float


def build_graph_data_from_onnx(
    onnx_path: str | Path,
    *,
    batch_size: int = 1,
    gpu_name: str = "v100",
) -> Data:
    assert batch_size > 0
    model_path = Path(onnx_path)
    inferred_model = onnx.shape_inference.infer_shapes(onnx.load(model_path))
    nodes = list(inferred_model.graph.node)
    if not nodes:
        raise ValueError(f"ONNX graph must contain at least one node: {model_path}")

    tensor_stats = collect_tensor_stats(inferred_model)
    initializer_arrays = {
        initializer.name: numpy_helper.to_array(initializer)
        for initializer in inferred_model.graph.initializer
    }
    producer_by_output = build_producer_index(nodes)
    node_stats = [
        build_node_stats(node, tensor_stats, initializer_arrays) for node in nodes
    ]
    raw_edges = build_raw_edges(nodes, producer_by_output)
    if not raw_edges:
        raw_edges = [(index, index, "") for index in range(len(nodes))]

    in_degree = [0] * len(nodes)
    out_degree = [0] * len(nodes)
    for source_index, target_index, _ in raw_edges:
        out_degree[source_index] += 1
        in_degree[target_index] += 1

    node_features = [build_node_feature_vector(stats) for stats in node_stats]
    edge_features = [
        build_edge_feature_vector(
            tensor_name,
            source_index,
            target_index,
            tensor_stats,
            in_degree,
            out_degree,
        )
        for source_index, target_index, tensor_name in raw_edges
    ]
    graph_features = build_graph_feature_vector(
        inferred_model,
        node_stats,
        raw_edges,
        initializer_arrays,
        batch_size=batch_size,
        gpu_name=gpu_name,
    )
    graph_metrics = build_graph_metric_vector(node_stats, tensor_stats, nodes)
    edge_index = torch.tensor(
        [[source_index, target_index] for source_index, target_index, _ in raw_edges],
        dtype=torch.long,
    ).t().contiguous()

    return Data(
        x=torch.tensor(node_features, dtype=torch.float32),
        edge_index=edge_index,
        edge_attr=torch.tensor(edge_features, dtype=torch.float32),
        graph_features=torch.tensor(graph_features, dtype=torch.float32).unsqueeze(0),
        graph_metrics=torch.tensor(graph_metrics, dtype=torch.float32).unsqueeze(0),
        node_op_token_id=torch.tensor(
            [
                OP_TYPE_TO_INDEX.get(stats.op_type, len(COMMON_OP_TYPES))
                for stats in node_stats
            ],
            dtype=torch.long,
        ),
        onnx_path=str(model_path),
    )


def collect_tensor_stats(model: onnx.ModelProto) -> dict[str, TensorStats]:
    tensor_stats: dict[str, TensorStats] = {}
    value_infos = [
        *model.graph.input,
        *model.graph.value_info,
        *model.graph.output,
    ]
    for value_info in value_infos:
        if not value_info.name or value_info.name in tensor_stats:
            continue
        tensor_type = value_info.type.tensor_type
        if not tensor_type.HasField("shape"):
            continue
        shape = tuple(
            resolve_dimension_value(dimension)
            for dimension in tensor_type.shape.dim
        )
        element_count = count_elements(shape)
        dtype = onnx.helper.tensor_dtype_to_np_dtype(tensor_type.elem_type)
        byte_size = int(element_count * np_dtype_nbytes(dtype))
        tensor_stats[value_info.name] = TensorStats(
            shape=shape,
            elem_type=tensor_type.elem_type,
            element_count=element_count,
            byte_size=byte_size,
        )
    return tensor_stats


def build_producer_index(nodes: list[onnx.NodeProto]) -> dict[str, int]:
    producer_by_output: dict[str, int] = {}
    for index, node in enumerate(nodes):
        for output_name in node.output:
            if output_name:
                producer_by_output[output_name] = index
    return producer_by_output


def build_raw_edges(
    nodes: list[onnx.NodeProto],
    producer_by_output: dict[str, int],
) -> list[tuple[int, int, str]]:
    edges: list[tuple[int, int, str]] = []
    for target_index, node in enumerate(nodes):
        edges.extend(
            (producer_by_output[input_name], target_index, input_name)
            for input_name in node.input
            if input_name and input_name in producer_by_output
        )
    return edges


def build_node_stats(
    node: onnx.NodeProto,
    tensor_stats: dict[str, TensorStats],
    initializer_arrays: dict[str, object],
) -> NodeStats:
    input_stats = [tensor_stats[name] for name in node.input if name in tensor_stats]
    output_stats = [tensor_stats[name] for name in node.output if name in tensor_stats]
    parameter_arrays = [
        initializer_arrays[name] for name in node.input if name in initializer_arrays
    ]
    parameter_count = int(sum(array.size for array in parameter_arrays))
    parameter_bytes = int(sum(array.nbytes for array in parameter_arrays))
    input_elements = int(sum(stats.element_count for stats in input_stats))
    output_elements = int(sum(stats.element_count for stats in output_stats))
    input_rank = average([stats.rank for stats in input_stats])
    output_rank = average([stats.rank for stats in output_stats])
    estimated_flops = estimate_node_flops(
        node.op_type,
        input_elements,
        output_elements,
        parameter_count,
    )
    return NodeStats(
        op_type=normalize_op_type(node.op_type),
        parameter_count=parameter_count,
        parameter_bytes=parameter_bytes,
        input_count=len(node.input),
        output_count=len(node.output),
        attr_count=len(node.attribute),
        input_rank=input_rank,
        output_rank=output_rank,
        input_elements=input_elements,
        output_elements=output_elements,
        estimated_flops=estimated_flops,
    )


def build_node_feature_vector(stats: NodeStats) -> list[float]:
    op_features = [0.0] * (len(COMMON_OP_TYPES) + 1)
    op_index = OP_TYPE_TO_INDEX.get(stats.op_type, len(COMMON_OP_TYPES))
    op_features[op_index] = 1.0
    numeric_features = [
        log_scale(stats.parameter_count, 8.0),
        log_scale(stats.estimated_flops, 10.0),
        clamp_ratio(stats.input_count, 8.0),
        clamp_ratio(stats.output_count, 8.0),
        clamp_ratio(stats.attr_count, 16.0),
        clamp_ratio(stats.input_rank, 8.0),
        clamp_ratio(stats.output_rank, 8.0),
        log_scale(stats.input_elements, 8.0),
        log_scale(stats.output_elements, 8.0),
    ]
    feature_vector = op_features + numeric_features
    if len(feature_vector) != NODE_FEATURE_DIM:
        raise ValueError(
            f"node feature dim mismatch: {len(feature_vector)} != {NODE_FEATURE_DIM}"
        )
    return feature_vector


def build_edge_feature_vector(
    tensor_name: str,
    source_index: int,
    target_index: int,
    tensor_stats: dict[str, TensorStats],
    in_degree: list[int],
    out_degree: list[int],
) -> list[float]:
    stats = tensor_stats.get(
        tensor_name,
        TensorStats(
            shape=(),
            elem_type=onnx.TensorProto.FLOAT,
            element_count=1,
            byte_size=4,
        ),
    )
    feature_vector = [
        log_scale(stats.element_count, 8.0),
        clamp_ratio(stats.rank, 8.0),
        log_scale(stats.byte_size, 10.0),
        clamp_ratio(out_degree[source_index], 8.0),
        clamp_ratio(in_degree[target_index], 8.0),
        1.0 if source_index == target_index else 0.0,
        1.0 if stats.rank > 0 else 0.0,
        1.0 if stats.elem_type == onnx.TensorProto.FLOAT else 0.0,
    ]
    if len(feature_vector) != EDGE_FEATURE_DIM:
        raise ValueError(
            f"edge feature dim mismatch: {len(feature_vector)} != {EDGE_FEATURE_DIM}"
        )
    return feature_vector


def build_graph_feature_vector(
    model: onnx.ModelProto,
    node_stats: list[NodeStats],
    raw_edges: list[tuple[int, int, str]],
    initializer_arrays: dict[str, object],
    *,
    batch_size: int,
    gpu_name: str,
) -> list[float]:
    node_count = len(node_stats)
    edge_count = len(raw_edges)
    max_edges = max(node_count * max(node_count - 1, 1), 1)
    density = edge_count / max_edges
    avg_in_degree = average([stats.input_count for stats in node_stats])
    avg_out_degree = average([stats.output_count for stats in node_stats])
    op_diversity = len({stats.op_type for stats in node_stats}) / max(node_count, 1)
    initializer_count = len(initializer_arrays)
    initializer_bytes = int(sum(array.nbytes for array in initializer_arrays.values()))
    avg_attr_count = average([stats.attr_count for stats in node_stats])
    avg_rank = average([stats.output_rank for stats in node_stats])
    runtime_input_count = len(model.graph.input)
    output_count = len(model.graph.output)
    feature_vector = [
        log_scale(node_count, 4.0),
        log_scale(edge_count, 4.0),
        max(0.0, min(1.0, density)),
        clamp_ratio(avg_in_degree, 8.0),
        clamp_ratio(avg_out_degree, 8.0),
        max(0.0, min(1.0, op_diversity)),
        log_scale(initializer_count, 4.0),
        log_scale(initializer_bytes, 10.0),
        clamp_ratio(runtime_input_count, 8.0),
        clamp_ratio(output_count, 8.0),
        clamp_ratio(avg_attr_count, 8.0),
        clamp_ratio(avg_rank, 8.0),
        *build_system_feature_vector(batch_size=batch_size, gpu_name=gpu_name),
    ]
    if len(feature_vector) != GRAPH_FEATURE_DIM:
        raise ValueError(
            f"graph feature dim mismatch: {len(feature_vector)} != {GRAPH_FEATURE_DIM}"
        )
    return feature_vector


def build_system_feature_vector(*, batch_size: int, gpu_name: str) -> list[float]:
    normalized_gpu_name = normalize_gpu_name(gpu_name)
    assert normalized_gpu_name in GPU_SPECS, gpu_name
    return [*GPU_SPECS[normalized_gpu_name], float(batch_size)]


def normalize_gpu_name(value: object) -> str:
    normalized = str(value).strip().lower()
    if "a100" in normalized:
        return "a100"
    if "v100" in normalized:
        return "v100"
    raise AssertionError(f"unsupported gpu_name: {value}")


def build_graph_metric_vector(
    node_stats: list[NodeStats],
    tensor_stats: dict[str, TensorStats],
    nodes: list[onnx.NodeProto],
) -> list[float]:
    total_flops = float(sum(stats.estimated_flops for stats in node_stats))
    activation_bytes = int(
        sum(
            tensor_stats[name].byte_size
            for node in nodes
            for name in node.output
            if name in tensor_stats
        )
    )
    parameter_bytes = int(sum(stats.parameter_bytes for stats in node_stats))
    estimated_latency_sec = (total_flops / 5e8) + (
        (activation_bytes + parameter_bytes) / 2e8
    )
    graph_memory_bytes = float(activation_bytes + parameter_bytes)
    metrics = [
        max(estimated_latency_sec, 1e-6),
        graph_memory_bytes,
        max(total_flops, 1.0),
    ]
    if len(metrics) != GRAPH_METRIC_DIM:
        raise ValueError(
            f"graph metric dim mismatch: {len(metrics)} != {GRAPH_METRIC_DIM}"
        )
    return metrics


def estimate_node_flops(
    op_type: str,
    input_elements: int,
    output_elements: int,
    parameter_count: int,
) -> float:
    normalized_type = normalize_op_type(op_type)
    if normalized_type == "Conv":
        return float(max(output_elements, 1) * max(parameter_count, 1))
    if normalized_type in {"Gemm", "MatMul"}:
        return float(max(input_elements, 1) * max(output_elements, 1))
    if normalized_type in {
        "Relu",
        "Add",
        "Mul",
        "BatchNormalization",
        "AveragePool",
        "MaxPool",
    }:
        return float(max(output_elements, 1))
    return float(max(input_elements + output_elements + parameter_count, 1))


def normalize_op_type(op_type: str) -> str:
    normalized = op_type.strip()
    if normalized in OP_TYPE_TO_INDEX:
        return normalized
    return normalized or "Unknown"


def resolve_dimension_value(dimension: onnx.TensorShapeProto.Dimension) -> int:
    if dimension.HasField("dim_value") and dimension.dim_value > 0:
        return int(dimension.dim_value)
    return 1


def count_elements(shape: tuple[int, ...]) -> int:
    if not shape:
        return 1
    return max(math.prod(shape), 1)


def np_dtype_nbytes(dtype: object) -> int:
    return int(np.dtype(dtype).itemsize)


def average(values: list[int | float]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def log_scale(value: int | float, denominator: float) -> float:
    return max(0.0, min(1.0, math.log10(max(float(value), 1.0)) / denominator))


def clamp_ratio(value: int | float, denominator: float) -> float:
    return max(0.0, min(1.0, float(value) / denominator))
