from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnx_tool
import torch
from torch_geometric.data import Data

from common.onnx_initializer import load_runtime_input_names
from gnn_model.data.constants import (
    EDGE_FEATURE_DIM,
    GPU_SPECS,
    GRAPH_FEATURE_DIM,
    NODE_FEATURE_DIM,
    PHASE_TO_INDEX,
)


def build_graph_data_from_onnx(
    onnx_path: str | Path,
    *,
    batch_size: int = 1,
    gpu_name: str = "v100",
    phase: str = "training",
    sample_count: int = 1,
) -> Data:
    assert batch_size > 0
    assert sample_count > 0
    model_path = Path(onnx_path)
    model = onnx.load(model_path)
    runtime_input_names = load_runtime_input_names(model)
    parameter_input_stats = collect_parameter_input_stats(
        model,
        runtime_input_names=runtime_input_names,
    )
    runtime_inputs = build_runtime_inputs(
        model,
        runtime_input_names=runtime_input_names,
        batch_size=batch_size,
    )

    tool_model = onnx_tool.loadmodel(str(model_path))
    graph = tool_model.graph
    graph.shape_infer(runtime_inputs)
    graph.profile()

    node_names = list(graph.nodemap.keys())
    assert node_names, model_path

    node_name_to_index = {name: index for index, name in enumerate(node_names)}
    output_name_to_node_name = {
        output_name: node_name
        for node_name, info in graph.nodemap.items()
        for output_name in info.output
        if output_name
    }
    raw_edges = build_raw_edges(graph, node_name_to_index, output_name_to_node_name)

    in_degree, out_degree = build_node_degrees(len(node_names), raw_edges)
    node_features = [
        build_node_feature_vector(
            graph.nodemap[node_name],
            in_degree=in_degree[node_name_to_index[node_name]],
            out_degree=out_degree[node_name_to_index[node_name]],
        )
        for node_name in node_names
    ]
    edge_features = [
        build_edge_feature_vector(
            graph,
            tensor_name,
            source_index,
            target_index,
            in_degree,
            out_degree,
        )
        for source_index, target_index, tensor_name in raw_edges
    ]
    graph_features = build_graph_feature_vector(
        phase=phase,
        batch_size=batch_size,
        sample_count=sample_count,
        gpu_name=gpu_name,
        parameter_input_stats=parameter_input_stats,
        graph=graph,
    )

    edge_index = (
        torch.tensor(
            [
                [source_index, target_index]
                for source_index, target_index, _ in raw_edges
            ],
            dtype=torch.long,
        )
        .t()
        .contiguous()
        if raw_edges
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
        onnx_path=str(model_path),
    )


def collect_parameter_input_stats(
    model: onnx.ModelProto,
    *,
    runtime_input_names: list[str],
) -> dict[str, float]:
    runtime_input_name_set = set(runtime_input_names)
    parameter_inputs = [
        value for value in model.graph.input if value.name not in runtime_input_name_set
    ]
    element_count = 0
    byte_count = 0
    for value in parameter_inputs:
        shape = resolve_static_tensor_shape(value)
        dtype = resolve_tensor_np_dtype(value)
        elements = count_elements(shape)
        element_count += elements
        byte_count += elements * np.dtype(dtype).itemsize
    return {
        "parameter_input_count": float(len(parameter_inputs)),
        "parameter_input_element_count": float(element_count),
        "parameter_input_bytes": float(byte_count),
    }


def build_runtime_inputs(
    model: onnx.ModelProto,
    *,
    runtime_input_names: list[str],
    batch_size: int,
) -> dict[str, np.ndarray]:
    graph_inputs = {value.name: value for value in model.graph.input}
    assert set(runtime_input_names) <= set(graph_inputs)
    return {
        name: np.zeros(
            resolve_runtime_tensor_shape(graph_inputs[name], batch_size=batch_size),
            dtype=resolve_tensor_np_dtype(graph_inputs[name]),
        )
        for name in runtime_input_names
    }


def build_raw_edges(
    graph: Any,
    node_name_to_index: dict[str, int],
    output_name_to_node_name: dict[str, str],
) -> list[tuple[int, int, str]]:
    edges: list[tuple[int, int, str]] = []
    for target_name, info in graph.nodemap.items():
        target_index = node_name_to_index[target_name]
        edges.extend(
            (
                node_name_to_index[output_name_to_node_name[input_name]],
                target_index,
                input_name,
            )
            for input_name in info.input
            if input_name in output_name_to_node_name
        )
    return edges


def build_node_degrees(
    node_count: int,
    raw_edges: list[tuple[int, int, str]],
) -> tuple[list[int], list[int]]:
    in_degree = [0] * node_count
    out_degree = [0] * node_count
    for source_index, target_index, _ in raw_edges:
        out_degree[source_index] += 1
        in_degree[target_index] += 1
    return in_degree, out_degree


def build_node_feature_vector(
    node_info: Any,
    *,
    in_degree: int,
    out_degree: int,
) -> list[float]:
    feature_vector = [
        float(sum_profile_values(node_info.macs)),
        float(node_info.memory),
        float(node_info.params),
        float(len(node_info.input)),
        float(len(node_info.output)),
        float(len(node_info.attr)),
        float(in_degree),
        float(out_degree),
    ]
    assert len(feature_vector) == NODE_FEATURE_DIM
    return feature_vector


def build_edge_feature_vector(
    graph: Any,
    tensor_name: str,
    source_index: int,
    target_index: int,
    in_degree: list[int],
    out_degree: list[int],
) -> list[float]:
    tensor_info = graph.tensormap[tensor_name]
    shape = resolve_profile_tensor_shape(tensor_info)
    element_count = count_elements(shape)
    byte_count = element_count * resolve_profile_tensor_itemsize(tensor_info)
    feature_vector = [
        float(byte_count),
        float(len(shape)),
        float(element_count),
        float(out_degree[source_index]),
        float(in_degree[target_index]),
    ]
    assert len(feature_vector) == EDGE_FEATURE_DIM
    return feature_vector


def build_graph_feature_vector(
    *,
    phase: str,
    batch_size: int,
    sample_count: int,
    gpu_name: str,
    parameter_input_stats: dict[str, float],
    graph: Any,
) -> list[float]:
    normalized_phase = phase.strip().lower()
    assert normalized_phase in PHASE_TO_INDEX, phase
    normalized_gpu_name = normalize_gpu_name(gpu_name)
    assert normalized_gpu_name in GPU_SPECS, gpu_name
    total_macs = float(sum_profile_values(graph.macs))
    feature_vector = [
        float(PHASE_TO_INDEX[normalized_phase]),
        float(batch_size),
        float(sample_count),
        *GPU_SPECS[normalized_gpu_name],
        parameter_input_stats["parameter_input_count"],
        parameter_input_stats["parameter_input_element_count"],
        parameter_input_stats["parameter_input_bytes"],
        total_macs,
        total_macs * 2.0,
        float(graph.memory),
        float(graph.params),
    ]
    assert len(feature_vector) == GRAPH_FEATURE_DIM
    return feature_vector


def resolve_static_tensor_shape(value: onnx.ValueInfoProto) -> tuple[int, ...]:
    shape: list[int] = []
    for dimension in value.type.tensor_type.shape.dim:
        assert dimension.HasField("dim_value") and dimension.dim_value > 0, value.name
        shape.append(int(dimension.dim_value))
    return tuple(shape)


def resolve_runtime_tensor_shape(
    value: onnx.ValueInfoProto,
    *,
    batch_size: int,
) -> tuple[int, ...]:
    shape: list[int] = []
    for index, dimension in enumerate(value.type.tensor_type.shape.dim):
        if dimension.HasField("dim_value") and dimension.dim_value > 0:
            shape.append(int(dimension.dim_value))
        else:
            assert index == 0, value.name
            shape.append(batch_size)
    return tuple(shape)


def resolve_tensor_np_dtype(value: onnx.ValueInfoProto) -> np.dtype:
    elem_type = value.type.tensor_type.elem_type
    assert elem_type != onnx.TensorProto.UNDEFINED, value.name
    return np.dtype(onnx.helper.tensor_dtype_to_np_dtype(elem_type))


def resolve_profile_tensor_shape(tensor_info: object) -> tuple[int, ...]:
    shape = getattr(tensor_info, "shape", None)
    assert shape is not None
    if isinstance(shape, int):
        assert shape > 0, shape
        return (shape,)
    parsed_shape = tuple(int(dimension) for dimension in shape)
    assert all(dimension > 0 for dimension in parsed_shape), shape
    return parsed_shape


def resolve_profile_tensor_itemsize(tensor_info: object) -> int:
    dtype = getattr(tensor_info, "dtype", None)
    assert dtype is not None, type(tensor_info)
    return int(np.dtype(dtype).itemsize)


def normalize_gpu_name(value: object) -> str:
    normalized = str(value).strip().lower()
    if "a100" in normalized:
        return "a100"
    if "v100" in normalized:
        return "v100"
    raise AssertionError(f"unsupported gpu_name: {value}")


def count_elements(shape: tuple[int, ...]) -> int:
    if not shape:
        return 1
    return math.prod(shape)


def sum_profile_values(values: object) -> float:
    if isinstance(values, (list, tuple)):
        total = 0.0
        for value in values:
            assert isinstance(value, (int, float, np.number)), type(value)
            total += float(value)
        return total
    assert isinstance(values, (int, float, np.number)), type(values)
    return float(values)
