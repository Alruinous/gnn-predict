from __future__ import annotations

import math
from collections import deque
from collections.abc import Iterable
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
    MAX_SHAPE_RANK,
    NODE_FEATURE_DIM,
    OP_TYPE_TO_INDEX,
    PHASE_TO_INDEX,
    normalize_gpu_name,
)
from onnx_support import install_onnx_tool_extensions

OP_TYPE_CATEGORY_BY_RAW_OP = {
    "Conv": "op_conv",
    "ConvTranspose": "op_conv",
    "Gemm": "op_dense",
    "MatMul": "op_dense",
    "Gather": "op_embedding",
    "GatherElements": "op_embedding",
    "GatherND": "op_embedding",
    "Softmax": "op_attention",
    "Einsum": "op_attention",
    "BatchNormalization": "op_norm",
    "LayerNormalization": "op_norm",
    "InstanceNormalization": "op_norm",
    "MaxPool": "op_pool",
    "AveragePool": "op_pool",
    "GlobalAveragePool": "op_pool",
    "GlobalMaxPool": "op_pool",
    "Relu": "op_activation",
    "Gelu": "op_activation",
    "Sigmoid": "op_activation",
    "Tanh": "op_activation",
    "Softplus": "op_activation",
    "Elu": "op_activation",
    "Selu": "op_activation",
    "LeakyRelu": "op_activation",
    "HardSigmoid": "op_activation",
    "HardSwish": "op_activation",
    "Erf": "op_activation",
    "Clip": "op_activation",
    "PRelu": "op_activation",
    "Add": "op_elementwise",
    "Sub": "op_elementwise",
    "Abs": "op_elementwise",
    "Mul": "op_elementwise",
    "Div": "op_elementwise",
    "Pow": "op_elementwise",
    "Sqrt": "op_elementwise",
    "Exp": "op_elementwise",
    "Log": "op_elementwise",
    "Sin": "op_elementwise",
    "Cos": "op_elementwise",
    "Neg": "op_elementwise",
    "Reciprocal": "op_elementwise",
    "Mod": "op_elementwise",
    "Not": "op_elementwise",
    "NonZero": "op_elementwise",
    "And": "op_elementwise",
    "Trilu": "op_elementwise",
    "Where": "op_elementwise",
    "Max": "op_elementwise",
    "Min": "op_elementwise",
    "Equal": "op_elementwise",
    "Greater": "op_elementwise",
    "GreaterOrEqual": "op_elementwise",
    "Less": "op_elementwise",
    "LessOrEqual": "op_elementwise",
    "IsNaN": "op_elementwise",
    "ReduceMean": "op_reduce",
    "ReduceSum": "op_reduce",
    "ReduceMax": "op_reduce",
    "ReduceMin": "op_reduce",
    "ArgMax": "op_reduce",
    "CumSum": "op_reduce",
    "Shape": "op_shape",
    "Size": "op_shape",
    "ConstantOfShape": "op_shape",
    "Range": "op_shape",
    "Reshape": "op_layout",
    "Transpose": "op_layout",
    "Flatten": "op_layout",
    "Squeeze": "op_layout",
    "Unsqueeze": "op_layout",
    "Resize": "op_layout",
    "Pad": "op_layout",
    "Concat": "op_join_split",
    "Split": "op_join_split",
    "Slice": "op_join_split",
    "Tile": "op_join_split",
    "Expand": "op_join_split",
    "ScatterND": "op_join_split",
    "Cast": "op_cast",
    "CastLike": "op_cast",
    "Constant": "op_constant",
    "Identity": "op_identity",
}


def build_graph_data_from_onnx(
    onnx_path: str | Path,
    *,
    batch_size: int = 1,
    gpu_name: str = "v100",
    phase: str = "training",
    decode_output_length: int = 0,
) -> Data:
    assert batch_size > 0
    assert decode_output_length >= 0
    model_path = Path(onnx_path)
    model = onnx.load(model_path)
    runtime_input_names = load_runtime_input_names(model)
    runtime_inputs = build_runtime_inputs(
        model,
        runtime_input_names=runtime_input_names,
        batch_size=batch_size,
    )
    graph_output_names = {output.name for output in model.graph.output if output.name}

    install_onnx_tool_extensions()
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
            graph,
            graph.nodemap[node_name],
            in_degree=in_degree[node_name_to_index[node_name]],
            out_degree=out_degree[node_name_to_index[node_name]],
        )
        for node_name in node_names
    ]
    op_type_ids = torch.tensor(
        build_op_type_ids(graph.nodemap[node_name] for node_name in node_names),
        dtype=torch.long,
    )
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
        model=model,
        runtime_input_names=runtime_input_names,
        phase=phase,
        batch_size=batch_size,
        decode_output_length=decode_output_length,
        gpu_name=gpu_name,
        graph=graph,
        node_infos=list(graph.nodemap.values()),
        raw_edges=raw_edges,
        runtime_inputs=runtime_inputs,
        graph_output_names=graph_output_names,
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
        op_type_ids=op_type_ids,
        onnx_path=str(model_path),
    )


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
    graph: Any,
    node_info: Any,
    *,
    in_degree: int,
    out_degree: int,
) -> list[float]:
    feature_vector = [
        float(sum(node_info.macs)),
        float(node_info.memory),
        float(node_info.params),
        float(len(node_info.input)),
        float(len(node_info.output)),
        float(len(node_info.attr)),
        float(in_degree),
        float(out_degree),
        *build_node_shape_feature_vector(graph, node_info),
    ]
    assert len(feature_vector) == NODE_FEATURE_DIM
    return feature_vector


def resolve_op_type_category(op_type: str) -> str:
    return OP_TYPE_CATEGORY_BY_RAW_OP[op_type]


def resolve_op_type_index(op_type: str) -> int:
    return OP_TYPE_TO_INDEX[resolve_op_type_category(op_type)]


def build_op_type_ids(node_infos: Iterable[Any]) -> list[int]:
    return [resolve_op_type_index(node_info.op_type) for node_info in node_infos]


def build_edge_feature_vector(
    graph: Any,
    tensor_name: str,
    source_index: int,
    target_index: int,
    in_degree: list[int],
    out_degree: list[int],
) -> list[float]:
    tensor_info = graph.tensormap[tensor_name]
    shape = resolve_tensor_shape(tensor_info)
    element_count = count_elements(shape)
    byte_count = element_count * resolve_tensor_itemsize(tensor_info)
    feature_vector = [
        float(byte_count),
        float(len(shape)),
        float(element_count),
        float(out_degree[source_index]),
        float(in_degree[target_index]),
        *build_edge_shape_feature_vector(shape, tensor_info),
    ]
    assert len(feature_vector) == EDGE_FEATURE_DIM
    return feature_vector


def build_node_shape_feature_vector(graph: Any, node_info: Any) -> list[float]:
    input_names = tuple(tensor_name for tensor_name in node_info.input if tensor_name)
    output_names = tuple(tensor_name for tensor_name in node_info.output if tensor_name)
    output_shape = (
        resolve_tensor_shape(graph.tensormap[output_names[0]]) if output_names else ()
    )
    output_itemsize = (
        resolve_tensor_itemsize(graph.tensormap[output_names[0]]) if output_names else 0
    )
    return [
        safe_log1p(sum_tensor_bytes(graph, input_names)),
        safe_log1p(sum_tensor_bytes(graph, output_names)),
        safe_log1p(sum_tensor_elements(graph, input_names)),
        safe_log1p(sum_tensor_elements(graph, output_names)),
        float(max_tensor_rank(graph, input_names)),
        float(max_tensor_rank(graph, output_names)),
        safe_log1p(sum_nonbatch_elements(graph, output_names)),
        *build_shape_dimension_logs(output_shape),
        float(output_itemsize),
    ]


def build_edge_shape_feature_vector(
    shape: tuple[int, ...],
    tensor_info: object,
) -> list[float]:
    return [
        *build_shape_dimension_logs(shape),
        safe_log1p(count_nonbatch_elements(shape)),
        float(resolve_tensor_itemsize(tensor_info)),
        float(not shape),
        float(any(dimension == 0 for dimension in shape)),
    ]


def build_graph_shape_feature_vector(
    graph: Any,
    *,
    node_infos: list[Any],
    raw_edges: list[tuple[int, int, str]],
    runtime_inputs: dict[str, np.ndarray],
    graph_output_names: set[str],
) -> list[float]:
    activation_tensor_names = {
        tensor_name
        for node_info in node_infos
        for tensor_name in node_info.output
        if tensor_name
    }
    tensor_shapes = [
        resolve_tensor_shape(tensor_info) for tensor_info in graph.tensormap.values()
    ]
    runtime_input_shapes = [
        tuple(int(dimension) for dimension in value.shape)
        for value in runtime_inputs.values()
    ]
    return [
        safe_log1p(len(node_infos)),
        safe_log1p(len(raw_edges)),
        safe_log1p(sum_tensor_bytes(graph, activation_tensor_names)),
        safe_log1p(
            estimate_peak_live_activation_bytes(
                graph=graph,
                node_names=list(graph.nodemap.keys()),
                raw_edges=raw_edges,
                graph_output_names=graph_output_names,
            )
        ),
        safe_log1p(sum_tensor_elements(graph, activation_tensor_names)),
        float(max((len(shape) for shape in tensor_shapes), default=0)),
        safe_log1p(
            max(
                (dimension for shape in tensor_shapes for dimension in shape),
                default=0,
            )
        ),
        float(len(runtime_inputs)),
        safe_log1p(sum(count_elements(shape) for shape in runtime_input_shapes)),
        safe_log1p(
            sum(count_nonbatch_elements(shape) for shape in runtime_input_shapes)
        ),
    ]


def estimate_peak_live_activation_bytes(
    *,
    graph: Any,
    node_names: list[str],
    raw_edges: list[tuple[int, int, str]],
    graph_output_names: set[str],
) -> int:
    node_count = len(node_names)
    if node_count == 0:
        return 0
    execution_order = validate_or_build_execution_order(node_count, raw_edges)
    producer: dict[str, int] = {}
    produced_by_node: list[list[str]] = [[] for _ in range(node_count)]
    tensor_bytes: dict[str, int] = {}
    for node_index, node_name in enumerate(node_names):
        node_info = graph.nodemap[node_name]
        for tensor_name in node_info.output:
            if not tensor_name:
                continue
            if tensor_name in producer:
                raise ValueError(f"duplicate activation tensor producer: {tensor_name}")
            producer[tensor_name] = node_index
            produced_by_node[node_index].append(tensor_name)
            tensor_bytes[tensor_name] = tensor_byte_count(graph, tensor_name)

    consumers: dict[str, list[int]] = {tensor_name: [] for tensor_name in producer}
    for node_index, node_name in enumerate(node_names):
        node_info = graph.nodemap[node_name]
        for tensor_name in node_info.input:
            if tensor_name in consumers:
                consumers[tensor_name].append(node_index)

    releases_by_node: list[list[str]] = [[] for _ in range(node_count)]
    for tensor_name, producer_index in producer.items():
        if tensor_name in graph_output_names:
            continue
        tensor_consumers = consumers[tensor_name]
        release_index = max(tensor_consumers) if tensor_consumers else producer_index
        releases_by_node[release_index].append(tensor_name)

    live_bytes = 0
    peak_live_bytes = 0
    for node_index in execution_order:
        for tensor_name in produced_by_node[node_index]:
            live_bytes += tensor_bytes[tensor_name]
        peak_live_bytes = max(peak_live_bytes, live_bytes)
        for tensor_name in releases_by_node[node_index]:
            live_bytes -= tensor_bytes[tensor_name]
            if live_bytes < 0:
                raise ValueError("peak live activation scan produced negative bytes")
    return peak_live_bytes


def validate_or_build_execution_order(
    node_count: int,
    raw_edges: list[tuple[int, int, str]],
) -> list[int]:
    if all(
        0 <= source_index < node_count
        and 0 <= target_index < node_count
        and source_index < target_index
        for source_index, target_index, _ in raw_edges
    ):
        return list(range(node_count))
    outgoing: list[list[int]] = [[] for _ in range(node_count)]
    indegree = [0] * node_count
    for source_index, target_index, _ in raw_edges:
        if not 0 <= source_index < node_count or not 0 <= target_index < node_count:
            raise ValueError("raw edge node index out of range")
        outgoing[source_index].append(target_index)
        indegree[target_index] += 1
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
        raise ValueError("ONNX graph contains cyclic dependencies")
    return order


def tensor_byte_count(graph: Any, tensor_name: str) -> int:
    tensor_info = graph.tensormap[tensor_name]
    return count_elements(resolve_tensor_shape(tensor_info)) * (
        resolve_tensor_itemsize(tensor_info)
    )


def build_shape_dimension_logs(shape: tuple[int, ...]) -> list[float]:
    dimensions = list(shape[:MAX_SHAPE_RANK])
    dimensions.extend([0] * (MAX_SHAPE_RANK - len(dimensions)))
    return [safe_log1p(dimension) for dimension in dimensions]


def sum_tensor_bytes(graph: Any, tensor_names: Iterable[str]) -> int:
    total = 0
    for tensor_name in tensor_names:
        total += tensor_byte_count(graph, tensor_name)
    return total


def sum_tensor_elements(graph: Any, tensor_names: Iterable[str]) -> int:
    return sum(
        count_elements(resolve_tensor_shape(graph.tensormap[tensor_name]))
        for tensor_name in tensor_names
    )


def sum_nonbatch_elements(graph: Any, tensor_names: Iterable[str]) -> int:
    return sum(
        count_nonbatch_elements(resolve_tensor_shape(graph.tensormap[tensor_name]))
        for tensor_name in tensor_names
    )


def max_tensor_rank(graph: Any, tensor_names: Iterable[str]) -> int:
    return max(
        (
            len(resolve_tensor_shape(graph.tensormap[tensor_name]))
            for tensor_name in tensor_names
        ),
        default=0,
    )


def build_graph_feature_vector(
    *,
    model: onnx.ModelProto,
    runtime_input_names: list[str],
    phase: str,
    batch_size: int,
    decode_output_length: int,
    gpu_name: str,
    graph: Any,
    node_infos: list[Any],
    raw_edges: list[tuple[int, int, str]],
    runtime_inputs: dict[str, np.ndarray],
    graph_output_names: set[str],
) -> list[float]:
    normalized_phase = phase.strip().lower()
    assert normalized_phase in PHASE_TO_INDEX, phase
    normalized_gpu_name = normalize_gpu_name(gpu_name)
    assert normalized_gpu_name in GPU_SPECS, gpu_name
    parameter_inputs = list(
        filter(lambda x: x.name not in set(runtime_input_names), model.graph.input)
    )
    parameter_input_count = len(parameter_inputs)
    element_count = 0
    byte_count = 0
    for value in parameter_inputs:
        shape = tuple(map(lambda x: int(x.dim_value), value.type.tensor_type.shape.dim))
        dtype = resolve_tensor_np_dtype(value)
        elements = count_elements(shape)
        element_count += elements
        byte_count += elements * np.dtype(dtype).itemsize
    feature_vector = [
        float(PHASE_TO_INDEX[normalized_phase]),
        float(batch_size),
        float(decode_output_length),
        *GPU_SPECS[normalized_gpu_name],
        float(parameter_input_count),
        float(element_count),
        float(byte_count),
        float(sum(graph.macs)),
        float(graph.memory),
        float(graph.params),
        *build_graph_shape_feature_vector(
            graph,
            node_infos=node_infos,
            raw_edges=raw_edges,
            runtime_inputs=runtime_inputs,
            graph_output_names=graph_output_names,
        ),
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


def resolve_tensor_shape(tensor_info: object) -> tuple[int, ...]:
    shape = getattr(tensor_info, "shape", None)
    assert shape is not None
    if isinstance(shape, int):
        assert shape >= 0, shape
        return (shape,)
    parsed_shape = tuple(int(dimension) for dimension in shape)
    assert all(dimension >= 0 for dimension in parsed_shape), shape
    return parsed_shape


def resolve_tensor_itemsize(tensor_info: object) -> int:
    dtype = getattr(tensor_info, "dtype", None)
    assert dtype is not None, type(tensor_info)
    return int(np.dtype(dtype).itemsize)


def count_elements(shape: tuple[int, ...]) -> int:
    if not shape:
        return 1
    return math.prod(shape)


def count_nonbatch_elements(shape: tuple[int, ...]) -> int:
    if not shape:
        return 0
    if len(shape) == 1:
        return 1
    return count_elements(shape[1:])


def safe_log1p(value: int | float) -> float:
    parsed = float(value)
    assert parsed >= 0.0 and math.isfinite(parsed), value
    return math.log1p(parsed)
