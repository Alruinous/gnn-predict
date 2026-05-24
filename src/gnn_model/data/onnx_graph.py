from __future__ import annotations

import math
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnx_tool
import torch
from onnx_tool.node import (
    ADD_MACS,
    CMP_MACS,
    EXP_MACS,
    LOG_MACS,
    MUL_MACS,
    PWNode,
)
from onnx_tool.node import (
    SqueezeNode as OnnxToolSqueezeNode,
)
from onnx_tool.utils import NODE_REGISTRY
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
    "Add": "op_elementwise",
    "Sub": "op_elementwise",
    "Mul": "op_elementwise",
    "Div": "op_elementwise",
    "Pow": "op_elementwise",
    "Where": "op_elementwise",
    "Max": "op_elementwise",
    "Min": "op_elementwise",
    "Equal": "op_elementwise",
    "Greater": "op_elementwise",
    "Less": "op_elementwise",
    "ReduceMean": "op_reduce",
    "ReduceSum": "op_reduce",
    "ReduceMax": "op_reduce",
    "ReduceMin": "op_reduce",
    "Shape": "op_shape",
    "Size": "op_shape",
    "ConstantOfShape": "op_shape",
    "Reshape": "op_layout",
    "Transpose": "op_layout",
    "Flatten": "op_layout",
    "Squeeze": "op_layout",
    "Unsqueeze": "op_layout",
    "Concat": "op_join_split",
    "Split": "op_join_split",
    "Slice": "op_join_split",
    "Tile": "op_join_split",
    "Expand": "op_join_split",
    "Cast": "op_cast",
    "CastLike": "op_cast",
    "Constant": "op_constant",
}


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

    register_onnx_tool_extensions()
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
        phase=phase,
        batch_size=batch_size,
        sample_count=sample_count,
        gpu_name=gpu_name,
        parameter_input_stats=parameter_input_stats,
        graph=graph,
        node_infos=list(graph.nodemap.values()),
        raw_edges=raw_edges,
        runtime_inputs=runtime_inputs,
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


def register_onnx_tool_extensions() -> None:
    replace_onnx_tool_node(SqueezeNode)
    for node_class in (SoftplusNode, EluNode, SeluNode):
        if NODE_REGISTRY.get(node_class.__name__) is None:
            NODE_REGISTRY.register(node_class)


def replace_onnx_tool_node(node_class: type[Any]) -> None:
    if NODE_REGISTRY.get(node_class.__name__) is not node_class:
        NODE_REGISTRY._obj_map[node_class.__name__] = node_class


class SqueezeNode(OnnxToolSqueezeNode):
    def shape_infer(self, intensors: list[Any], outtensors: list[Any]) -> None:
        inshape = intensors[0].get_shape()
        axes = resolve_squeeze_axes(inshape, self, intensors)
        outshape = [
            dimension for index, dimension in enumerate(inshape) if index not in axes
        ]
        outtensors[0].update_shape(outshape)
        outtensors[0].update_dtype(intensors[0].dtype)

    def value_infer(self, intensors: list[Any], outtensors: list[Any]) -> None:
        value = intensors[0].get_numpy().copy()
        axes = resolve_squeeze_axes(list(value.shape), self, intensors)
        value = np.squeeze(value, axis=tuple(sorted(axes)))
        outtensors[0].update_tensor(value)


def resolve_squeeze_axes(
    inshape: list[int],
    node_info: Any,
    intensors: list[Any],
) -> set[int]:
    if len(intensors) == 2:
        raw_axes = np.asarray(intensors[1].get_numpy()).reshape(-1)
    elif "axes" in node_info.attr:
        raw_axes = np.asarray(node_info.axes).reshape(-1)
    else:
        return {index for index, dimension in enumerate(inshape) if dimension == 1}
    return {
        axis if axis >= 0 else len(inshape) + axis
        for axis in (int(raw_axis) for raw_axis in raw_axes)
    }


class SoftplusNode(PWNode):
    def __init__(self, node_proto: onnx.NodeProto) -> None:
        super().__init__(node_proto)
        self.op_mac = EXP_MACS + ADD_MACS + LOG_MACS
        self.ratio = 1

    def value_infer(self, intensors: list[Any], outtensors: list[Any]) -> None:
        outtensors[0].update_tensor(np.logaddexp(intensors[0].get_numpy(), 0))


class EluNode(PWNode):
    alpha: float

    def __init__(self, node_proto: onnx.NodeProto) -> None:
        super().__init__(node_proto)
        self.op_mac = CMP_MACS + EXP_MACS + ADD_MACS + MUL_MACS
        self.ratio = 1
        self.add_default_value("alpha", 1.0)

    def value_infer(self, intensors: list[Any], outtensors: list[Any]) -> None:
        x = intensors[0].get_numpy()
        outtensors[0].update_tensor(np.where(x >= 0, x, self.alpha * np.expm1(x)))


class SeluNode(PWNode):
    alpha: float
    gamma: float

    def __init__(self, node_proto: onnx.NodeProto) -> None:
        super().__init__(node_proto)
        self.op_mac = CMP_MACS + EXP_MACS + ADD_MACS + MUL_MACS * 2
        self.ratio = 1
        self.add_default_value("alpha", 1.6732631921768188)
        self.add_default_value("gamma", 1.0507010221481323)

    def value_infer(self, intensors: list[Any], outtensors: list[Any]) -> None:
        x = intensors[0].get_numpy()
        negative = self.gamma * self.alpha * np.expm1(x)
        outtensors[0].update_tensor(np.where(x > 0, self.gamma * x, negative))


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
    graph: Any,
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
        *build_node_shape_feature_vector(graph, node_info),
    ]
    assert len(feature_vector) == NODE_FEATURE_DIM
    return feature_vector


def resolve_op_type_category(op_type: str) -> str:
    return OP_TYPE_CATEGORY_BY_RAW_OP.get(op_type, "op_other")


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
    shape = resolve_profile_tensor_shape(tensor_info)
    element_count = count_elements(shape)
    byte_count = element_count * resolve_profile_tensor_itemsize(tensor_info)
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
        resolve_profile_tensor_shape(graph.tensormap[output_names[0]])
        if output_names
        else ()
    )
    output_itemsize = (
        resolve_profile_tensor_itemsize(graph.tensormap[output_names[0]])
        if output_names
        else 0
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
        float(resolve_profile_tensor_itemsize(tensor_info)),
        float(not shape),
        float(any(dimension == 0 for dimension in shape)),
    ]


def build_graph_shape_feature_vector(
    graph: Any,
    *,
    node_infos: list[Any],
    raw_edges: list[tuple[int, int, str]],
    runtime_inputs: dict[str, np.ndarray],
) -> list[float]:
    activation_tensor_names = {
        tensor_name
        for node_info in node_infos
        for tensor_name in node_info.output
        if tensor_name
    }
    tensor_shapes = [
        resolve_profile_tensor_shape(tensor_info)
        for tensor_info in graph.tensormap.values()
    ]
    runtime_input_shapes = [
        tuple(int(dimension) for dimension in value.shape)
        for value in runtime_inputs.values()
    ]
    return [
        safe_log1p(len(node_infos)),
        safe_log1p(len(raw_edges)),
        safe_log1p(sum_tensor_bytes(graph, activation_tensor_names)),
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


def build_shape_dimension_logs(shape: tuple[int, ...]) -> list[float]:
    dimensions = list(shape[:MAX_SHAPE_RANK])
    dimensions.extend([0] * (MAX_SHAPE_RANK - len(dimensions)))
    return [safe_log1p(dimension) for dimension in dimensions]


def sum_tensor_bytes(graph: Any, tensor_names: Iterable[str]) -> int:
    total = 0
    for tensor_name in tensor_names:
        tensor_info = graph.tensormap[tensor_name]
        total += count_elements(resolve_profile_tensor_shape(tensor_info)) * (
            resolve_profile_tensor_itemsize(tensor_info)
        )
    return total


def sum_tensor_elements(graph: Any, tensor_names: Iterable[str]) -> int:
    return sum(
        count_elements(resolve_profile_tensor_shape(graph.tensormap[tensor_name]))
        for tensor_name in tensor_names
    )


def sum_nonbatch_elements(graph: Any, tensor_names: Iterable[str]) -> int:
    return sum(
        count_nonbatch_elements(
            resolve_profile_tensor_shape(graph.tensormap[tensor_name])
        )
        for tensor_name in tensor_names
    )


def max_tensor_rank(graph: Any, tensor_names: Iterable[str]) -> int:
    return max(
        (
            len(resolve_profile_tensor_shape(graph.tensormap[tensor_name]))
            for tensor_name in tensor_names
        ),
        default=0,
    )


def build_graph_feature_vector(
    *,
    phase: str,
    batch_size: int,
    sample_count: int,
    gpu_name: str,
    parameter_input_stats: dict[str, float],
    graph: Any,
    node_infos: list[Any],
    raw_edges: list[tuple[int, int, str]],
    runtime_inputs: dict[str, np.ndarray],
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
        *build_graph_shape_feature_vector(
            graph,
            node_infos=node_infos,
            raw_edges=raw_edges,
            runtime_inputs=runtime_inputs,
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


def resolve_profile_tensor_shape(tensor_info: object) -> tuple[int, ...]:
    shape = getattr(tensor_info, "shape", None)
    assert shape is not None
    if isinstance(shape, int):
        assert shape >= 0, shape
        return (shape,)
    parsed_shape = tuple(int(dimension) for dimension in shape)
    assert all(dimension >= 0 for dimension in parsed_shape), shape
    return parsed_shape


def resolve_profile_tensor_itemsize(tensor_info: object) -> int:
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


def sum_profile_values(values: object) -> float:
    if isinstance(values, (list, tuple)):
        total = 0.0
        for value in values:
            assert isinstance(value, (int, float, np.number)), type(value)
            total += float(value)
        return total
    assert isinstance(values, (int, float, np.number)), type(values)
    return float(values)
