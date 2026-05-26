from __future__ import annotations

import math
from collections import deque
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any, cast

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
    RUNTIME_PROFILE_FEATURE_NAMES,
    normalize_gpu_name,
)
from gnn_model.data.variant_context import build_variant_context_feature_vector

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

PROFILE_TOP_EVENT_CATEGORIES = (
    "conv",
    "matmul",
    "copy",
    "activation",
    "norm",
    "memory",
    "kernel",
    "other",
)


def build_graph_data_from_onnx(
    onnx_path: str | Path,
    *,
    batch_size: int = 1,
    gpu_name: str = "v100",
    phase: str = "training",
    sample_count: int = 1,
    profile_summary: dict[str, object] | None = None,
    model_name: str | None = None,
    variant_name: str | None = None,
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
    graph_output_names = build_graph_output_name_set(model)

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
        graph_output_names=graph_output_names,
        profile_summary=profile_summary,
        model_name=model_name,
        variant_name=variant_name or model_path.stem,
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


def build_graph_output_name_set(model: onnx.ModelProto) -> set[str]:
    return {output.name for output in model.graph.output if output.name}


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
    graph_output_names: set[str],
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
    return count_elements(resolve_profile_tensor_shape(tensor_info)) * (
        resolve_profile_tensor_itemsize(tensor_info)
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
    graph_output_names: set[str],
    profile_summary: dict[str, object] | None,
    model_name: str | None,
    variant_name: str | None,
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
            graph_output_names=graph_output_names,
        ),
        *build_runtime_profile_feature_vector(profile_summary),
        *build_variant_context_feature_vector(
            model_name=model_name or "",
            variant_name=variant_name or "",
        ),
    ]
    assert len(feature_vector) == GRAPH_FEATURE_DIM
    return feature_vector


def build_runtime_profile_feature_vector(
    profile_summary: dict[str, object] | None,
) -> list[float]:
    if not profile_summary:
        return [0.0] * len(RUNTIME_PROFILE_FEATURE_NAMES)

    def value(name: str) -> float:
        raw_value = profile_summary.get(name, 0.0)
        if isinstance(raw_value, bool):
            return float(raw_value)
        if isinstance(raw_value, (int, float, np.number)):
            parsed = float(raw_value)
            return parsed if math.isfinite(parsed) else 0.0
        return 0.0

    top_events = profile_summary.get("top_events", [])
    top_device_times: list[float] = []
    top_event_features = build_top_event_feature_vector(
        top_events,
        total_device_time_us=max(value("total_device_time_us"), 1.0),
    )
    if isinstance(top_events, list):
        for event in top_events[:3]:
            if isinstance(event, dict):
                raw_time = event.get("device_time_us", 0.0)
                if isinstance(raw_time, (int, float, np.number)):
                    top_device_times.append(max(float(raw_time), 0.0))
    while len(top_device_times) < 3:
        top_device_times.append(0.0)

    return [
        1.0,
        value("profiled_steps"),
        safe_log1p(max(value("wall_time_sec"), 0.0)),
        safe_log1p(max(value("event_count"), 0.0)),
        safe_log1p(max(value("op_count"), 0.0)),
        safe_log1p(max(value("launch_event_count"), 0.0)),
        safe_log1p(max(value("kernel_event_count"), 0.0)),
        safe_log1p(max(value("total_count"), 0.0)),
        safe_log1p(max(value("total_cpu_time_us"), 0.0)),
        safe_log1p(max(value("total_self_cpu_time_us"), 0.0)),
        safe_log1p(max(value("total_device_time_us"), 0.0)),
        safe_log1p(max(value("total_self_device_time_us"), 0.0)),
        safe_log1p(max(value("total_device_memory_pos"), 0.0)),
        safe_log1p(max(value("max_event_device_memory"), 0.0)),
        safe_log1p(max(value("peak_device_memory"), 0.0)),
        safe_log1p(max(value("total_flops"), 0.0)),
        value("conv_device_time_share"),
        value("matmul_device_time_share"),
        value("copy_device_time_share"),
        value("activation_device_time_share"),
        value("norm_device_time_share"),
        safe_log1p(top_device_times[0]),
        safe_log1p(top_device_times[1]),
        safe_log1p(top_device_times[2]),
        *top_event_features,
    ]


def build_top_event_feature_vector(
    top_events: object,
    *,
    total_device_time_us: float,
) -> list[float]:
    category_times = {category: 0.0 for category in PROFILE_TOP_EVENT_CATEGORIES}
    top_times: list[float] = []
    if isinstance(top_events, list):
        for event in top_events:
            if not isinstance(event, Mapping):
                continue
            event_map = cast(Mapping[str, object], event)
            raw_time = event_map.get("device_time_us")
            if not isinstance(raw_time, (int, float, np.number)):
                continue
            device_time = max(float(raw_time), 0.0)
            top_times.append(device_time)
            raw_key = event_map.get("key")
            category = classify_profile_event(str(raw_key) if raw_key else "")
            category_times[category] += device_time
    shares = [
        category_times[category] / total_device_time_us
        for category in PROFILE_TOP_EVENT_CATEGORIES
    ]
    positive_shares = [share for share in shares if share > 0.0]
    entropy = -sum(share * math.log(share) for share in positive_shares)
    return [
        (top_times[0] / total_device_time_us) if top_times else 0.0,
        sum(top_times[:3]) / total_device_time_us,
        entropy,
        max(shares, default=0.0),
        float(len(positive_shares)),
        *shares,
    ]


def classify_profile_event(key: str) -> str:
    lower_key = key.lower()
    if "conv" in lower_key or "cudnn" in lower_key:
        return "conv"
    if any(token in lower_key for token in ("mm", "matmul", "gemm", "bmm")):
        return "matmul"
    if any(token in lower_key for token in ("copy", "memcpy", "memset")):
        return "copy"
    if any(
        token in lower_key
        for token in ("relu", "silu", "gelu", "sigmoid", "softmax")
    ):
        return "activation"
    if "norm" in lower_key:
        return "norm"
    if lower_key.startswith("mem"):
        return "memory"
    if (
        key == "cudaLaunchKernel"
        or key.startswith("void ")
        or key.startswith("ampere_")
        or key.startswith("volta_")
        or key.startswith("maxwell_")
        or key.startswith("cutlass")
    ):
        return "kernel"
    return "other"


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
