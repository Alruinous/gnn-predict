from __future__ import annotations

from typing import Any

import numpy as np
import onnx_tool.graph as onnx_tool_graph
import onnx_tool.node as onnx_tool_node
import onnx_tool.tensor as onnx_tool_tensor
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

import onnx

ONNX_DTYPE_ATTR = "onnxdtype2npdtype"
TENSORPROTO_TO_NDARRAY_ATTR = "tensorproto2ndarray"
GET_ATTRIBUTE_DATA_ATTR = "get_attribute_data"
BFLOAT16_PATCH_FLAG = "_gnn_predict_bfloat16_patch"


def install_onnx_tool_extensions() -> None:
    patch_onnx_tool_bfloat16()
    replace_onnx_tool_node(SqueezeNode)
    for node_class in (SoftplusNode, EluNode, SeluNode, IsNaNNode):
        if NODE_REGISTRY.get(node_class.__name__) is None:
            NODE_REGISTRY.register(node_class)


def patch_onnx_tool_bfloat16() -> None:
    if getattr(onnx_tool_tensor, BFLOAT16_PATCH_FLAG, False):
        return

    original_onnxdtype2npdtype = onnx_tool_tensor.onnxdtype2npdtype
    original_tensorproto2ndarray = onnx_tool_tensor.tensorproto2ndarray
    original_get_attribute_data = onnx_tool_tensor.get_attribute_data

    def onnxdtype2npdtype(data_type: int) -> Any:
        if data_type == onnx.TensorProto.BFLOAT16:
            return onnx.helper.tensor_dtype_to_np_dtype(data_type)
        return original_onnxdtype2npdtype(data_type)

    def tensorproto2ndarray(initial: onnx.TensorProto) -> np.ndarray:
        if initial.data_type != onnx.TensorProto.BFLOAT16:
            return original_tensorproto2ndarray(initial)
        shape = tuple(int(dimension) for dimension in initial.dims)
        dtype = np.dtype(onnx.helper.tensor_dtype_to_np_dtype(initial.data_type))
        if initial.raw_data:
            return np.frombuffer(initial.raw_data, dtype=dtype).reshape(shape)
        if initial.int32_data:
            raw = np.fromiter(initial.int32_data, dtype=np.uint16)
            return raw.view(dtype).reshape(shape)
        return np.zeros(shape, dtype=dtype)

    def get_attribute_data(attribute: onnx.AttributeProto) -> Any:
        if attribute.type == attribute.TENSOR:
            return tensorproto2ndarray(attribute.t)
        return original_get_attribute_data(attribute)

    # onnx_tool 1.0.1 omits BFLOAT16, while Qwen3.5 exports use it in Constant tensors.
    setattr(onnx_tool_tensor, ONNX_DTYPE_ATTR, onnxdtype2npdtype)
    setattr(onnx_tool_tensor, TENSORPROTO_TO_NDARRAY_ATTR, tensorproto2ndarray)
    setattr(onnx_tool_tensor, GET_ATTRIBUTE_DATA_ATTR, get_attribute_data)
    setattr(onnx_tool_node, ONNX_DTYPE_ATTR, onnxdtype2npdtype)
    setattr(onnx_tool_node, GET_ATTRIBUTE_DATA_ATTR, get_attribute_data)
    setattr(onnx_tool_graph, GET_ATTRIBUTE_DATA_ATTR, get_attribute_data)
    setattr(onnx_tool_tensor, BFLOAT16_PATCH_FLAG, True)


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


class IsNaNNode(PWNode):
    def __init__(self, node_proto: onnx.NodeProto) -> None:
        super().__init__(node_proto)
        self.op_mac = CMP_MACS
        self.ratio = 1

    def value_infer(self, intensors: list[Any], outtensors: list[Any]) -> None:
        outtensors[0].update_tensor(np.isnan(intensors[0].get_numpy()))


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
