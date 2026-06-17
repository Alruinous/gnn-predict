from __future__ import annotations

import json
from pathlib import Path

import onnx
import onnx_tool
from onnx import TensorProto, helper

from common.onnx_initializer import RUNTIME_INPUT_NAMES_METADATA_KEY
from common.onnx_initializer import set_model_metadata_value
from gnn_model.data.constants import GRAPH_FEATURE_DIM, NODE_FEATURE_DIM
from gnn_model.data.onnx_graph import build_graph_data_from_onnx
from onnx_support import install_onnx_tool_extensions


def write_bfloat16_constant_model(onnx_path: Path) -> None:
    constant = TensorProto(
        name="constant",
        data_type=TensorProto.BFLOAT16,
        dims=[2],
        raw_data=(0).to_bytes(2, "little") + (16256).to_bytes(2, "little"),
    )
    node = helper.make_node("Constant", [], ["output"], value=constant, name="const")
    graph = helper.make_graph(
        [node],
        "bf16_constant",
        [helper.make_tensor_value_info("input_ids", TensorProto.INT64, [1])],
        [helper.make_tensor_value_info("output", TensorProto.BFLOAT16, [2])],
    )
    model = helper.make_model(
        graph,
        opset_imports=[helper.make_operatorsetid("", 17)],
    )
    set_model_metadata_value(
        model,
        RUNTIME_INPUT_NAMES_METADATA_KEY,
        json.dumps(["input_ids"]),
    )
    onnx.checker.check_model(model)
    onnx.save(model, onnx_path)


def test_install_onnx_tool_extensions_handles_bfloat16_constant(
    tmp_path: Path,
) -> None:
    onnx_path = tmp_path / "bf16_constant.onnx"
    write_bfloat16_constant_model(onnx_path)

    install_onnx_tool_extensions()
    install_onnx_tool_extensions()
    tool_model = onnx_tool.loadmodel(str(onnx_path))

    assert tool_model.graph.tensormap["output"].get_shape() == [2]
    assert tool_model.graph.tensormap["output"].get_elementsize() == 2


def test_build_graph_data_handles_bfloat16_constant(tmp_path: Path) -> None:
    onnx_path = tmp_path / "bf16_constant.onnx"
    write_bfloat16_constant_model(onnx_path)

    data = build_graph_data_from_onnx(onnx_path)
    x = data.x
    assert x is not None

    assert x.shape == (1, NODE_FEATURE_DIM)
    assert data.graph_features.shape == (1, GRAPH_FEATURE_DIM)
    assert x[0, -1].item() == 2.0
