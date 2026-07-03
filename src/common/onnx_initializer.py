from __future__ import annotations

import json
from collections.abc import Sequence
from copy import deepcopy
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper

RUNTIME_INPUT_NAMES_METADATA_KEY = "gnn_archs.runtime_input_names"
ONNX_EXPORT_MODE_METADATA_KEY = "gnn_archs.onnx_export_mode"
ONNX_OPSET_VERSION = 14


def set_model_metadata_value(model: onnx.ModelProto, key: str, value: str) -> None:
    for entry in model.metadata_props:
        if entry.key == key:
            entry.value = value
            return
    entry = model.metadata_props.add()
    entry.key = key
    entry.value = value


def get_model_metadata_value(model: onnx.ModelProto, key: str) -> str | None:
    for entry in model.metadata_props:
        if entry.key == key:
            return entry.value
    return None


def load_runtime_input_names(model: onnx.ModelProto) -> list[str]:
    raw_names = get_model_metadata_value(model, RUNTIME_INPUT_NAMES_METADATA_KEY)
    if raw_names is None:
        raise ValueError(
            "ONNX model is missing gnn_archs runtime input metadata needed to "
            "separate real inputs from parameter inputs"
        )

    try:
        runtime_input_names = json.loads(raw_names)
    except json.JSONDecodeError as exc:
        raise ValueError("runtime input metadata is not valid JSON") from exc

    if (
        not isinstance(runtime_input_names, list)
        or not runtime_input_names
        or any(not isinstance(name, str) or not name for name in runtime_input_names)
    ):
        raise ValueError("runtime input metadata must be a non-empty list of strings")
    return runtime_input_names


def randomize_parameter_inputs(
    model: onnx.ModelProto,
    *,
    runtime_input_names: Sequence[str] | None = None,
    seed: int | None = None,
    std: float = 0.02,
) -> onnx.ModelProto:
    resolved_runtime_input_names = (
        list(runtime_input_names)
        if runtime_input_names is not None
        else load_runtime_input_names(model)
    )
    runtime_input_name_set = set(resolved_runtime_input_names)
    if len(runtime_input_name_set) != len(resolved_runtime_input_names):
        raise ValueError("runtime_input_names must not contain duplicates")

    randomized_model = deepcopy(model)
    graph = randomized_model.graph
    graph_input_names = {value.name for value in graph.input}
    missing_runtime_inputs = runtime_input_name_set - graph_input_names
    if missing_runtime_inputs:
        raise ValueError(
            "runtime_input_names are not present in the ONNX graph inputs: "
            f"{sorted(missing_runtime_inputs)}"
        )

    parameter_inputs = [
        deepcopy(value)
        for value in graph.input
        if value.name not in runtime_input_name_set
    ]
    if not parameter_inputs:
        raise ValueError("ONNX model does not expose any parameter inputs to randomize")

    initializer_names = {initializer.name for initializer in graph.initializer}
    duplicate_initializers = initializer_names & {
        value.name for value in parameter_inputs
    }
    if duplicate_initializers:
        raise ValueError(
            "ONNX model already contains initializers for parameter inputs: "
            f"{sorted(duplicate_initializers)}"
        )

    generator = np.random.default_rng(seed)
    for value_info in parameter_inputs:
        random_tensor = build_random_tensor(value_info, generator, std)
        graph.initializer.append(
            numpy_helper.from_array(random_tensor, value_info.name)
        )

    remaining_inputs = [
        deepcopy(value) for value in graph.input if value.name in runtime_input_name_set
    ]
    graph.ClearField("input")
    graph.input.extend(remaining_inputs)
    set_model_metadata_value(
        randomized_model,
        RUNTIME_INPUT_NAMES_METADATA_KEY,
        json.dumps(resolved_runtime_input_names),
    )
    set_model_metadata_value(
        randomized_model,
        ONNX_EXPORT_MODE_METADATA_KEY,
        "full",
    )
    onnx.checker.check_model(randomized_model)
    return randomized_model


def write_randomized_onnx_model(
    source_path: Path,
    output_path: Path,
    *,
    runtime_input_names: Sequence[str] | None = None,
    seed: int | None = None,
    std: float = 0.02,
) -> Path:
    source_model = onnx.load(source_path)
    randomized_model = randomize_parameter_inputs(
        source_model,
        runtime_input_names=runtime_input_names,
        seed=seed,
        std=std,
    )
    onnx.save(randomized_model, output_path)
    return output_path


def build_random_tensor(
    value_info: onnx.ValueInfoProto,
    generator: np.random.Generator,
    std: float,
) -> np.ndarray:
    tensor_type = value_info.type.tensor_type
    if tensor_type.elem_type == onnx.TensorProto.UNDEFINED:
        raise ValueError(f"parameter input '{value_info.name}' is missing elem_type")

    dtype = np.dtype(onnx.helper.tensor_dtype_to_np_dtype(tensor_type.elem_type))
    shape = resolve_static_tensor_shape(value_info)

    if np.issubdtype(dtype, np.floating):
        return generator.normal(loc=0.0, scale=std, size=shape).astype(dtype)
    if np.issubdtype(dtype, np.signedinteger):
        return generator.integers(-2, 3, size=shape, dtype=dtype)
    if np.issubdtype(dtype, np.unsignedinteger):
        return generator.integers(0, 5, size=shape, dtype=dtype)
    if dtype == np.bool_:
        return generator.integers(0, 2, size=shape, dtype=np.int64).astype(np.bool_)

    raise ValueError(
        f"parameter input '{value_info.name}' uses unsupported dtype '{dtype}'"
    )


def resolve_static_tensor_shape(value_info: onnx.ValueInfoProto) -> tuple[int, ...]:
    dims: list[int] = []
    for dimension in value_info.type.tensor_type.shape.dim:
        if dimension.HasField("dim_value") and dimension.dim_value > 0:
            dims.append(dimension.dim_value)
            continue
        raise ValueError(
            "parameter input must have static positive dimensions for random "
            f"initialization: {value_info.name}"
        )
    return tuple(dims)
