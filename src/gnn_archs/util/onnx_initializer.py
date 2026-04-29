from __future__ import annotations

from common import onnx_initializer as _shared

# ETL imports common directly; this module only shields legacy gnn_archs import paths.
ONNX_EXPORT_MODE_METADATA_KEY = _shared.ONNX_EXPORT_MODE_METADATA_KEY
RUNTIME_INPUT_NAMES_METADATA_KEY = _shared.RUNTIME_INPUT_NAMES_METADATA_KEY
# Randomization fabricates weights, so these aliases are limited to legacy export tests.
build_random_tensor = _shared.build_random_tensor
get_model_metadata_value = _shared.get_model_metadata_value
load_runtime_input_names = _shared.load_runtime_input_names
randomize_parameter_inputs = _shared.randomize_parameter_inputs
resolve_static_tensor_shape = _shared.resolve_static_tensor_shape
set_model_metadata_value = _shared.set_model_metadata_value
write_randomized_onnx_model = _shared.write_randomized_onnx_model
