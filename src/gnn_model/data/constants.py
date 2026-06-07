from __future__ import annotations

MAX_SHAPE_RANK = 6

GRAPH_ONNX_TOOL_METRIC_FEATURE_NAMES = (
    "graph_macs",
    "graph_memory_bytes",
    "graph_params",
)

OP_TYPE_NAMES = (
    "op_conv",
    "op_dense",
    "op_embedding",
    "op_attention",
    "op_norm",
    "op_pool",
    "op_activation",
    "op_elementwise",
    "op_reduce",
    "op_shape",
    "op_layout",
    "op_join_split",
    "op_cast",
    "op_constant",
    "op_other",
)
OP_TYPE_TO_INDEX = {name: index for index, name in enumerate(OP_TYPE_NAMES)}
OP_TYPE_COUNT = len(OP_TYPE_NAMES)
OP_TYPE_EMBEDDING_DIM = 8

NODE_SHAPE_FEATURE_NAMES = (
    "input_tensor_bytes_sum_log",
    "output_tensor_bytes_sum_log",
    "input_tensor_elements_sum_log",
    "output_tensor_elements_sum_log",
    "input_tensor_rank_max",
    "output_tensor_rank_max",
    "output_tensor_nonbatch_elements_log",
    *(f"output_tensor_dim{index}_log" for index in range(MAX_SHAPE_RANK)),
    "output_tensor_dtype_itemsize",
)

EDGE_SHAPE_FEATURE_NAMES = (
    *(f"tensor_dim{index}_log" for index in range(MAX_SHAPE_RANK)),
    "tensor_nonbatch_element_count_log",
    "tensor_dtype_itemsize",
    "tensor_is_scalar",
    "tensor_is_zero_sized",
)

GRAPH_SHAPE_FEATURE_NAMES = (
    "node_count_log",
    "edge_count_log",
    "activation_bytes_sum_log",
    "peak_live_activation_bytes_log",
    "activation_elements_sum_log",
    "max_tensor_rank",
    "max_tensor_dim_log",
    "runtime_input_count",
    "runtime_input_elements_sum_log",
    "runtime_input_nonbatch_elements_sum_log",
)

NODE_FEATURE_NAMES = (
    "node_macs",
    "node_memory_bytes",
    "node_params",
    "input_count",
    "output_count",
    "attr_count",
    "in_degree",
    "out_degree",
    *NODE_SHAPE_FEATURE_NAMES,
)

EDGE_FEATURE_NAMES = (
    "tensor_bytes",
    "tensor_rank",
    "tensor_element_count",
    "source_out_degree",
    "target_in_degree",
    *EDGE_SHAPE_FEATURE_NAMES,
)

GPU_SPEC_FIELDS = (
    "fp64_peak_tflops",
    "fp32_peak_tflops",
    "tensor_peak_tflops",
    "memory_size_gb",
    "memory_bandwidth_gbs",
    "l2_cache_mb",
    "sm_count",
    "cuda_cores",
    "tdp_watts",
    "nvlink_bandwidth_gbs",
    "pcie_lanes",
)

GRAPH_FEATURE_NAMES = (
    "phase_token_id",
    "batch_size",
    "sample_count",
    *GPU_SPEC_FIELDS,
    "parameter_input_count",
    "parameter_input_element_count",
    "parameter_input_bytes",
    *GRAPH_ONNX_TOOL_METRIC_FEATURE_NAMES,
    *GRAPH_SHAPE_FEATURE_NAMES,
)

GPU_SPECS = {
    "a100": (
        9.7,
        19.5,
        312.0,
        80.0,
        1935.0,
        40.0,
        108.0,
        6912.0,
        300.0,
        600.0,
        16.0,
    ),
    "v100": (
        7.8,
        15.7,
        125.0,
        32.0,
        900.0,
        6.0,
        80.0,
        5120.0,
        300.0,
        300.0,
        16.0,
    ),
}

PHASE_TO_INDEX = {
    "training": 0,
    "inference": 1,
    "prefill": 2,
}


def normalize_gpu_name(value: object) -> str:
    normalized = str(value).strip().lower()
    if "a100" in normalized:
        return "a100"
    if "v100" in normalized:
        return "v100"
    raise AssertionError(f"unsupported gpu_name: {value}")


NODE_FEATURE_DIM = len(NODE_FEATURE_NAMES)
EDGE_FEATURE_DIM = len(EDGE_FEATURE_NAMES)
GRAPH_FEATURE_DIM = len(GRAPH_FEATURE_NAMES)
