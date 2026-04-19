from __future__ import annotations

GRAPH_METRIC_NAMES = (
    "estimated_latency_sec",
    "graph_memory_bytes",
    "total_flops",
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

COMMON_OP_TYPES = (
    "Conv",
    "Relu",
    "Gemm",
    "MatMul",
    "BatchNormalization",
    "Add",
    "Mul",
    "MaxPool",
    "AveragePool",
    "Reshape",
    "Transpose",
    "Concat",
)

OP_TYPE_TO_INDEX = {op_type: index for index, op_type in enumerate(COMMON_OP_TYPES)}

NODE_FEATURE_DIM = len(COMMON_OP_TYPES) + 1 + 9
EDGE_FEATURE_DIM = 8
GRAPH_FEATURE_DIM = 12 + len(GPU_SPEC_FIELDS) + 1
GRAPH_METRIC_DIM = len(GRAPH_METRIC_NAMES)
