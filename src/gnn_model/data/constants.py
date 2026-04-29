from __future__ import annotations

GRAPH_PROFILE_FEATURE_NAMES = (
    "profile_total_macs",
    "profile_total_flops",
    "profile_memory_bytes",
    "profile_params",
)

NODE_FEATURE_NAMES = (
    "profile_macs",
    "profile_memory_bytes",
    "profile_params",
    "input_count",
    "output_count",
    "attr_count",
    "in_degree",
    "out_degree",
)

EDGE_FEATURE_NAMES = (
    "tensor_bytes",
    "tensor_rank",
    "tensor_element_count",
    "source_out_degree",
    "target_in_degree",
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
    *GRAPH_PROFILE_FEATURE_NAMES,
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
}

NODE_FEATURE_DIM = len(NODE_FEATURE_NAMES)
EDGE_FEATURE_DIM = len(EDGE_FEATURE_NAMES)
GRAPH_FEATURE_DIM = len(GRAPH_FEATURE_NAMES)
