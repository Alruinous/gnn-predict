"""Analytical size/bandwidth resource predictor (llmfit / LIFE / BestServe family).

Physics-only baseline: weight bytes + GQA KV cache for peak VRAM, a
memory-bandwidth roofline for decode run time, and a bandwidth estimate for
load time. It has zero profiling/training cost and deliberately ignores
framework overhead, activation dynamics, fragmentation, and vLLM KV
preallocation — the systematic gaps the learned GNN predictor is meant to close.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from workflow.artifacts import (
    ResourceContract,
    ResourceContractCache,
    ResourceContractSource,
    ResourceEvidence,
)
from workflow.types import WorkflowModelFeatureKey

BYTES_PER_MIB = 1024 * 1024
BYTES_PER_GIB = 1024 * 1024 * 1024
DTYPE_BYTES = 2  # float16 weights + KV
EFFICIENCY = 0.55  # llmfit memory-bandwidth efficiency factor
CUDA_CONTEXT_MIB = 1200.0  # fixed CUDA/runtime resident footprint
VRAM_MARGIN_FRACTION = 0.10  # matches profile fixed_margin_fallback


@dataclass(frozen=True, slots=True)
class ModelArch:
    params: float  # total parameter count
    num_layers: int
    num_kv_heads: int  # GQA key/value heads
    head_dim: int


@dataclass(frozen=True, slots=True)
class GpuSpec:
    mem_bandwidth_bytes_s: float
    load_bandwidth_bytes_s: float  # host->device weight staging (framework-blind)
    idle_power_w: float
    tdp_w: float


# Public Qwen3 HF config values; build script prefers reading model_path/config.json.
STATIC_ARCH: Mapping[str, ModelArch] = {
    "Qwen3-0.6B": ModelArch(0.6e9, 28, 8, 128),
    "Qwen3-1.7B": ModelArch(1.7e9, 28, 8, 128),
    "Qwen3-4B": ModelArch(4.0e9, 36, 8, 128),
    "Qwen3-8B": ModelArch(8.2e9, 36, 8, 128),
    "Qwen3-14B": ModelArch(14.8e9, 40, 8, 128),
    "Qwen3-32B": ModelArch(32.8e9, 64, 8, 128),
}

STATIC_GPU: Mapping[str, GpuSpec] = {
    "a100": GpuSpec(1935e9, 25e9, 50.0, 400.0),
    "v100": GpuSpec(900e9, 12e9, 40.0, 300.0),
}


def load_arch_table(cache_config_path: Path | None) -> dict[str, ModelArch]:
    """Prefer real HF configs listed in config/workflow/cache.yaml, else fall back."""
    table = dict(STATIC_ARCH)
    if cache_config_path is None or not cache_config_path.exists():
        return table
    config = yaml.safe_load(cache_config_path.read_text(encoding="utf-8"))
    for group in config.get("cache_groups", ()):
        arch = _read_hf_arch(Path(group["model_path"]) / "config.json")
        if arch is not None:
            table[group["model_name"]] = arch
    return table


def _read_hf_arch(config_path: Path) -> ModelArch | None:
    if not config_path.exists():
        return None
    cfg: dict[str, Any] = json.loads(config_path.read_text(encoding="utf-8"))
    hidden = int(cfg["hidden_size"])
    heads = int(cfg["num_attention_heads"])
    head_dim = int(cfg.get("head_dim", hidden // heads))
    return ModelArch(
        params=_estimate_params(cfg, hidden, head_dim, heads),
        num_layers=int(cfg["num_hidden_layers"]),
        num_kv_heads=int(cfg.get("num_key_value_heads", heads)),
        head_dim=head_dim,
    )


def _estimate_params(
    cfg: Mapping[str, Any], hidden: int, head_dim: int, heads: int
) -> float:
    layers = int(cfg["num_hidden_layers"])
    kv_heads = int(cfg.get("num_key_value_heads", heads))
    inter = int(cfg["intermediate_size"])
    vocab = int(cfg["vocab_size"])
    q_dim = heads * head_dim
    kv_dim = kv_heads * head_dim
    attn = hidden * (q_dim + 2 * kv_dim) + q_dim * hidden
    mlp = 3 * hidden * inter  # gate + up + down
    return float(layers * (attn + mlp) + 2 * vocab * hidden)


def _kv_cache_bytes(arch: ModelArch, context_tokens: int, batch_size: int) -> float:
    return (
        2  # key + value
        * arch.num_layers
        * arch.num_kv_heads
        * arch.head_dim
        * DTYPE_BYTES
        * context_tokens
        * batch_size
    )


def _static_contract(
    key: WorkflowModelFeatureKey, arch: ModelArch, gpu: GpuSpec
) -> ResourceContract:
    weight_bytes = arch.params * DTYPE_BYTES
    peak_context = key.sequence_length + key.decode_output_length
    kv_bytes = _kv_cache_bytes(arch, peak_context, key.batch_size)

    peak_vram_mb = (weight_bytes + kv_bytes) / BYTES_PER_MIB + CUDA_CONTEXT_MIB
    bytes_per_token = weight_bytes + kv_bytes  # roofline read per decode step
    run_sec = (
        key.decode_output_length
        * bytes_per_token
        / (gpu.mem_bandwidth_bytes_s * EFFICIENCY)
    )
    load_sec = weight_bytes / gpu.load_bandwidth_bytes_s
    power_w = gpu.idle_power_w + (gpu.tdp_w - gpu.idle_power_w) * EFFICIENCY

    return ResourceContract(
        key=key,
        source=ResourceContractSource.SYNTHETIC_FIXTURE,
        predicted_load_sec=max(load_sec, 1e-6),
        predicted_run_sec=max(run_sec, 1e-6),
        predicted_peak_vram_mb=peak_vram_mb,
        peak_vram_mb_upper_bound=peak_vram_mb * (1.0 + VRAM_MARGIN_FRACTION),
        peak_vram_mb_evidence=ResourceEvidence(
            method="fixed_margin_fallback",
            sample_count=1,
            margin_fraction=VRAM_MARGIN_FRACTION,
        ),
        predicted_power_watts=power_w,
        predictor_metadata={
            "method": "analytical_size_bandwidth",
            "efficiency_factor": EFFICIENCY,
            "family": "llmfit_life_bestserve",
        },
    )


def build_static_prediction_cache(
    keys: Iterable[WorkflowModelFeatureKey],
    arch_table: Mapping[str, ModelArch] = STATIC_ARCH,
    gpu_table: Mapping[str, GpuSpec] = STATIC_GPU,
) -> ResourceContractCache:
    entries: list[ResourceContract] = []
    for key in keys:
        assert key.phase == "decode", f"static model supports decode only: {key.phase}"
        arch = arch_table.get(key.model_name)
        gpu = gpu_table.get(key.gpu_name)
        assert arch is not None, f"no arch spec for {key.model_name}"
        assert gpu is not None, f"no gpu spec for {key.gpu_name}"
        entries.append(_static_contract(key, arch, gpu))
    return ResourceContractCache(
        version=2,
        environment={
            "source": "analytical_size_bandwidth",
            "efficiency_factor": EFFICIENCY,
            "dtype_bytes": DTYPE_BYTES,
            "reference": "llmfit / LIFE (2508.00904) / BestServe (2506.05871)",
        },
        entries=tuple(entries),
    )
