from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import torch
from torch.export import ExportedProgram
from torch_geometric.data import Data

from common import get_logger
from common.graph_artifact import capture_inference_graph, save_graph_artifact
from gnn_archs.causal_lm_builder import (
    CausalLMDecodeGraph,
    CausalLMPrefillGraph,
    build_causal_lm_kv_input_names,
    collect_causal_lm_kv_pairs,
    run_causal_lm_dry_prefill,
    temporary_causal_lm_graph_mode,
)
from gnn_model.data.fx_graph import build_graph_data_from_exported_program

WorkflowPhase = Literal["prefill", "decode"]


def build_model_input(
    batch_size: int,
    sequence_length: int,
    vocab_size: int,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    input_ids = torch.randint(
        0,
        vocab_size,
        (batch_size, sequence_length),
        generator=generator,
    )
    attention_mask = torch.ones((batch_size, sequence_length), dtype=torch.long)
    return input_ids, attention_mask


def build_model_graph_feature(
    model_name: str,
    model: torch.nn.Module,
    input_map: dict[str, Any],
    input_names: list[str],
    phase: WorkflowPhase = "decode",
    gpu_name: Literal["v100", "a100"] = "v100",
    batch_size: int = 1,
    decode_output_length: int = 0,
) -> Data:
    input_ids = require_tensor(input_map, "input_ids")
    if input_ids.shape[0] != batch_size:
        raise ValueError(
            f"batch_size does not match input_ids: {batch_size} != {input_ids.shape[0]}"
        )
    sequence_length = input_ids.shape[-1]
    exported_program, runtime_input_names = capture_phase_graph(
        model=model,
        input_map=input_map,
        input_names=input_names,
        phase=phase,
        sequence_length=sequence_length,
        decode_output_length=decode_output_length,
    )
    feature = build_graph_data_from_exported_program(
        exported_program,
        runtime_input_names=runtime_input_names,
        batch_size=batch_size,
        phase=phase,
        gpu_name=gpu_name,
        decode_output_length=decode_output_length,
    )
    get_logger("build_model_graph_feature").info(f"{model_name}: {feature}")
    return feature


def export_phase_cached_graph(
    *,
    model: torch.nn.Module,
    graph_path: Path,
    input_map: dict[str, Any],
    input_names: list[str],
    phase: WorkflowPhase,
    sequence_length: int,
    decode_output_length: int,
) -> list[str]:
    exported_program, runtime_input_names = capture_phase_graph(
        model=model,
        input_map=input_map,
        input_names=input_names,
        phase=phase,
        sequence_length=sequence_length,
        decode_output_length=decode_output_length,
    )
    save_graph_artifact(
        exported_program,
        graph_path,
        runtime_input_names=runtime_input_names,
    )
    return runtime_input_names


def capture_phase_graph(
    *,
    model: torch.nn.Module,
    input_map: dict[str, Any],
    input_names: list[str],
    phase: WorkflowPhase,
    sequence_length: int,
    decode_output_length: int,
) -> tuple[ExportedProgram, list[str]]:
    model.eval()
    device = next(model.parameters()).device
    if phase == "prefill":
        runtime_input_names = input_names or ["input_ids", "attention_mask"]
        args = tuple(
            require_tensor(input_map, name).to(device) for name in runtime_input_names
        )
        wrapper = CausalLMPrefillGraph(model).to(device)
        with temporary_causal_lm_graph_mode(model):
            return capture_inference_graph(wrapper, args), runtime_input_names
    if phase != "decode":
        raise ValueError(f"unsupported causal LM phase: {phase}")
    if decode_output_length <= 0:
        raise ValueError("decode_output_length must be positive for decode graphs")

    input_ids = require_tensor(input_map, "input_ids")
    batch_size = input_ids.shape[0]
    past_length = sequence_length + decode_output_length - 1
    generator = torch.Generator().manual_seed(42)
    from gnn_archs.variant_runner import build_text_input_ids

    dry_input_ids = build_text_input_ids(
        model,
        batch_size,
        past_length,
        generator,
    ).to(device)
    dry_attention_mask = torch.ones(
        (batch_size, past_length),
        dtype=torch.long,
        device=device,
    )
    past_cache = run_causal_lm_dry_prefill(model, dry_input_ids, dry_attention_mask)
    past_kv_pairs = collect_causal_lm_kv_pairs(past_cache)
    if not past_kv_pairs:
        raise ValueError("causal LM dry prefill produced no KV cache layers")
    past_kv_flat = tuple(
        tensor.contiguous() for pair in past_kv_pairs for tensor in pair
    )
    decode_input_ids = torch.zeros((batch_size, 1), dtype=torch.long, device=device)
    decode_attention_mask = torch.ones(
        (batch_size, sequence_length + decode_output_length),
        dtype=torch.long,
        device=device,
    )
    runtime_input_names = [
        "input_ids",
        "attention_mask",
        *build_causal_lm_kv_input_names(len(past_kv_pairs)),
    ]
    wrapper = CausalLMDecodeGraph(model).to(device)
    with temporary_causal_lm_graph_mode(model):
        exported_program = capture_inference_graph(
            wrapper,
            (decode_input_ids, decode_attention_mask, *past_kv_flat),
        )
    return exported_program, runtime_input_names


def require_tensor(input_map: dict[str, Any], name: str) -> torch.Tensor:
    value = input_map[name]
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"model input must be a tensor: {name}")
    return value
