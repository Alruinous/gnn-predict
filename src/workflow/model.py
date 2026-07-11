from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Literal

import torch
from torch_geometric.data import Data

from common import get_logger
from common.onnx_initializer import ONNX_OPSET_VERSION
from gnn_archs.causal_lm_builder import (
    CausalLMDecodeOnnxExport,
    CausalLMPrefillOnnxExport,
    build_causal_lm_kv_input_names,
    build_causal_lm_present_output_names,
    collect_causal_lm_kv_pairs,
    run_causal_lm_dry_prefill,
)
from gnn_archs.variant_runner import (
    temporary_causal_lm_export_mode,
)
from gnn_model.data.onnx_graph import build_graph_data_from_onnx


def build_model_input(
    batch_size: int,
    sequence_length: int,
    vocab_size: int,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    LLM 模型需要的输入是 input_ids 和 attention_mask
    """
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
    output_names: list[str],
    phase: Literal["prefill", "decode"] = "decode",
    # INFO: src/archs decode 生成数据集覆盖完整 prefill+decode (一次完整 generate).
    # 传入 decode 即考虑 LLM 完整推理过程. 分阶段修复暂不考虑.
    gpu_name: Literal["v100", "a100"] = "v100",
    batch_size: int = 1,
    decode_output_length: int = 0,
) -> Data:
    """
    LLM 推理分 prefill/decode 两阶段. phase-aware cached ONNX 导出:
    prefill 写预填充图, decode 用末步 KV shape (S+O-1) 单步 cached forward 图.
    """
    logger = get_logger("build_model_graph_feature")
    model = model.eval()
    sequence_length = input_map["input_ids"].shape[-1]
    export_mode = "architecture_only"
    with temporary_causal_lm_export_mode(model), TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        tmp_path.mkdir(parents=True, exist_ok=True)
        onnx_path = tmp_path / f"{model_name}_{phase}.onnx"
        runtime_input_names, _output_name_list = export_phase_cached_onnx(
            model=model,
            onnx_path=onnx_path,
            input_map=input_map,
            input_names=input_names,
            output_names=output_names,
            phase=phase,
            sequence_length=sequence_length,
            decode_output_length=decode_output_length,
            export_mode=export_mode,
        )
        feature = build_graph_data_from_onnx(
            onnx_path,
            batch_size=batch_size,
            runtime_input_names=runtime_input_names,
            phase=phase,
            gpu_name=gpu_name,
            decode_output_length=decode_output_length,
        )

    logger.info(f"{feature}")
    return feature


def export_phase_cached_onnx(
    *,
    model: torch.nn.Module,
    onnx_path: Path,
    input_map: dict[str, Any],
    input_names: list[str],
    output_names: list[str],
    phase: Literal["prefill", "decode"],
    sequence_length: int,
    decode_output_length: int,
    export_mode: str,
) -> tuple[list[str], list[str]]:
    if phase == "prefill":
        prefill_input_names = input_names or ["input_ids", "attention_mask"]
        prefill_args = tuple(input_map[name] for name in prefill_input_names)
        layer_count = resolve_layer_count(model)
        prefill_output_names = build_causal_lm_present_output_names(layer_count)
        export_wrapper = CausalLMPrefillOnnxExport(model).eval()
        torch.onnx.export(
            export_wrapper,
            prefill_args,
            onnx_path,
            input_names=prefill_input_names,
            output_names=prefill_output_names,
            opset_version=ONNX_OPSET_VERSION,
            dynamo=False,
            export_params=export_mode == "full",
        )
        return prefill_input_names, prefill_output_names

    if phase == "decode":
        batch_size = input_map["input_ids"].shape[0]
        past_len = sequence_length + decode_output_length - 1
        generator = torch.Generator().manual_seed(42)
        from gnn_archs.variant_runner import build_text_input_ids

        dry_input_ids = build_text_input_ids(
            model,
            batch_size,
            past_len,
            generator,
        )
        dry_attention_mask = torch.ones(
            (batch_size, past_len),
            dtype=torch.long,
        )
        past_cache = run_causal_lm_dry_prefill(model, dry_input_ids, dry_attention_mask)
        past_kv_pairs = collect_causal_lm_kv_pairs(past_cache)
        layer_count = len(past_kv_pairs)
        past_kv_flat = tuple(
            tensor.contiguous() for pair in past_kv_pairs for tensor in pair
        )
        decode_input_ids = torch.zeros((batch_size, 1), dtype=torch.long)
        decode_attention_mask = torch.ones(
            (batch_size, sequence_length + decode_output_length),
            dtype=torch.long,
        )
        runtime_input_names = (
            input_names or ["input_ids", "attention_mask"]
        ) + build_causal_lm_kv_input_names(layer_count)
        runtime_args = (
            decode_input_ids,
            decode_attention_mask,
            *past_kv_flat,
        )
        if output_names:
            decode_output_names = output_names
        else:
            decode_output_names = build_causal_lm_present_output_names(layer_count)
        export_wrapper = CausalLMDecodeOnnxExport(model).eval()
        torch.onnx.export(
            export_wrapper,
            runtime_args,
            onnx_path,
            input_names=runtime_input_names,
            output_names=decode_output_names,
            opset_version=ONNX_OPSET_VERSION,
            dynamo=False,
            export_params=export_mode == "full",
        )
        return runtime_input_names, decode_output_names

    raise ValueError(f"unsupported causal LM phase: {phase}")


def resolve_layer_count(model: torch.nn.Module) -> int:
    from gnn_archs.variant_runner import resolve_causal_lm_layer_count

    return resolve_causal_lm_layer_count(model)
