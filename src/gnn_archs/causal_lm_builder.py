from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
from transformers import (
    AutoModelForCausalLM,
    Gemma4ForCausalLM,
    Gemma4TextConfig,
    Qwen3Config,
)

from gnn_archs.config import get_causal_lm_family, normalize_model_identifier

if TYPE_CHECKING:
    from gnn_archs.config import ResolvedVariantSpec


CAUSAL_LM_MODEL_ROOTS = {
    "llama": Path("/data/Models/unsloth"),
    "gemma": Path("/data/Models/unsloth"),
}


class CausalLMOnnxLogitsExport(nn.Module):
    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
        ).logits


def build_causal_lm_variant_model(spec: ResolvedVariantSpec) -> nn.Module:
    validate_causal_lm_variant_spec(spec)
    if spec.variant_config.qwen3_config is not None:
        return build_qwen3_config_model(spec)
    if spec.variant_config.gemma4_config is not None:
        return build_gemma4_config_model(spec)

    model_path = resolve_causal_lm_model_path(spec.base_model.name)
    family = get_causal_lm_family(spec.base_model.name)
    if not model_path.is_dir():
        raise FileNotFoundError(
            f"{family} checkpoint directory does not exist: {model_path}"
        )

    return AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.float16,
        local_files_only=True,
    )


def build_qwen3_config_model(spec: ResolvedVariantSpec) -> nn.Module:
    qwen3_config = spec.variant_config.qwen3_config
    if qwen3_config is None:
        raise ValueError("qwen variants require variant_config.qwen3_config")

    config = Qwen3Config(
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=None,
        use_cache=True,
        **qwen3_config.model_dump(mode="python"),
    )
    validate_qwen3_runtime_config(spec, config)
    return AutoModelForCausalLM.from_config(config, dtype=torch.float16)


def build_gemma4_config_model(spec: ResolvedVariantSpec) -> nn.Module:
    gemma4_config = spec.variant_config.gemma4_config
    if gemma4_config is None:
        raise ValueError("gemma4 variants require variant_config.gemma4_config")

    config_values = gemma4_config.model_dump(mode="python", exclude_none=True)
    config_values["layer_types"] = config_values.get(
        "layer_types"
    ) or build_gemma4_layer_types(gemma4_config.num_hidden_layers)
    config = Gemma4TextConfig(
        pad_token_id=0,
        bos_token_id=2,
        eos_token_id=1,
        use_cache=True,
        **config_values,
    )
    validate_gemma4_runtime_config(spec, config)
    return Gemma4ForCausalLM(config).to(dtype=torch.float16)


def build_gemma4_layer_types(num_hidden_layers: int) -> list[str]:
    return [
        "full_attention"
        if layer_index == num_hidden_layers - 1 or (layer_index + 1) % 6 == 0
        else "sliding_attention"
        for layer_index in range(num_hidden_layers)
    ]


def validate_causal_lm_variant_spec(spec: ResolvedVariantSpec) -> None:
    family = get_causal_lm_family(spec.base_model.name)
    model_name = normalize_model_identifier(spec.base_model.name)
    if spec.mutations:
        raise ValueError(f"{family} variants do not support mutations")
    if len(spec.variant_config.example_input_shape) != 2:
        raise ValueError(f"{family} variants require flat text example_input_shape")
    if spec.variant_config.target_input_channels is not None:
        raise ValueError(f"{family} variants must omit target_input_channels")
    if spec.variant_config.target_output_classes is not None:
        raise ValueError(f"{family} variants must omit target_output_classes")
    if family == "qwen":
        if spec.base_model.pretrained:
            raise ValueError("qwen3 config variants require pretrained=false")
        if spec.variant_config.gemma4_config is not None:
            raise ValueError("qwen variants must omit gemma4_config")
        if spec.variant_config.qwen3_config is None:
            raise ValueError("qwen variants require variant_config.qwen3_config")
        return
    if model_name == "gemma4":
        if spec.base_model.pretrained:
            raise ValueError("gemma4 config variants require pretrained=false")
        if spec.variant_config.qwen3_config is not None:
            raise ValueError("gemma4 variants must omit qwen3_config")
        if spec.variant_config.gemma4_config is None:
            raise ValueError("gemma4 variants require variant_config.gemma4_config")
        return
    if spec.variant_config.qwen3_config is not None:
        raise ValueError(f"{family} variants must omit qwen3_config")
    if spec.variant_config.gemma4_config is not None:
        raise ValueError(f"{family} variants must omit gemma4_config")


def validate_qwen3_runtime_config(
    spec: ResolvedVariantSpec,
    config: Qwen3Config,
) -> None:
    batch_size, sequence_length = spec.variant_config.example_input_shape
    if spec.variant_config.batch_size != batch_size:
        raise ValueError("qwen variants require batch_size to match input")
    required_length = sequence_length
    if spec.variant_config.run_decode:
        required_length += spec.variant_config.decode_max_output_length
    if required_length > config.max_position_embeddings:
        raise ValueError(
            "qwen sequence and decode length exceed max_position_embeddings"
        )


def validate_gemma4_runtime_config(
    spec: ResolvedVariantSpec,
    config: Gemma4TextConfig,
) -> None:
    batch_size, sequence_length = spec.variant_config.example_input_shape
    if spec.variant_config.batch_size != batch_size:
        raise ValueError("gemma4 variants require batch_size to match input")
    required_length = sequence_length
    if spec.variant_config.run_decode:
        required_length += spec.variant_config.decode_max_output_length
    if required_length > config.max_position_embeddings:
        raise ValueError(
            "gemma4 sequence and decode length exceed max_position_embeddings"
        )


def resolve_causal_lm_model_path(model_name: str) -> Path:
    family = get_causal_lm_family(model_name)
    if family not in CAUSAL_LM_MODEL_ROOTS:
        raise ValueError(f"{family} variants require variant_config.qwen3_config")
    checkpoint_name = model_name.strip().split("/")[-1]
    if not checkpoint_name:
        raise ValueError(f"{family} model name must not be empty")
    return CAUSAL_LM_MODEL_ROOTS[family] / checkpoint_name
