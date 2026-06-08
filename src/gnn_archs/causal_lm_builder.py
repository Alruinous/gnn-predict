from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, Qwen3Config

from gnn_archs.config import get_causal_lm_family

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
        return self.model(input_ids=input_ids, attention_mask=attention_mask).logits


def build_causal_lm_variant_model(spec: ResolvedVariantSpec) -> nn.Module:
    validate_causal_lm_variant_spec(spec)
    if spec.variant_config.qwen3_config is not None:
        model = build_qwen3_config_model(spec)
        disable_causal_lm_cache(model)
        return model

    model_path = resolve_causal_lm_model_path(spec.base_model.name)
    family = get_causal_lm_family(spec.base_model.name)
    if not model_path.is_dir():
        raise FileNotFoundError(
            f"{family} checkpoint directory does not exist: {model_path}"
        )

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.float16,
        local_files_only=True,
    )
    disable_causal_lm_cache(model)
    return model


def build_qwen3_config_model(spec: ResolvedVariantSpec) -> nn.Module:
    qwen3_config = spec.variant_config.qwen3_config
    if qwen3_config is None:
        raise ValueError("qwen variants require variant_config.qwen3_config")

    config = Qwen3Config(
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=None,
        use_cache=False,
        **qwen3_config.model_dump(mode="python"),
    )
    validate_qwen3_runtime_config(spec, config)
    return AutoModelForCausalLM.from_config(config, dtype=torch.float16)


def validate_causal_lm_variant_spec(spec: ResolvedVariantSpec) -> None:
    family = get_causal_lm_family(spec.base_model.name)
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
        if spec.variant_config.qwen3_config is None:
            raise ValueError("qwen variants require variant_config.qwen3_config")
        return
    if spec.variant_config.qwen3_config is not None:
        raise ValueError(f"{family} variants must omit qwen3_config")


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


def resolve_causal_lm_model_path(model_name: str) -> Path:
    family = get_causal_lm_family(model_name)
    if family not in CAUSAL_LM_MODEL_ROOTS:
        raise ValueError(f"{family} variants require variant_config.qwen3_config")
    checkpoint_name = model_name.strip().split("/")[-1]
    if not checkpoint_name:
        raise ValueError(f"{family} model name must not be empty")
    return CAUSAL_LM_MODEL_ROOTS[family] / checkpoint_name


def disable_causal_lm_cache(model: nn.Module) -> None:
    config = getattr(model, "config", None)
    if config is not None:
        config.use_cache = False
        text_config = getattr(config, "text_config", None)
        if text_config is not None:
            text_config.use_cache = False
    generation_config = getattr(model, "generation_config", None)
    if generation_config is not None:
        generation_config.use_cache = False
