from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM

if TYPE_CHECKING:
    from gnn_archs.config import ResolvedVariantSpec


QWEN_MODEL_ROOT = Path("/data/Models/Qwen")


class QwenOnnxLogitsExport(nn.Module):
    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        return self.model(input_ids=input_ids, attention_mask=attention_mask).logits


def build_qwen_variant_model(spec: ResolvedVariantSpec) -> nn.Module:
    if spec.mutations:
        raise ValueError("qwen variants do not support mutations")
    if len(spec.variant_config.example_input_shape) != 2:
        raise ValueError("qwen variants require flat text example_input_shape")
    if spec.variant_config.target_input_channels is not None:
        raise ValueError("qwen variants must omit target_input_channels")
    if spec.variant_config.target_output_classes is not None:
        raise ValueError("qwen variants must omit target_output_classes")

    model_path = resolve_qwen_model_path(spec.base_model.name)
    if not model_path.is_dir():
        raise FileNotFoundError(
            f"qwen checkpoint directory does not exist: {model_path}"
        )

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.float16,
        local_files_only=True,
    )
    disable_qwen_cache(model)
    return model


def resolve_qwen_model_path(model_name: str) -> Path:
    checkpoint_name = model_name.strip().split("/")[-1]
    if not checkpoint_name:
        raise ValueError("qwen model name must not be empty")
    return QWEN_MODEL_ROOT / checkpoint_name


def disable_qwen_cache(model: nn.Module) -> None:
    config = getattr(model, "config", None)
    if config is not None:
        config.use_cache = False
        text_config = getattr(config, "text_config", None)
        if text_config is not None:
            text_config.use_cache = False
    generation_config = getattr(model, "generation_config", None)
    if generation_config is not None:
        generation_config.use_cache = False
