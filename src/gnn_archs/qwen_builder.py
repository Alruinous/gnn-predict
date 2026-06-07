from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import torch.nn as nn

from gnn_archs.causal_lm_builder import (
    CausalLMOnnxLogitsExport,
    build_causal_lm_variant_model,
    disable_causal_lm_cache,
    resolve_causal_lm_model_path,
)

if TYPE_CHECKING:
    from gnn_archs.config import ResolvedVariantSpec


QWEN_MODEL_ROOT = Path("/data/Models/Qwen")
QwenOnnxLogitsExport = CausalLMOnnxLogitsExport


def build_qwen_variant_model(spec: ResolvedVariantSpec) -> nn.Module:
    return build_causal_lm_variant_model(spec)


def resolve_qwen_model_path(model_name: str) -> Path:
    return resolve_causal_lm_model_path(model_name)


def disable_qwen_cache(model: nn.Module) -> None:
    disable_causal_lm_cache(model)
