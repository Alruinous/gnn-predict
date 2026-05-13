from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
import torch.nn as nn
from transformers import GPT2Config, GPT2ForSequenceClassification

if TYPE_CHECKING:
    from gnn_archs.config import ResolvedVariantSpec


class Gpt2ForGnnArchsSequenceClassification(nn.Module):
    def __init__(self, config: GPT2Config) -> None:
        super().__init__()
        self.model = GPT2ForSequenceClassification(config)
        self.config = self.model.config
        self.transformer = self.model.transformer

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> Any:
        prepared_mask = build_gpt2_export_attention_mask(input_ids, attention_mask)
        return self.model(
            input_ids=input_ids,
            attention_mask=prepared_mask,
            labels=labels,
        )


def build_gpt2_variant_model(spec: ResolvedVariantSpec) -> nn.Module:
    if spec.mutations:
        raise ValueError("gpt2 variants do not support mutations")
    gpt2_config = spec.variant_config.gpt2_config
    if gpt2_config is None:
        raise ValueError("gpt2 variants require variant_config.gpt2_config")
    if spec.variant_config.target_output_classes is None:
        raise ValueError("gpt2 variants require target_output_classes")

    config = GPT2Config(
        num_labels=spec.variant_config.target_output_classes,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        use_cache=False,
        attn_implementation="eager",
        **gpt2_config.model_dump(),
    )
    validate_gpt2_config(config, spec.variant_config.example_input_shape[1])
    return Gpt2ForGnnArchsSequenceClassification(config)


def build_gpt2_export_attention_mask(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    batch_size, sequence_length = input_ids.shape
    mask_value = torch.finfo(torch.float32).min
    causal = torch.tril(
        torch.ones(
            (sequence_length, sequence_length),
            dtype=torch.bool,
            device=input_ids.device,
        )
    )
    causal_mask = torch.zeros(
        (sequence_length, sequence_length),
        dtype=torch.float32,
        device=input_ids.device,
    ).masked_fill(~causal, mask_value)
    causal_mask = causal_mask.view(
        1,
        1,
        sequence_length,
        sequence_length,
    ).expand(batch_size, 1, sequence_length, sequence_length)
    padding_mask = attention_mask.to(torch.bool).view(batch_size, 1, 1, sequence_length)
    return causal_mask.masked_fill(~padding_mask, mask_value)


def validate_gpt2_config(config: GPT2Config, sequence_length: int) -> None:
    if config.n_embd <= 0:
        raise ValueError("gpt2 n_embd must be positive")
    if config.n_layer <= 0:
        raise ValueError("gpt2 n_layer must be positive")
    if config.n_head <= 0:
        raise ValueError("gpt2 n_head must be positive")
    if config.n_embd % config.n_head != 0:
        raise ValueError("gpt2 n_embd must be divisible by n_head")
    if config.n_inner is not None and config.n_inner <= 0:
        raise ValueError("gpt2 n_inner must be positive when provided")
    if config.n_positions < sequence_length:
        raise ValueError("gpt2 n_positions must cover example_input_shape sequence")
    if config.vocab_size <= max(
        config.bos_token_id,
        config.eos_token_id,
        config.pad_token_id,
    ):
        raise ValueError("gpt2 vocab_size must exceed special token ids")
    for field_name in ("resid_pdrop", "embd_pdrop", "attn_pdrop"):
        value = float(getattr(config, field_name))
        if value < 0 or value >= 1:
            raise ValueError(f"gpt2 {field_name} must be in [0, 1)")
