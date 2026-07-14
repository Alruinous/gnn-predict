from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch
import torch.nn as nn
from transformers import T5Config, T5ForSequenceClassification

if TYPE_CHECKING:
    from gnn_archs.config import ResolvedVariantSpec


class T5SequenceClassificationGraph(nn.Module):
    def __init__(self, model: T5ForSequenceClassification) -> None:
        super().__init__()
        self.model = model

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        decoder_input_ids = self.model._shift_right(input_ids)
        outputs = self.model.transformer(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            use_cache=False,
            return_dict=False,
        )
        sequence_output = cast(torch.Tensor, outputs[0])
        sentence_representation = sequence_output[:, -1, :]  # [B,S,H]->[B,H]
        return self.model.classification_head(sentence_representation)


def build_t5_variant_model(spec: ResolvedVariantSpec) -> nn.Module:
    if spec.mutations:
        raise ValueError("t5 variants do not support mutations")
    t5_config = spec.variant_config.t5_config
    if t5_config is None:
        raise ValueError("t5 variants require variant_config.t5_config")
    if spec.variant_config.target_output_classes is None:
        raise ValueError("t5 variants require target_output_classes")
    if len(spec.variant_config.example_input_shape) != 2:
        raise ValueError("t5 variants require flat text example_input_shape")

    config_values = t5_config.model_dump()
    if config_values["d_kv"] is None:
        d_model = config_values["d_model"]
        num_heads = config_values["num_heads"]
        if d_model <= 0:
            raise ValueError("t5 d_model must be positive")
        if num_heads <= 0:
            raise ValueError("t5 num_heads must be positive")
        if d_model % num_heads != 0:
            raise ValueError(
                "t5 d_model must be divisible by num_heads when d_kv is omitted"
            )
        config_values["d_kv"] = d_model // num_heads

    config = T5Config(
        num_labels=spec.variant_config.target_output_classes,
        pad_token_id=0,
        eos_token_id=1,
        decoder_start_token_id=0,
        use_cache=False,
        **config_values,
    )
    validate_t5_config(config)
    return T5ForSequenceClassification(config)


def validate_t5_config(config: T5Config) -> None:
    if config.d_model <= 0:
        raise ValueError("t5 d_model must be positive")
    if config.d_kv <= 0:
        raise ValueError("t5 d_kv must be positive")
    if config.d_ff <= 0:
        raise ValueError("t5 d_ff must be positive")
    if config.num_layers <= 0:
        raise ValueError("t5 num_layers must be positive")
    if config.num_decoder_layers <= 0:
        raise ValueError("t5 num_decoder_layers must be positive")
    if config.num_heads <= 0:
        raise ValueError("t5 num_heads must be positive")
    if config.relative_attention_num_buckets <= 0:
        raise ValueError("t5 relative_attention_num_buckets must be positive")
    if config.relative_attention_max_distance <= 0:
        raise ValueError("t5 relative_attention_max_distance must be positive")
    pad_token_id = require_int_config_value(config.pad_token_id, "pad_token_id")
    eos_token_id = require_int_config_value(config.eos_token_id, "eos_token_id")
    decoder_start_token_id = require_int_config_value(
        config.decoder_start_token_id,
        "decoder_start_token_id",
    )
    if config.vocab_size <= max(pad_token_id, eos_token_id, decoder_start_token_id):
        raise ValueError("t5 vocab_size must exceed special token ids")
    if config.layer_norm_epsilon <= 0:
        raise ValueError("t5 layer_norm_epsilon must be positive")
    if config.initializer_factor <= 0:
        raise ValueError("t5 initializer_factor must be positive")
    if config.feed_forward_proj not in {"relu", "gated-gelu"}:
        raise ValueError("t5 feed_forward_proj must be relu or gated-gelu")
    for field_name in ("dropout_rate", "classifier_dropout"):
        value = float(getattr(config, field_name))
        if value < 0 or value >= 1:
            raise ValueError(f"t5 {field_name} must be in [0, 1)")


def require_int_config_value(value: object, field_name: str) -> int:
    if not isinstance(value, int):
        raise ValueError(f"t5 {field_name} must be set")
    return value
