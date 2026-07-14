from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from gnn_archs.config import (
    ArchConfig,
    get_causal_lm_family,
    is_causal_lm_model_name,
    is_text_model_name,
)
from gnn_archs.util.variant_expander import expand_arch_config, expand_group_variants

ROOT = Path(__file__).resolve().parents[1]


def build_qwen_variant_config_grid(
    *,
    axes: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    resolved_axes = (
        axes
        if axes is not None
        else [
            {
                "name": "hidden",
                "values": [
                    {
                        "name": "h16",
                        "overrides": {"qwen3_config.hidden_size": 16},
                    },
                    {
                        "name": "h32",
                        "overrides": {"qwen3_config.hidden_size": 32},
                    },
                ],
            },
            {
                "name": "batch",
                "values": [
                    {
                        "name": "bs2",
                        "overrides": {
                            "example_input_shape": [2, 8],
                            "batch_size": 2,
                        },
                    },
                    {
                        "name": "bs8",
                        "overrides": {
                            "example_input_shape": [8, 8],
                            "batch_size": 8,
                        },
                    },
                ],
            },
        ]
    )
    return {
        "base_variant_config_template": {
            "example_input_shape": [2, 8],
            "batch_size": 2,
            "use_fake_text_dataset": True,
            "qwen3_config": {
                "vocab_size": 32,
                "hidden_size": 16,
                "intermediate_size": 32,
                "num_hidden_layers": 1,
                "num_attention_heads": 2,
                "num_key_value_heads": 2,
                "head_dim": 8,
                "max_position_embeddings": 16,
            },
        },
        "axes": resolved_axes,
        "mutations": [],
    }


def build_qwen_variant_config_grid_arch(
    *,
    variant_config_grid: dict[str, object] | None = None,
    extra_group_fields: dict[str, object] | None = None,
) -> ArchConfig:
    group: dict[str, object] = {
        "base_model": {"name": "qwen3", "pretrained": False},
        "variant_config_grid": variant_config_grid or build_qwen_variant_config_grid(),
    }
    if extra_group_fields is not None:
        group.update(extra_group_fields)
    return ArchConfig.model_validate({"base_model_groups": [group]})


def test_expand_group_variants_generates_grid_and_fc_variants() -> None:
    config = ArchConfig.model_validate(
        {
            "base_model_groups": [
                {
                    "base_model": {"name": "resnet18", "pretrained": True},
                    "combinatorial_variant_grid": {
                        "input_channels": [1, 3],
                        "output_classes": [10],
                        "base_variant_config_template": {
                            "example_input_shape": [1, 3, 224, 224],
                            "run_training": False,
                            "run_inference": True,
                            "export_graph": True,
                        },
                        "mutation_sets": [
                            {"name": "no_mutations", "mutations": []},
                            {
                                "name": "swap_activation",
                                "mutations": [
                                    {
                                        "type": "ActivationSwap",
                                        "params": {
                                            "target_func_name": "relu",
                                            "new_func_name": "gelu",
                                        },
                                    }
                                ],
                            },
                        ],
                    },
                    "fc_mutation_sets": [
                        {
                            "name": "extra_fc",
                            "mutations": [
                                {
                                    "type": "AddIntermediateFCLayer",
                                    "params": {
                                        "hidden_size": 256,
                                        "dropout_rate": 0.1,
                                        "activation": "gelu",
                                    },
                                }
                            ],
                        }
                    ],
                }
            ]
        }
    )

    variants = expand_group_variants(config.base_model_groups[0])

    assert len(variants) == 8
    assert variants[0].group_total_variants_defined == 8
    assert any(variant.name == "resnet18_ic1_oc10_no_mutations" for variant in variants)
    assert any(
        variant.name == "resnet18_ic3_oc10_swap_activation_fc_extra_fc"
        for variant in variants
    )
    assert all(
        variant.variant_config.target_output_classes == 10 for variant in variants
    )
    assert all(variant.variant_config.export_graph for variant in variants)
    assert all(
        variant.variant_config.pre_inference_cooldown_seconds == 3.0
        for variant in variants
    )
    assert all(
        variant.variant_config.inference_measurement_min_seconds == 5.0
        for variant in variants
    )
    assert all(
        variant.variant_config.training_measurement_min_seconds == 5.0
        for variant in variants
    )


def test_expand_group_variants_generates_variant_config_grid_variants() -> None:
    config = build_qwen_variant_config_grid_arch()

    variants = expand_group_variants(config.base_model_groups[0])

    assert [variant.name for variant in variants] == [
        "qwen3_h16_bs2",
        "qwen3_h16_bs8",
        "qwen3_h32_bs2",
        "qwen3_h32_bs8",
    ]
    assert all(variant.source == "variant_config_grid" for variant in variants)
    assert all(variant.group_total_variants_defined == 4 for variant in variants)
    assert [variant.variant_config.example_input_shape for variant in variants] == [
        [2, 8],
        [8, 8],
        [2, 8],
        [8, 8],
    ]
    assert [variant.variant_config.batch_size for variant in variants] == [2, 8, 2, 8]
    assert variants[0].variant_config.qwen3_config is not None
    assert variants[0].variant_config.qwen3_config.hidden_size == 16
    assert variants[2].variant_config.qwen3_config is not None
    assert variants[2].variant_config.qwen3_config.hidden_size == 32


@pytest.mark.parametrize(
    ("axes", "match"),
    [
        ([], "axes must not be empty"),
        (
            [
                {
                    "name": "hidden",
                    "values": [],
                }
            ],
            "axis values must not be empty",
        ),
        (
            [
                {
                    "name": "mlp",
                    "values": [
                        {"name": "same", "overrides": {"qwen3_config.hidden_size": 16}},
                        {"name": "same", "overrides": {"qwen3_config.hidden_size": 32}},
                    ],
                }
            ],
            "value names must be unique",
        ),
        (
            [
                {
                    "name": "dup",
                    "values": [
                        {"name": "a", "overrides": {"qwen3_config.hidden_size": 16}},
                    ],
                },
                {
                    "name": "dup",
                    "values": [
                        {"name": "b", "overrides": {"qwen3_config.hidden_size": 32}},
                    ],
                },
            ],
            "axis names must be unique",
        ),
    ],
)
def test_variant_config_grid_rejects_invalid_axes(
    axes: list[dict[str, object]],
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        build_qwen_variant_config_grid_arch(
            variant_config_grid=build_qwen_variant_config_grid(axes=axes)
        )


def test_variant_config_grid_rejects_unknown_override_path() -> None:
    config = build_qwen_variant_config_grid_arch(
        variant_config_grid=build_qwen_variant_config_grid(
            axes=[
                {
                    "name": "bad",
                    "values": [
                        {
                            "name": "unknown_path",
                            "overrides": {"qwen3_config.unknown_field": 1},
                        }
                    ],
                }
            ]
        )
    )

    with pytest.raises(ValueError, match="override path is unknown"):
        expand_group_variants(config.base_model_groups[0])


def test_variant_config_grid_rejects_duplicate_variant_names() -> None:
    config = build_qwen_variant_config_grid_arch(
        extra_group_fields={
            "single_variant_define": [
                {
                    "name": "qwen3_h16_bs2",
                    "variant_config": {
                        "example_input_shape": [2, 8],
                        "batch_size": 2,
                        "use_fake_text_dataset": True,
                        "qwen3_config": {
                            "vocab_size": 32,
                            "hidden_size": 16,
                            "intermediate_size": 32,
                            "num_hidden_layers": 1,
                            "num_attention_heads": 2,
                            "num_key_value_heads": 2,
                            "head_dim": 8,
                            "max_position_embeddings": 16,
                        },
                    },
                    "mutations": [],
                }
            ]
        },
    )

    with pytest.raises(ValueError, match="variant names must be unique"):
        expand_group_variants(config.base_model_groups[0])


def test_base_model_group_rejects_multiple_grid_types() -> None:
    with pytest.raises(ValueError, match="must not define both"):
        build_qwen_variant_config_grid_arch(
            extra_group_fields={
                "combinatorial_variant_grid": {
                    "input_channels": [1],
                    "output_classes": [2],
                    "base_variant_config_template": {
                        "example_input_shape": [1, 3, 8, 8],
                    },
                    "mutation_sets": [{"name": "none", "mutations": []}],
                }
            }
        )


def test_expand_group_variants_keeps_text_input_shape_flat() -> None:
    config = ArchConfig.model_validate(
        {
            "base_model_groups": [
                {
                    "base_model": {"name": "bert-base-uncased", "pretrained": True},
                    "combinatorial_variant_grid": {
                        "input_channels": [1],
                        "output_classes": [2, 4],
                        "base_variant_config_template": {
                            "example_input_shape": [1, 128],
                            "run_training": False,
                            "run_inference": True,
                            "export_graph": False,
                            "use_fake_text_dataset": True,
                        },
                        "mutation_sets": [
                            {
                                "name": "reduce_layers",
                                "mutations": [
                                    {
                                        "type": "TransformerLayerReduction",
                                        "params": {
                                            "target_layers": 8,
                                            "reduction_strategy": "keep_early",
                                        },
                                    }
                                ],
                            }
                        ],
                    },
                }
            ]
        }
    )

    variants = expand_group_variants(config.base_model_groups[0])

    assert len(variants) == 2
    assert all(
        variant.variant_config.example_input_shape == [1, 128] for variant in variants
    )
    assert all(
        variant.variant_config.target_input_channels == 1 for variant in variants
    )


def test_variant_config_rejects_training_epochs_field() -> None:
    with pytest.raises(ValueError, match="training_epochs"):
        ArchConfig.model_validate(
            {
                "base_model_groups": [
                    {
                        "base_model": {"name": "resnet18", "pretrained": True},
                        "single_variant_define": [
                            {
                                "name": "old_training_field",
                                "variant_config": {
                                    "target_input_channels": 3,
                                    "target_output_classes": 10,
                                    "example_input_shape": [1, 3, 224, 224],
                                    "training_epochs": 3,
                                },
                            }
                        ],
                    }
                ]
            }
        )


def test_variant_config_rejects_fake_dataset_size_field() -> None:
    with pytest.raises(ValueError, match="fake_dataset_size"):
        ArchConfig.model_validate(
            {
                "base_model_groups": [
                    {
                        "base_model": {"name": "resnet18", "pretrained": True},
                        "single_variant_define": [
                            {
                                "name": "old_dataset_field",
                                "variant_config": {
                                    "target_input_channels": 3,
                                    "target_output_classes": 10,
                                    "example_input_shape": [1, 3, 224, 224],
                                    "fake_dataset_size": 4,
                                },
                            }
                        ],
                    }
                ]
            }
        )


def test_is_text_model_name_uses_normalized_prefixes() -> None:
    assert is_text_model_name("bert-base-uncased")
    assert is_text_model_name("google/flan-t5-base")
    assert is_text_model_name("unsloth/Llama-3.2-1B")
    assert is_text_model_name("unsloth/gemma-3-1b-pt")
    assert not is_text_model_name("resnet50")


def test_causal_lm_model_name_uses_normalized_prefixes() -> None:
    assert is_causal_lm_model_name("Qwen3-1.7B")
    assert is_causal_lm_model_name("unsloth/Llama-3.2-1B")
    assert is_causal_lm_model_name("unsloth/gemma-3-1b-pt")
    assert get_causal_lm_family("Qwen3-1.7B") == "qwen"
    assert get_causal_lm_family("unsloth/Llama-3.2-1B") == "llama"
    assert get_causal_lm_family("unsloth/gemma-3-1b-pt") == "gemma"
    assert not is_causal_lm_model_name("gpt2")


def test_expand_arch_config_keeps_resnet50_as_image_model() -> None:
    config_path = ROOT / "config" / "arch" / "resnet50_variants.yaml"
    raw_config = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    config = ArchConfig.model_validate(raw_config)
    variants = expand_arch_config(config)

    assert variants
    assert all(not is_text_model_name(variant.base_model.name) for variant in variants)
    assert all(
        len(variant.variant_config.example_input_shape) == 4 for variant in variants
    )
    assert all(
        variant.variant_config.target_input_channels == 1 for variant in variants
    )
