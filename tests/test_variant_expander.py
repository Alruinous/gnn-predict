from __future__ import annotations

from pathlib import Path

import yaml

from gnn_archs.config import ArchConfig, is_text_model_name
from gnn_archs.util.variant_expander import expand_arch_config, expand_group_variants


ROOT = Path(__file__).resolve().parents[1]


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
                            "export_onnx": True,
                            "onnx_export_mode": "architecture_only",
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
    assert all(
        variant.variant_config.onnx_export_mode == "architecture_only"
        for variant in variants
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
                            "export_onnx": False,
                            "use_fake_text_dataset": True,
                            "max_sequence_length": 128,
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
    assert all(variant.variant_config.target_input_channels == 1 for variant in variants)


def test_is_text_model_name_uses_normalized_prefixes() -> None:
    assert is_text_model_name("bert-base-uncased")
    assert is_text_model_name("google/flan-t5-base")
    assert not is_text_model_name("resnet50")


def test_expand_arch_config_keeps_resnet50_as_image_model() -> None:
    config_path = ROOT / "config" / "arch" / "resnet50_variants.yaml"
    raw_config = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    config = ArchConfig.model_validate(raw_config)
    variants = expand_arch_config(config)

    assert variants
    assert all(not is_text_model_name(variant.base_model.name) for variant in variants)
    assert all(len(variant.variant_config.example_input_shape) == 4 for variant in variants)
    assert all(variant.variant_config.target_input_channels == 1 for variant in variants)
