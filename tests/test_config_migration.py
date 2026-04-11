from __future__ import annotations

from pathlib import Path

import yaml

from gnn_archs.config import ArchConfig
from gnn_archs.util.config_migration import migrate_arch_config_dict


ROOT = Path(__file__).resolve().parents[1]


def test_migrated_resnet_config_matches_new_schema() -> None:
    migrated_path = ROOT / "config" / "arch" / "resnet_variants.yaml"
    raw_config = yaml.safe_load(migrated_path.read_text(encoding="utf-8"))

    config = ArchConfig.model_validate(raw_config)
    group = config.base_model_groups[0]
    template = group.combinatorial_variant_grid.base_variant_config_template

    assert template.run_training is True
    assert template.run_inference is True
    assert template.export_onnx is True
    assert template.onnx_export_mode == "architecture_only"
    assert template.target_input_channels is None
    assert template.target_output_classes is None
    assert group.combinatorial_variant_grid.mutation_sets[1].mutations[0].params[
        "layer_name"
    ] == "conv1"


def test_legacy_mutation_keys_are_moved_into_params() -> None:
    legacy_config = {
        "base_model_groups": [
            {
                "base_model": {"name": "resnet18", "pretrained": True},
                "single_variant_define": [
                    {
                        "name": "sample_variant",
                        "variant_config": {
                            "target_input_channels": 3,
                            "target_output_classes": 10,
                            "example_input_shape": [1, 3, 224, 224],
                            "run_workload": True,
                            "export_onnx": True,
                        },
                        "mutations": [
                            {
                                "type": "ActivationSwap",
                                "target_func_name": "relu",
                                "new_func_name": "gelu",
                            }
                        ],
                    }
                ],
            }
        ]
    }

    migrated = migrate_arch_config_dict(legacy_config)
    mutation = migrated["base_model_groups"][0]["single_variant_define"][0]["mutations"][
        0
    ]
    variant_config = migrated["base_model_groups"][0]["single_variant_define"][0][
        "variant_config"
    ]

    assert mutation == {
        "type": "ActivationSwap",
        "params": {"target_func_name": "relu", "new_func_name": "gelu"},
    }
    assert "run_workload" not in variant_config
    assert variant_config["run_inference"] is True
    assert variant_config["onnx_export_mode"] == "full"
