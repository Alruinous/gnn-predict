from __future__ import annotations

import json
from pathlib import Path

import yaml

from main import main


def test_main_processes_single_config_file(tmp_path: Path) -> None:
    config_path = tmp_path / "runtime_variants.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "base_model_groups": [
                    {
                        "base_model": {"name": "resnet18", "pretrained": False},
                        "single_variant_define": [
                            {
                                "name": "main_smoke_variant",
                                "variant_config": {
                                    "target_input_channels": 3,
                                    "target_output_classes": 2,
                                    "example_input_shape": [1, 3, 32, 32],
                                    "run_training": False,
                                    "run_inference": False,
                                    "export_onnx": False,
                                },
                                "mutations": [],
                            }
                        ],
                    }
                ]
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    output_dir = tmp_path / "run_output"
    config_output_dir = output_dir / "runtime"
    exit_code = main(
        [
            "--config",
            str(config_path),
            "--output_dir",
            str(output_dir),
            "--gpu_node",
            "cpu-test",
        ]
    )

    result_files = sorted((config_output_dir / "results").glob("*.json"))
    log_files = sorted((config_output_dir / "logs").glob("*.log"))

    assert exit_code == 0
    assert len(result_files) == 1
    assert len(log_files) == 1
    payload = json.loads(result_files[0].read_text(encoding="utf-8"))
    log_text = log_files[0].read_text(encoding="utf-8")

    assert payload["schema_version"] == "2.0.0"
    assert payload["summary"]["variant_count"] == 1
    assert payload["variants"][0]["name"] == "main_smoke_variant"
    assert "processing variant 1/1: main_smoke_variant" in log_text
    assert "gpu_ids" not in payload
