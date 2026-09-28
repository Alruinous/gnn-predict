from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import yaml

import gnn_archs.result as result_module
import gnn_archs.variant_runner as variant_runner_module
from gnn_archs.config import ResolvedVariantSpec
from gnn_archs.result import VariantResult
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
                                    "export_graph": False,
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

    assert payload["schema_version"] == "3.0.0"
    assert payload["summary"]["variant_count"] == 1
    assert payload["variants"][0]["name"] == "main_smoke_variant"
    assert "processing variant 1/1: main_smoke_variant" in log_text
    assert "gpu_ids" not in payload


def test_main_continues_after_variant_failure_and_checkpoints_results(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    names = ["first", "broken", "last"]
    config_path = tmp_path / "batch_variants.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "base_model_groups": [
                    {
                        "base_model": {"name": "resnet18", "pretrained": False},
                        "single_variant_define": [
                            {
                                "name": name,
                                "variant_config": {
                                    "target_input_channels": 3,
                                    "target_output_classes": 2,
                                    "example_input_shape": [1, 3, 32, 32],
                                },
                                "mutations": [],
                            }
                            for name in names
                        ],
                    }
                ]
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    output_dir = tmp_path / "output"
    saved_snapshots: list[dict[str, Any]] = []
    original_write = result_module.write_result_document

    def record_write(path: Path, document: result_module.ResultDocument) -> None:
        original_write(path, document)
        saved_snapshots.append(json.loads(path.read_text(encoding="utf-8")))

    def fake_run_variant(
        spec: ResolvedVariantSpec, context: variant_runner_module.RunContext
    ) -> VariantResult:
        name = spec.name
        if name == "broken":
            if context.stage_tracker is not None:
                context.stage_tracker("graph_export")
            raise RuntimeError("unsupported export operator")
        return VariantResult(
            name=name,
            base_model_name="resnet18",
            base_model_pretrained=False,
            source="single_variant_define",
            group_total_variants_defined=3,
            variant_config={},
            mutations=[],
        )

    monkeypatch.setattr(result_module, "write_result_document", record_write)
    monkeypatch.setattr(variant_runner_module, "run_variant", fake_run_variant)
    exit_code = main(
        [
            "--config",
            str(config_path),
            "--output_dir",
            str(output_dir),
            "--gpu_node",
            "cpu-test",
            "--continue_on_variant_error",
        ]
    )

    assert exit_code == 1
    assert [
        [row["name"] for row in snapshot["variants"]] for snapshot in saved_snapshots
    ] == [
        ["first"],
        ["first"],
        ["first", "last"],
        ["first", "last"],
    ]
    final = saved_snapshots[-1]
    assert final["summary"]["planned_variant_count"] == 3
    assert final["summary"]["variant_count"] == 2
    assert final["summary"]["failed_variant_count"] == 1
    assert final["failures"] == [
        {
            "name": "broken",
            "stage": "graph_export",
            "error_type": "RuntimeError",
            "message": "unsupported export operator",
        }
    ]


def test_main_keeps_checkpoint_when_later_variant_is_interrupted(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = tmp_path / "interrupt_variants.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "base_model_groups": [
                    {
                        "base_model": {"name": "resnet18", "pretrained": False},
                        "single_variant_define": [
                            {
                                "name": name,
                                "variant_config": {
                                    "target_input_channels": 3,
                                    "target_output_classes": 2,
                                    "example_input_shape": [1, 3, 32, 32],
                                },
                                "mutations": [],
                            }
                            for name in ("first", "interrupted")
                        ],
                    }
                ]
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    output_dir = tmp_path / "output"

    def fake_run_variant(
        spec: ResolvedVariantSpec, _context: variant_runner_module.RunContext
    ) -> VariantResult:
        name = spec.name
        if name == "interrupted":
            raise KeyboardInterrupt
        return VariantResult(
            name=name,
            base_model_name="resnet18",
            base_model_pretrained=False,
            source="single_variant_define",
            group_total_variants_defined=2,
            variant_config={},
            mutations=[],
        )

    monkeypatch.setattr(variant_runner_module, "run_variant", fake_run_variant)
    with pytest.raises(KeyboardInterrupt):
        main(
            [
                "--config",
                str(config_path),
                "--output_dir",
                str(output_dir),
                "--gpu_node",
                "cpu-test",
                "--continue_on_variant_error",
            ]
        )

    result_files = list((output_dir / "interrupt" / "results").glob("*.json"))
    assert len(result_files) == 1
    payload = json.loads(result_files[0].read_text(encoding="utf-8"))
    assert [variant["name"] for variant in payload["variants"]] == ["first"]
    assert payload["summary"]["planned_variant_count"] == 2
