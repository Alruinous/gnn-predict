from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest
import torch
import yaml
from ultralytics.nn.tasks import yaml_model_load

from gnn_archs.config import ArchConfig
from gnn_archs.util.variant_expander import expand_arch_config
from gnn_archs.variant_runner import (
    RunContext,
    build_variant_model,
    prepare_output_layout,
)
from gnn_archs.yolo_builder import (
    _adapt_module_replacement_args,
    _apply_yaml_mutations,
    export_detection_onnx,
    run_detection_inference,
    train_detection_model,
)

if TYPE_CHECKING:
    from gnn_archs.config import ResolvedVariantSpec


ROOT = Path(__file__).resolve().parents[1]


def _build_yolo_variant(
    mutations: list[dict[str, Any]],
    *,
    base_model_name: str = "yolo11n",
    variant_name: str = "yolo_builder_smoke",
    variant_config_overrides: dict[str, object] | None = None,
) -> ResolvedVariantSpec:
    variant_config: dict[str, object] = {
        "target_input_channels": 3,
        "target_output_classes": 10,
        "example_input_shape": [1, 3, 64, 64],
        "onnx_export_mode": "architecture_only",
    }
    if variant_config_overrides is not None:
        variant_config.update(variant_config_overrides)

    config = ArchConfig.model_validate(
        {
            "base_model_groups": [
                {
                    "base_model": {"name": base_model_name, "pretrained": False},
                    "single_variant_define": [
                        {
                            "name": variant_name,
                            "variant_config": variant_config,
                            "mutations": mutations,
                        }
                    ],
                }
            ]
        }
    )
    return expand_arch_config(config)[0]


class _FixedTrainDetectionModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.0))

    def forward(self, inputs: torch.Tensor) -> dict[str, object]:
        batch_size = inputs.shape[0]
        return {
            "boxes": self.weight
            * torch.ones((batch_size, 2, 3), device=inputs.device),
            "scores": self.weight
            * torch.ones((batch_size, 4, 3), device=inputs.device),
            "feats": [],
        }


class _FixedInferenceDetectionModel(torch.nn.Module):
    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, dict[str, object]]:
        batch_size = inputs.shape[0]
        predictions = torch.zeros((batch_size, 14, 21), device=inputs.device)
        raw = {
            "boxes": torch.zeros((batch_size, 64, 21), device=inputs.device),
            "scores": torch.zeros((batch_size, 10, 21), device=inputs.device),
            "feats": [],
        }
        return predictions, raw


def _retained_custom_scale_names() -> list[str]:
    names: list[str] = []
    for config_path in sorted((ROOT / "config" / "arch").glob("*yolo*_variants.yaml")):
        config = ArchConfig.model_validate(
            yaml.safe_load(config_path.read_text(encoding="utf-8"))
        )
        names.extend(
            group.base_model.name
            for group in config.base_model_groups
            if "_" in group.base_model.name
        )
    return names


def test_c3k2_replacement_args_keep_expansion_out_of_groups() -> None:
    c2f_args = _adapt_module_replacement_args("C3k2", "C2f", [256, False, 0.25])
    c3_args = _adapt_module_replacement_args("C3k2", "C3", [256, False, 0.25])

    assert c2f_args == [256, True, 1, 0.25]
    assert c3_args == [256, True, 1, 0.25]


def test_repncspelan4_replacement_uses_target_module_defaults() -> None:
    c2f_args = _adapt_module_replacement_args(
        "RepNCSPELAN4", "C2f", [64, 64, 32, 3]
    )

    assert c2f_args == [64, False, 1, 0.5]


def test_yolo11_c3k2_to_c2f_builds_with_native_scale() -> None:
    variant = _build_yolo_variant(
        [
            {
                "type": "BackboneModuleReplace",
                "params": {"from_module": "C3k2", "to_module": "C2f"},
            }
        ]
    )

    model = build_variant_model(variant)

    assert model.yaml["scale"] == "n"
    assert model.yaml["nc"] == 10


def test_yolo11_custom_scale_survives_dict_build() -> None:
    variant = _build_yolo_variant([], base_model_name="yolo11_pico")

    model = build_variant_model(variant)

    assert model.yaml["scale"] == "pico"
    assert "pico" in model.yaml["scales"]


@pytest.mark.parametrize("base_model_name", _retained_custom_scale_names())
def test_retained_custom_scale_baselines_build(base_model_name: str) -> None:
    variant = _build_yolo_variant([], base_model_name=base_model_name)

    model = build_variant_model(variant)

    assert model.yaml["scale"] == base_model_name.split("_", 1)[1]
    assert model.yaml["nc"] == 10


def test_module_replace_layers_are_match_indexes() -> None:
    yolo_dict = yaml_model_load("yolo11n.yaml")

    mutated = _apply_yaml_mutations(
        yolo_dict,
        [
            _build_yolo_variant(
                [
                    {
                        "type": "HeadModuleReplace",
                        "params": {
                            "from_module": "C3k2",
                            "to_module": "C2f",
                            "layers": [0, 1],
                        },
                    }
                ]
            ).mutations[0]
        ],
    )

    assert mutated["head"][2][2] == "C2f"
    assert mutated["head"][5][2] == "C2f"
    assert mutated["head"][8][2] == "C3k2"
    assert mutated["head"][11][2] == "C3k2"


def test_module_replace_rejects_missing_from_module() -> None:
    yolo_dict = yaml_model_load("yolo11n.yaml")
    mutation = _build_yolo_variant(
        [
            {
                "type": "BackboneModuleReplace",
                "params": {"from_module": "C2f", "to_module": "C3"},
            }
        ]
    ).mutations[0]

    with pytest.raises(ValueError, match=r"No C2f layers found"):
        _apply_yaml_mutations(yolo_dict, [mutation])


def test_module_replace_rejects_unknown_layer_index() -> None:
    yolo_dict = yaml_model_load("yolo11n.yaml")
    mutation = _build_yolo_variant(
        [
            {
                "type": "BackboneModuleReplace",
                "params": {
                    "from_module": "C3k2",
                    "to_module": "C2f",
                    "layers": [99],
                },
            }
        ]
    ).mutations[0]

    with pytest.raises(ValueError, match=r"layer indexes not found"):
        _apply_yaml_mutations(yolo_dict, [mutation])


def test_export_detection_onnx_restores_export_flags(tmp_path: Path) -> None:
    variant = _build_yolo_variant([], variant_name="yolo_export_restore")
    model = build_variant_model(variant)
    export_flags_before = [
        module.export for module in model.modules() if hasattr(module, "export")
    ]
    output_layout = prepare_output_layout(tmp_path / "output", tmp_path / "cfg.yaml")
    context = RunContext(
        config_path=tmp_path / "cfg.yaml",
        output_layout=output_layout,
        device=torch.device("cpu"),
        gpu_node="cpu-test",
        logger=logging.getLogger("test_export_detection_onnx"),
    )

    result = export_detection_onnx(variant, model, context)

    export_flags_after = [
        module.export for module in model.modules() if hasattr(module, "export")
    ]
    assert export_flags_after == export_flags_before
    assert result.graph_info["runtime_input_names"] == ["images"]


def test_train_detection_model_uses_fixed_train_output_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    variant = _build_yolo_variant(
        [],
        variant_config_overrides={
            "example_input_shape": [1, 3, 8, 8],
            "training_batch_sizes": [2],
            "training_epochs": 1,
            "fake_dataset_size": 4,
        },
    )
    model = _FixedTrainDetectionModel()
    original_randn = torch.randn
    generated_shapes: list[tuple[int, ...]] = []

    def track_randn(*args: Any, **kwargs: Any) -> torch.Tensor:
        shape = args[0]
        if isinstance(shape, tuple):
            generated_shapes.append(shape)
        return original_randn(*args, **kwargs)

    monkeypatch.setattr(torch, "randn", track_randn)

    result = train_detection_model(variant, model, torch.device("cpu"))

    assert result.metrics["total_steps"] == 2
    assert generated_shapes == [(2, 3, 8, 8), (2, 3, 8, 8)]


def test_run_detection_inference_uses_fixed_eval_output_contract() -> None:
    variant = _build_yolo_variant(
        [],
        variant_config_overrides={
            "example_input_shape": [2, 3, 8, 8],
            "inference_measurement_min_seconds": 1e-9,
        },
    )
    model = _FixedInferenceDetectionModel()

    result = run_detection_inference(variant, model, torch.device("cpu"))

    assert result.metrics["batch_size"] == 2
    assert result.metrics["num_outputs"] == 21
    assert result.metrics["iterations"] >= 1


def test_detection_variant_rejects_unknown_mutation_type() -> None:
    variant = _build_yolo_variant(
        [{"type": "NonExistentMutation", "params": {"foo": "bar"}}]
    )

    with pytest.raises(ValueError, match=r"未知的 YOLO mutation 类型"):
        build_variant_model(variant)
