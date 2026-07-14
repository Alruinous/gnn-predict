"""YOLO detection helpers built around Ultralytics YAML parsing."""

from __future__ import annotations

import copy
import time
from typing import TYPE_CHECKING, Any

import torch
import torch.nn as nn

from common.graph_artifact import capture_inference_graph
from gnn_archs.result import (
    GraphExportResult,
    InferenceResult,
    TimeWindow,
    TrainingResult,
)

if TYPE_CHECKING:
    from gnn_archs.config import ResolvedVariantSpec
    from gnn_archs.variant_runner import RunContext


YOLO_YAML_MUTATION_TYPES = frozenset(
    {
        "BackboneModuleReplace",
        "HeadModuleReplace",
        "ActivationOverride",
        "SPPFKernelReplace",
        "BackboneDepthOverride",
        "ChannelScale",
    }
)

CSP_MODULE_ARG_ORDER = {
    "C2f": ("c2", "shortcut", "g", "e"),
    "C3": ("c2", "shortcut", "g", "e"),
    "C3k2": ("c2", "c3k", "e", "attn", "g", "shortcut"),
    "C2fCIB": ("c2", "shortcut", "lk", "g", "e"),
}

CSP_MODULE_DEFAULTS = {
    "C2f": {"shortcut": False, "g": 1, "e": 0.5},
    "C3": {"shortcut": True, "g": 1, "e": 0.5},
    "C3k2": {
        "c3k": False,
        "e": 0.5,
        "attn": False,
        "g": 1,
        "shortcut": True,
    },
    "C2fCIB": {"shortcut": False, "lk": False, "g": 1, "e": 0.5},
}


def build_detection_model(spec: ResolvedVariantSpec) -> nn.Module:
    from ultralytics import YOLO
    from ultralytics.nn.tasks import DetectionModel

    model_name = spec.base_model.name
    has_mutations = len(spec.mutations) > 0
    is_custom_scale = "_" in model_name

    custom_nc = spec.variant_config.target_output_classes
    needs_custom_nc = custom_nc is not None and custom_nc != 80
    target_input_channels = spec.variant_config.target_input_channels or 3
    needs_custom_input_channels = target_input_channels != 3

    if (
        not has_mutations
        and not is_custom_scale
        and not needs_custom_nc
        and not needs_custom_input_channels
    ):
        yolo = YOLO(f"{model_name}.yaml", task="detect")
        return yolo.model

    d = _load_yolo_yaml_dict(model_name)

    if is_custom_scale:
        d = copy.deepcopy(d)
        scale_name = model_name.split("_", 1)[1]
        if scale_name not in CUSTOM_SCALES:
            raise ValueError(
                f"Unknown custom scale '{scale_name}'. "
                f"Available: {sorted(CUSTOM_SCALES.keys())}"
            )
        d.setdefault("scales", {})[scale_name] = CUSTOM_SCALES[scale_name]
        d["scale"] = scale_name

    if needs_custom_nc:
        d["nc"] = custom_nc

    if has_mutations:
        d = _apply_yaml_mutations(d, spec.mutations)

    build_d = copy.deepcopy(d)
    build_d.pop("yaml_file", None)
    return DetectionModel(
        build_d,
        ch=target_input_channels,
        nc=build_d.get("nc"),
        verbose=False,
    )


def _load_yolo_yaml_dict(model_name: str) -> dict[str, Any]:
    from ultralytics.nn.tasks import yaml_model_load

    if "_" in model_name:
        base_name = model_name.split("_", 1)[0]
        return yaml_model_load(f"{base_name}.yaml")

    return yaml_model_load(f"{model_name}.yaml")


CUSTOM_SCALES = {
    "pico": [0.33, 0.25, 512],
    "micro": [0.33, 0.25, 768],
    "ns_1": [0.50, 0.31, 1024],
    "ns_2": [0.50, 0.375, 1024],
    "sm_1": [0.50, 0.625, 768],
    "sm_2": [0.50, 0.75, 768],
    "ml_1": [0.625, 1.00, 512],
    "ml_2": [0.75, 1.00, 512],
    "lx_1": [1.00, 1.125, 512],
    "lx_2": [1.00, 1.25, 512],
    "huge": [1.50, 2.00, 1024],
    "mega": [2.00, 2.00, 1024],
    "tiny_wide": [0.33, 0.50, 512],
    "shallow_wide": [0.33, 1.00, 512],
    "deep_narrow": [1.50, 0.25, 1024],
    "deep_mid": [1.50, 0.50, 768],
    "ultra_deep": [2.00, 0.50, 512],
}


def _apply_yaml_mutations(d: dict[str, Any], mutations: list[Any]) -> dict[str, Any]:
    d = copy.deepcopy(d)

    for mutation in mutations:
        mt = mutation.type
        params = mutation.params

        if mt not in YOLO_YAML_MUTATION_TYPES:
            raise ValueError(
                f"未知的 YOLO mutation 类型 '{mt}'。"
                f"合法类型: {sorted(YOLO_YAML_MUTATION_TYPES)}"
            )

        if mt == "BackboneModuleReplace":
            _replace_module_in_section(d, "backbone", params)

        elif mt == "HeadModuleReplace":
            _replace_module_in_section(d, "head", params)

        elif mt == "ActivationOverride":
            d["activation"] = params["activation"]

        elif mt == "SPPFKernelReplace":
            for section in ("backbone", "head"):
                for layer in d.get(section, []):
                    if len(layer) >= 4 and layer[2] == "SPPF" and len(layer[3]) > 1:
                        layer[3][1] = params["kernel_size"]

        elif mt == "BackboneDepthOverride":
            repeat_val = params["repeat_value"]
            for layer in d.get("backbone", []):
                if layer[1] > 1:
                    layer[1] = repeat_val

        elif mt == "ChannelScale":
            factor = params["scale_factor"]
            for section in ("backbone", "head"):
                for layer in d.get(section, []):
                    if (
                        len(layer) >= 4
                        and isinstance(layer[3], list)
                        and len(layer[3]) > 0
                        and isinstance(layer[3][0], (int, float))
                        and layer[3][0] > 1
                    ):
                        layer[3][0] = max(8, int(layer[3][0] * factor))

    return d


def _replace_module_in_section(
    d: dict[str, Any], section: str, params: dict[str, Any]
) -> None:
    from_mod = params["from_module"]
    to_mod = params["to_module"]
    target_layers = params.get("layers")
    pending_layers = set(target_layers) if target_layers is not None else None
    matched_index = 0
    replaced_count = 0

    for layer in d.get(section, []):
        if not (len(layer) >= 3 and layer[2] == from_mod):
            continue
        should_replace = pending_layers is None or matched_index in pending_layers
        if not should_replace:
            matched_index += 1
            continue
        if len(layer) >= 4 and isinstance(layer[3], list):
            layer[3] = _adapt_module_replacement_args(from_mod, to_mod, layer[3])
        layer[2] = to_mod
        replaced_count += 1
        if pending_layers is not None:
            pending_layers.remove(matched_index)
        matched_index += 1

    if matched_index == 0:
        raise ValueError(f"No {from_mod} layers found in YOLO {section}")
    if pending_layers:
        raise ValueError(
            f"YOLO {section} {from_mod} layer indexes not found: "
            f"{sorted(pending_layers)}"
        )
    if replaced_count == 0:
        raise ValueError(f"No {from_mod} layers replaced in YOLO {section}")


def _adapt_module_replacement_args(
    from_module: str, to_module: str, args: list[Any]
) -> list[Any]:
    if to_module not in CSP_MODULE_ARG_ORDER:
        return list(args)

    if not args:
        raise ValueError(f"{from_module} args must include output channels")

    parsed_args = {"c2": args[0]}
    if from_module in CSP_MODULE_ARG_ORDER:
        source_order = CSP_MODULE_ARG_ORDER[from_module]
        if len(args) > len(source_order):
            raise ValueError(
                f"{from_module} accepts at most {len(source_order)} YAML args"
            )
        parsed_args.update(CSP_MODULE_DEFAULTS[from_module])
        parsed_args.update(
            {key: value for key, value in zip(source_order[1:], args[1:], strict=False)}
        )

    adapted_args = []
    for key in CSP_MODULE_ARG_ORDER[to_module]:
        if key in parsed_args:
            adapted_args.append(parsed_args[key])
        else:
            adapted_args.append(CSP_MODULE_DEFAULTS[to_module][key])
    return adapted_args


def export_detection_graph(
    spec: ResolvedVariantSpec,
    model: nn.Module,
    context: RunContext,
) -> GraphExportResult:
    from gnn_archs.graph_export import write_graph_export

    model.eval()
    device = next(model.parameters()).device
    dummy_input = torch.randn(*spec.variant_config.example_input_shape, device=device)
    with torch.no_grad():
        model(dummy_input)

    export_states = []
    for m in model.modules():
        if hasattr(m, "export"):
            export_states.append((m, m.export))
            m.export = True

    export_path = context.output_layout.fx_graphs_dir / f"{spec.name}.pt2"
    try:
        exported_program = capture_inference_graph(
            model,
            (dummy_input,),
        )
        return write_graph_export(exported_program, export_path, ["images"])
    finally:
        for module, export_state in export_states:
            module.export = export_state


def train_detection_model(
    spec: ResolvedVariantSpec,
    model: nn.Module,
    device: torch.device,
) -> TrainingResult:
    """Run synthetic training for detection models."""
    batch_size = spec.variant_config.batch_size
    _, channels, height, width = spec.variant_config.example_input_shape
    generator = torch.Generator(device=device).manual_seed(42)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    measurement_min_seconds = spec.variant_config.training_measurement_min_seconds
    total_steps = 0
    last_loss = 0.0

    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(device)
    training_started_at = time.time()
    training_ended_at = training_started_at

    model.train()
    while (
        total_steps == 0
        or training_ended_at - training_started_at < measurement_min_seconds
    ):
        optimizer.zero_grad(set_to_none=True)
        img_batch = torch.randn(
            (batch_size, channels, height, width),
            generator=generator,
            device=device,
        )
        outputs = model(img_batch)
        if not isinstance(outputs, dict):
            raise TypeError(f"unexpected YOLO training output type: {type(outputs)}")
        boxes = outputs.get("boxes")
        scores = outputs.get("scores")
        if not isinstance(boxes, torch.Tensor) or not isinstance(scores, torch.Tensor):
            raise TypeError("YOLO training output must contain boxes and scores")

        loss = boxes.sum() + scores.sum()
        loss.backward()
        optimizer.step()

        total_steps += 1
        last_loss = float(loss.detach().item())
        if device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize(device)
        training_ended_at = time.time()

    return TrainingResult(
        hyperparameters={
            "batch_size": batch_size,
            "measurement_min_seconds": measurement_min_seconds,
        },
        optimizer={"name": "AdamW", "lr": 1e-3},
        metrics={
            "final_loss": last_loss,
            "total_steps": total_steps,
        },
        timings=_build_time_window(training_started_at, training_ended_at),
    )


def run_detection_inference(
    spec: ResolvedVariantSpec,
    model: nn.Module,
    device: torch.device,
) -> InferenceResult:
    """Measure detection model inference latency."""
    generator = torch.Generator(device=device).manual_seed(42)
    dummy_input = torch.randn(
        *spec.variant_config.example_input_shape,
        generator=generator,
        device=device,
    )

    measurement_min_seconds = spec.variant_config.inference_measurement_min_seconds
    iterations = 0
    outputs: Any | None = None

    model.eval()
    with torch.no_grad():
        for _ in range(2):
            _ = model(dummy_input)

        if device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize(device)

        started_at = time.time()
        ended_at = started_at
        while ended_at - started_at < measurement_min_seconds:
            outputs = model(dummy_input)
            if device.type == "cuda" and torch.cuda.is_available():
                torch.cuda.synchronize(device)
            iterations += 1
            ended_at = time.time()

    assert outputs is not None
    total_duration = ended_at - started_at
    if not isinstance(outputs, tuple) or len(outputs) != 2:
        raise TypeError(f"unexpected YOLO inference output type: {type(outputs)}")
    predictions = outputs[0]
    if not isinstance(predictions, torch.Tensor):
        raise TypeError(f"unexpected YOLO prediction output type: {type(predictions)}")

    return InferenceResult(
        metrics={
            "iterations": iterations,
            "batch_size": spec.variant_config.example_input_shape[0],
            "avg_latency_ms": round(total_duration / iterations * 1000, 4),
            "num_outputs": int(predictions.shape[-1]),
        },
        timings=_build_time_window(started_at, ended_at),
    )


def _build_time_window(started_at: float, ended_at: float) -> TimeWindow:
    from datetime import UTC, datetime

    return TimeWindow(
        started_at_ts=started_at,
        ended_at_ts=ended_at,
        started_at_text=datetime.fromtimestamp(started_at, tz=UTC).isoformat(),
        ended_at_text=datetime.fromtimestamp(ended_at, tz=UTC).isoformat(),
    )
