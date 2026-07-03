from __future__ import annotations

import gc
import json
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Protocol, cast

import onnx
import timm
import torch
import torch.nn as nn
from transformers import BertConfig, BertForSequenceClassification

from gnn_archs.causal_lm_builder import CausalLMOnnxLogitsExport
from gnn_archs.config import (
    get_causal_lm_family,
    is_causal_lm_model_name,
    is_detection_model_name,
    is_recommender_model_name,
    is_text_model_name,
    normalize_model_identifier,
)
from gnn_archs.mutations import (
    IMAGE_MUTATION_TYPES,
    TEXT_MUTATION_TYPES,
    apply_image_mutations,
    apply_text_config_mutations,
    validate_mutation_types,
)
from gnn_archs.result import (
    InferenceResult,
    OnnxExportResult,
    TimeWindow,
    TrainingResult,
    VariantResult,
)
from gnn_archs.util.onnx_initializer import (
    ONNX_EXPORT_MODE_METADATA_KEY,
    RUNTIME_INPUT_NAMES_METADATA_KEY,
    set_model_metadata_value,
)

if TYPE_CHECKING:
    import logging
    from pathlib import Path

    from gnn_archs.config import BaseModelConfig, ResolvedVariantSpec, VariantConfig


CONFIG_VARIANTS_SUFFIX = "_variants"


@dataclass(frozen=True)
class OutputLayout:
    root: Path
    logs_dir: Path
    results_dir: Path
    onnx_models_dir: Path


@dataclass(frozen=True)
class RunContext:
    config_path: Path
    output_layout: OutputLayout
    device: torch.device
    gpu_node: str
    logger: logging.Logger


class CausalLMGenerator(Protocol):
    def generate(self, **kwargs: Any) -> torch.Tensor: ...


class OnnxExportWrapper(nn.Module):
    def __init__(
        self,
        model: nn.Module,
        is_text_model: bool,
        recommender_feature_names: list[str] | None = None,
    ) -> None:
        super().__init__()
        self.model = model
        self.is_text_model = is_text_model
        self.recommender_feature_names = recommender_feature_names

    def forward(self, *inputs: torch.Tensor) -> torch.Tensor:
        if self.recommender_feature_names is not None:
            batch = dict(zip(self.recommender_feature_names, inputs, strict=True))
            return extract_logits(self.model(batch)).unsqueeze(1)  # [B]->[B,1]
        if self.is_text_model:
            outputs = self.model(input_ids=inputs[0], attention_mask=inputs[1])
            return extract_logits(outputs)
        outputs = self.model(inputs[0])
        return extract_logits(outputs)


def derive_config_output_name(config_path: Path) -> str:
    config_name = config_path.stem.strip()
    if not config_name:
        raise ValueError(f"config path must have a non-empty file name: {config_path}")
    if config_name.endswith(CONFIG_VARIANTS_SUFFIX):
        config_name = config_name.removesuffix(CONFIG_VARIANTS_SUFFIX)
    if not config_name:
        raise ValueError(
            f"config path must resolve to a non-empty output name: {config_path}"
        )
    return config_name


def prepare_output_layout(output_root: Path, config_path: Path) -> OutputLayout:
    config_output_root = output_root / derive_config_output_name(config_path)
    layout = OutputLayout(
        root=config_output_root,
        logs_dir=config_output_root / "logs",
        results_dir=config_output_root / "results",
        onnx_models_dir=config_output_root / "onnx_models",
    )
    for directory in (
        layout.root,
        layout.logs_dir,
        layout.results_dir,
        layout.onnx_models_dir,
    ):
        directory.mkdir(parents=True, exist_ok=True)
    return layout


def run_variant(spec: ResolvedVariantSpec, context: RunContext) -> VariantResult:
    run_started_at = time.time()
    timings: dict[str, TimeWindow] = {}
    is_text_model = is_text_model_name(spec.base_model.name)
    is_detection_model = is_detection_model_name(spec.base_model.name)
    is_recommender_model = is_recommender_model_name(spec.base_model.name)
    model: nn.Module | None = None
    onnx_result: OnnxExportResult | None = None
    training_result: TrainingResult | None = None
    inference_result: InferenceResult | None = None
    prefill_result: InferenceResult | None = None
    decode_result: InferenceResult | None = None
    try:
        model_build_started_at = time.time()
        model = build_variant_model(spec)
        model = model.to(context.device)
        timings["model_build"] = build_time_window(model_build_started_at, time.time())

        validation_started_at = time.time()
        validation_metrics = validate_model(
            spec,
            model,
            context.device,
            is_text_model,
            is_recommender_model,
        )
        timings["validation"] = build_time_window(validation_started_at, time.time())

        if spec.variant_config.export_onnx:
            onnx_started_at = time.time()
            onnx_result = export_onnx_model(
                spec,
                model,
                context,
                is_text_model,
                is_recommender_model,
            )
            timings["onnx_export"] = build_time_window(onnx_started_at, time.time())

        if spec.variant_config.run_training:
            training_result = train_model(
                spec,
                model,
                context.device,
            )
            timings["training"] = training_result.timings

        if spec.variant_config.run_inference:
            if training_result is not None:
                wait_for_inference_cooldown(
                    spec.variant_config.pre_inference_cooldown_seconds,
                    context.device,
                )
            inference_result = run_inference(
                spec,
                model,
                context.device,
                is_text_model,
                is_recommender_model,
            )
            timings["inference"] = inference_result.timings

        if spec.variant_config.run_prefill:
            if training_result is not None or inference_result is not None:
                wait_for_inference_cooldown(
                    spec.variant_config.pre_prefill_cooldown_seconds,
                    context.device,
                )
            prefill_result = run_prefill(spec, model, context.device)
            timings["prefill"] = prefill_result.timings

        if spec.variant_config.run_decode:
            if (
                training_result is not None
                or inference_result is not None
                or prefill_result is not None
            ):
                wait_for_inference_cooldown(
                    spec.variant_config.pre_decode_cooldown_seconds,
                    context.device,
                )
            decode_result = run_decode(spec, model, context.device)
            timings["decode"] = decode_result.timings

        timings["full"] = build_time_window(run_started_at, time.time())

        return VariantResult(
            name=spec.name,
            base_model_name=spec.base_model.name,
            base_model_pretrained=spec.base_model.pretrained,
            source=spec.source,
            group_total_variants_defined=spec.group_total_variants_defined,
            variant_config=spec.variant_config.model_dump(mode="json"),
            mutations=[mutation.model_dump(mode="json") for mutation in spec.mutations],
            timings=timings,
            training=training_result,
            inference=inference_result,
            prefill=prefill_result,
            decode=decode_result,
            onnx_export=onnx_result,
            metadata={
                "device": str(context.device),
                "gpu_node": context.gpu_node,
                "model_kind": resolve_model_kind(
                    spec.base_model.name,
                    is_detection_model=is_detection_model,
                    is_recommender_model=is_recommender_model,
                    is_text_model=is_text_model,
                ),
                "parameter_count": count_parameters(model),
                "validation_batch_size": validation_metrics["batch_size"],
                "validation_num_outputs": validation_metrics["num_outputs"],
                "pretrained_weights_loaded": spec.base_model.pretrained,
            },
        )
    finally:
        model = None
        cleanup_workload_boundary(context.device)


def resolve_model_kind(
    model_name: str,
    *,
    is_detection_model: bool,
    is_recommender_model: bool,
    is_text_model: bool,
) -> str:
    if is_detection_model:
        return "detection"
    if is_recommender_model:
        return "recommender"
    if is_causal_lm_model_name(model_name):
        return get_causal_lm_family(model_name)
    if is_text_model:
        return "text"
    return "image"


def cleanup_workload_boundary(device: torch.device) -> None:
    gc.collect()
    if device.type != "cuda" or not torch.cuda.is_available():
        return
    synchronize_device(device)
    torch.cuda.empty_cache()


def synchronize_device(device: torch.device) -> None:
    if device.type != "cuda" or not torch.cuda.is_available():
        return
    torch.cuda.synchronize(device)


def wait_for_inference_cooldown(
    cooldown_seconds: float,
    device: torch.device,
) -> None:
    if cooldown_seconds <= 0:
        return
    synchronize_device(device)
    time.sleep(cooldown_seconds)


def build_variant_model(spec: ResolvedVariantSpec) -> nn.Module:
    if is_detection_model_name(spec.base_model.name):
        assert spec.variant_config.target_output_classes is not None
        from gnn_archs.yolo_builder import build_detection_model

        return build_detection_model(spec)

    if is_recommender_model_name(spec.base_model.name):
        assert spec.variant_config.target_output_classes is not None
        model_name = normalize_model_identifier(spec.base_model.name)
        if model_name == "deepfm":
            from gnn_archs.deepfm_builder import build_deepfm_model

            return build_deepfm_model(spec)
        if model_name == "dcn":
            from gnn_archs.dcn_builder import build_dcn_model

            return build_dcn_model(spec)
        if model_name == "dcnv2":
            from gnn_archs.dcnv2_builder import build_dcnv2_model

            return build_dcnv2_model(spec)
        if model_name == "edcn":
            from gnn_archs.edcn_builder import build_edcn_model

            return build_edcn_model(spec)
        raise ValueError(f"unsupported recommender model: {spec.base_model.name}")

    if is_causal_lm_model_name(spec.base_model.name):
        from gnn_archs.causal_lm_builder import build_causal_lm_variant_model

        return build_causal_lm_variant_model(spec)

    if normalize_model_identifier(spec.base_model.name) == "gpt2":
        assert spec.variant_config.target_output_classes is not None
        from gnn_archs.gpt2_builder import build_gpt2_variant_model

        return build_gpt2_variant_model(spec)

    if normalize_model_identifier(spec.base_model.name) == "t5":
        assert spec.variant_config.target_output_classes is not None
        from gnn_archs.t5_builder import build_t5_variant_model

        return build_t5_variant_model(spec)

    if is_text_model_name(spec.base_model.name):
        assert spec.variant_config.target_output_classes is not None
        validate_mutation_types(spec.mutations, TEXT_MUTATION_TYPES, "text")
        config = build_bert_config(
            base_model=spec.base_model,
            variant_config=spec.variant_config,
        )
        config = apply_text_config_mutations(config, spec.mutations)
        return BertForSequenceClassification(config)

    validate_mutation_types(spec.mutations, IMAGE_MUTATION_TYPES, "image")
    assert spec.variant_config.target_output_classes is not None
    target_input_channels = spec.variant_config.target_input_channels
    if target_input_channels is None:
        raise ValueError("image variants require target_input_channels")

    model = timm.create_model(
        spec.base_model.name,
        pretrained=False,
        in_chans=target_input_channels,
        num_classes=spec.variant_config.target_output_classes,
    )
    return apply_image_mutations(
        model,
        spec.mutations,
        target_output_classes=spec.variant_config.target_output_classes,
        example_input_shape=spec.variant_config.example_input_shape,
    )


def build_bert_config(
    base_model: BaseModelConfig,
    variant_config: VariantConfig,
) -> BertConfig:
    if "bert-large" in base_model.name:
        config = BertConfig(
            hidden_size=1024,
            intermediate_size=4096,
            num_attention_heads=16,
            num_hidden_layers=24,
        )
    else:
        config = BertConfig()

    if variant_config.target_output_classes is None:
        raise ValueError("text variants require target_output_classes")
    config.num_labels = variant_config.target_output_classes
    config.max_position_embeddings = max(
        config.max_position_embeddings,
        variant_config.example_input_shape[1],
    )
    return config


def validate_model(
    spec: ResolvedVariantSpec,
    model: nn.Module,
    device: torch.device,
    is_text_model: bool,
    is_recommender_model: bool = False,
) -> dict[str, int]:
    model.eval()
    with torch.no_grad():
        batch = build_example_batch(spec.variant_config, model, is_text_model)
        batch = {name: tensor.to(device) for name, tensor in batch.items()}
        outputs = forward_model(model, batch, is_text_model, is_recommender_model)
        logits = extract_logits(outputs)

    expected_batch_size = spec.variant_config.example_input_shape[0]
    if logits.shape[0] != expected_batch_size:
        raise ValueError(
            "validation output batch size mismatch: "
            f"{logits.shape[0]} != {expected_batch_size}"
        )
    return {
        "batch_size": expected_batch_size,
        "num_outputs": 1 if is_recommender_model else int(logits.shape[-1]),
    }


def export_onnx_model(
    spec: ResolvedVariantSpec,
    model: nn.Module,
    context: RunContext,
    is_text_model: bool,
    is_recommender_model: bool = False,
) -> OnnxExportResult:
    if is_detection_model_name(spec.base_model.name):
        from gnn_archs.yolo_builder import export_detection_onnx

        return export_detection_onnx(spec, model, context)

    model.eval()
    if is_causal_lm_model_name(spec.base_model.name):
        batch = build_example_batch(spec.variant_config, model, is_text_model)
        batch = {name: tensor.to(context.device) for name, tensor in batch.items()}
        return export_causal_lm_onnx_model(spec, model, context, batch)

    recommender_feature_names: list[str] | None = None
    if is_recommender_model:
        from gnn_archs.recommender.common import get_recommender_feature_names

        recommender_feature_names = get_recommender_feature_names(spec.variant_config)

    export_wrapper = OnnxExportWrapper(
        model,
        is_text_model,
        recommender_feature_names,
    ).to(context.device)
    batch = build_example_batch(spec.variant_config, model, is_text_model)
    batch = {name: tensor.to(context.device) for name, tensor in batch.items()}
    export_path = context.output_layout.onnx_models_dir / f"{spec.name}.onnx"
    opset_version = 14

    if is_text_model:
        args = (batch["input_ids"], batch["attention_mask"])
        input_names = ["input_ids", "attention_mask"]
    elif recommender_feature_names is not None:
        args = tuple(batch[name] for name in recommender_feature_names)
        input_names = recommender_feature_names
    else:
        args = (batch["inputs"],)
        input_names = ["inputs"]

    export_mode = spec.variant_config.onnx_export_mode
    torch.onnx.export(
        export_wrapper,
        args,
        export_path,
        input_names=input_names,
        output_names=["logits"],
        opset_version=opset_version,
        dynamo=False,
        export_params=export_mode == "full",
    )

    onnx_model = onnx.load(export_path)
    set_model_metadata_value(
        onnx_model,
        RUNTIME_INPUT_NAMES_METADATA_KEY,
        json.dumps(input_names),
    )
    set_model_metadata_value(
        onnx_model,
        ONNX_EXPORT_MODE_METADATA_KEY,
        export_mode,
    )
    onnx.save(onnx_model, export_path)
    onnx.checker.check_model(onnx_model)
    graph_input_names = [value.name for value in onnx_model.graph.input]
    parameter_input_names = [
        name for name in graph_input_names if name not in input_names
    ]
    initializer_names = [value.name for value in onnx_model.graph.initializer]
    graph_info: dict[str, int | float | str | list[str]] = {
        "node_count": len(onnx_model.graph.node),
        "input_names": graph_input_names,
        "output_names": [value.name for value in onnx_model.graph.output],
        "op_types": sorted({node.op_type for node in onnx_model.graph.node}),
        "runtime_input_names": input_names,
        "parameter_input_names": parameter_input_names,
        "initializer_names": initializer_names,
        "initializer_count": len(initializer_names),
    }
    return OnnxExportResult(
        path=str(export_path),
        opset_version=opset_version,
        file_size_bytes=export_path.stat().st_size,
        graph_info=graph_info,
    )


def export_causal_lm_onnx_model(
    spec: ResolvedVariantSpec,
    model: nn.Module,
    context: RunContext,
    batch: dict[str, torch.Tensor],
) -> OnnxExportResult:
    export_path = context.output_layout.onnx_models_dir / f"{spec.name}.onnx"
    input_names = ["input_ids", "attention_mask"]
    opset_version = 14
    export_mode = spec.variant_config.onnx_export_mode
    export_model = CausalLMOnnxLogitsExport(model).to(context.device)
    with temporary_causal_lm_export_mode(model):
        torch.onnx.export(
            export_model,
            (batch["input_ids"], batch["attention_mask"]),
            export_path,
            input_names=input_names,
            output_names=["logits"],
            opset_version=opset_version,
            dynamo=False,
            export_params=export_mode == "full",
        )

    onnx_model = onnx.load(export_path)
    set_model_metadata_value(
        onnx_model,
        RUNTIME_INPUT_NAMES_METADATA_KEY,
        json.dumps(input_names),
    )
    set_model_metadata_value(
        onnx_model,
        ONNX_EXPORT_MODE_METADATA_KEY,
        export_mode,
    )
    onnx.save(onnx_model, export_path)
    onnx.checker.check_model(onnx_model)
    graph_input_names = [value.name for value in onnx_model.graph.input]
    parameter_input_names = [
        name for name in graph_input_names if name not in input_names
    ]
    initializer_names = [value.name for value in onnx_model.graph.initializer]
    graph_info: dict[str, int | float | str | list[str]] = {
        "node_count": len(onnx_model.graph.node),
        "input_names": graph_input_names,
        "output_names": [value.name for value in onnx_model.graph.output],
        "op_types": sorted({node.op_type for node in onnx_model.graph.node}),
        "runtime_input_names": input_names,
        "parameter_input_names": parameter_input_names,
        "initializer_names": initializer_names,
        "initializer_count": len(initializer_names),
    }
    return OnnxExportResult(
        path=str(export_path),
        opset_version=opset_version,
        file_size_bytes=export_path.stat().st_size,
        graph_info=graph_info,
    )


@contextmanager
def temporary_causal_lm_export_mode(model: nn.Module) -> Iterator[None]:
    previous_attention_values: list[tuple[Any, Any]] = []
    previous_cache_values: list[tuple[Any, Any]] = []
    for config in iter_causal_lm_export_configs(model):
        if hasattr(config, "use_cache"):
            previous_cache_values.append((config, config.use_cache))
            config.use_cache = False
        if hasattr(config, "_attn_implementation"):
            previous_attention_values.append((config, config._attn_implementation))
            config._attn_implementation = "eager"
    generation_config = getattr(model, "generation_config", None)
    if generation_config is not None and hasattr(generation_config, "use_cache"):
        previous_cache_values.append((generation_config, generation_config.use_cache))
        generation_config.use_cache = False
    try:
        yield
    finally:
        for config, value in previous_attention_values:
            config._attn_implementation = value
        for config, value in previous_cache_values:
            config.use_cache = value


def iter_causal_lm_export_configs(model: nn.Module) -> tuple[Any, ...]:
    config = getattr(model, "config", None)
    if config is None:
        return ()
    text_config = getattr(config, "text_config", None)
    if text_config is None:
        return (config,)
    return (config, text_config)


def train_model(
    spec: ResolvedVariantSpec,
    model: nn.Module,
    device: torch.device,
) -> TrainingResult:
    if is_detection_model_name(spec.base_model.name):
        from gnn_archs.yolo_builder import train_detection_model

        return train_detection_model(spec, model, device)

    is_recommender_model = is_recommender_model_name(spec.base_model.name)
    if is_recommender_model:
        if not spec.variant_config.use_fake_recommender_dataset:
            raise NotImplementedError(
                "real recommender dataset preparation is not migrated yet"
            )
    elif is_text_model_name(spec.base_model.name):
        if not spec.variant_config.use_fake_text_dataset:
            raise NotImplementedError(
                "real text dataset preparation is not migrated yet"
            )
    elif not spec.variant_config.use_fake_imagenet:
        raise NotImplementedError("real image dataset preparation is not migrated yet")

    is_text_model = is_text_model_name(spec.base_model.name)
    batch_size = spec.variant_config.batch_size
    generator = torch.Generator().manual_seed(42)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    criterion: nn.Module = (
        nn.BCELoss() if is_recommender_model else nn.CrossEntropyLoss()
    )

    measurement_min_seconds = spec.variant_config.training_measurement_min_seconds
    total_steps = 0
    last_loss = 0.0

    synchronize_device(device)
    training_started_at = time.time()
    training_ended_at = training_started_at

    model.train()
    while (
        total_steps == 0
        or training_ended_at - training_started_at < measurement_min_seconds
    ):
        batch = build_training_batch(
            spec=spec,
            model=model,
            batch_size=batch_size,
            generator=generator,
        )
        optimizer.zero_grad(set_to_none=True)

        if is_recommender_model:
            feature_batch, labels = batch
            feature_batch = {
                name: tensor.to(device) for name, tensor in feature_batch.items()
            }
            labels = labels.to(device)
            scores = extract_logits(model(feature_batch))
            loss = criterion(scores, labels)
        elif is_text_model:
            input_ids, attention_mask, labels = [tensor.to(device) for tensor in batch]
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = outputs.loss
        else:
            inputs, labels = [tensor.to(device) for tensor in batch]
            logits = extract_logits(model(inputs))
            loss = criterion(logits, labels)

        loss.backward()
        optimizer.step()

        total_steps += 1
        last_loss = float(loss.detach().item())
        synchronize_device(device)
        training_ended_at = time.time()

    return TrainingResult(
        hyperparameters={
            "batch_size": batch_size,
            "measurement_min_seconds": measurement_min_seconds,
        },
        optimizer={
            "name": "AdamW",
            "lr": 1e-3,
        },
        metrics={
            "final_loss": last_loss,
            "total_steps": total_steps,
        },
        timings=build_time_window(training_started_at, training_ended_at),
    )


def run_inference(
    spec: ResolvedVariantSpec,
    model: nn.Module,
    device: torch.device,
    is_text_model: bool,
    is_recommender_model: bool = False,
) -> InferenceResult:
    if is_detection_model_name(spec.base_model.name):
        from gnn_archs.yolo_builder import run_detection_inference

        return run_detection_inference(spec, model, device)

    batch = build_example_batch(spec.variant_config, model, is_text_model)
    batch = {name: tensor.to(device) for name, tensor in batch.items()}
    measurement_min_seconds = spec.variant_config.inference_measurement_min_seconds
    iterations = 0

    model.eval()
    with torch.no_grad():
        for _ in range(2):
            _ = forward_model(model, batch, is_text_model, is_recommender_model)

        synchronize_device(device)

        started_at = time.time()
        ended_at = started_at
        outputs: Any | None = None
        while ended_at - started_at < measurement_min_seconds:
            outputs = forward_model(model, batch, is_text_model, is_recommender_model)
            synchronize_device(device)
            iterations += 1
            ended_at = time.time()

    assert outputs is not None
    logits = extract_logits(outputs)
    total_duration = ended_at - started_at
    return InferenceResult(
        metrics={
            "iterations": iterations,
            "batch_size": spec.variant_config.example_input_shape[0],
            "avg_latency_ms": round(total_duration / iterations * 1000, 4),
            "num_outputs": 1 if is_recommender_model else int(logits.shape[-1]),
        },
        timings=build_time_window(started_at, ended_at),
    )


def run_prefill(
    spec: ResolvedVariantSpec,
    model: nn.Module,
    device: torch.device,
) -> InferenceResult:
    if not is_causal_lm_model_name(spec.base_model.name):
        raise ValueError("prefill phase is only supported for causal lm variants")

    batch = build_example_batch(spec.variant_config, model, True)
    batch = {name: tensor.to(device) for name, tensor in batch.items()}
    measurement_min_seconds = spec.variant_config.prefill_measurement_min_seconds
    iterations = 0

    model.eval()
    with torch.no_grad():
        for _ in range(2):
            _ = forward_model(model, batch, True)

        synchronize_device(device)

        started_at = time.time()
        ended_at = started_at
        outputs: Any | None = None
        while ended_at - started_at < measurement_min_seconds:
            outputs = forward_model(model, batch, True)
            synchronize_device(device)
            iterations += 1
            ended_at = time.time()

    assert outputs is not None
    logits = extract_logits(outputs)
    total_duration = ended_at - started_at
    return InferenceResult(
        metrics={
            "iterations": iterations,
            "batch_size": spec.variant_config.example_input_shape[0],
            "sequence_length": spec.variant_config.example_input_shape[1],
            "avg_latency_ms": round(total_duration / iterations * 1000, 4),
            "num_outputs": int(logits.shape[-1]),
        },
        timings=build_time_window(started_at, ended_at),
    )


def run_decode(
    spec: ResolvedVariantSpec,
    model: nn.Module,
    device: torch.device,
) -> InferenceResult:
    if not is_causal_lm_model_name(spec.base_model.name):
        raise ValueError("decode phase is only supported for causal lm variants")

    batch = build_example_batch(spec.variant_config, model, True)
    batch = {name: tensor.to(device) for name, tensor in batch.items()}
    measurement_min_seconds = spec.variant_config.decode_measurement_min_seconds
    max_output_length = spec.variant_config.decode_max_output_length
    iterations = 0

    model.eval()
    with torch.no_grad():
        for _ in range(2):
            _ = generate_decode_batch(model, batch, max_output_length)

        synchronize_device(device)

        started_at = time.time()
        ended_at = started_at
        generated: torch.Tensor | None = None
        while ended_at - started_at < measurement_min_seconds:
            generated = generate_decode_batch(model, batch, max_output_length)
            synchronize_device(device)
            iterations += 1
            ended_at = time.time()

    assert generated is not None
    prompt_length = spec.variant_config.example_input_shape[1]
    generated_output_length = int(generated.shape[1] - prompt_length)
    total_duration = ended_at - started_at
    return InferenceResult(
        metrics={
            "iterations": iterations,
            "batch_size": spec.variant_config.example_input_shape[0],
            "sequence_length": prompt_length,
            "decode_max_output_length": max_output_length,
            "generated_output_length": generated_output_length,
            "avg_latency_ms": round(total_duration / iterations * 1000, 4),
        },
        timings=build_time_window(started_at, ended_at),
    )


def generate_decode_batch(
    model: nn.Module,
    batch: dict[str, torch.Tensor],
    max_output_length: int,
) -> torch.Tensor:
    generated = cast(CausalLMGenerator, model).generate(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        max_new_tokens=max_output_length,
        do_sample=False,
        num_beams=1,
        use_cache=True,
        pad_token_id=0,
        eos_token_id=None,
    )
    assert isinstance(generated, torch.Tensor)
    return generated


def build_training_batch(
    spec: ResolvedVariantSpec,
    model: nn.Module,
    batch_size: int,
    generator: torch.Generator,
) -> tuple[Any, ...]:
    if is_recommender_model_name(spec.base_model.name):
        output_classes = spec.variant_config.target_output_classes
        assert output_classes is not None
        from gnn_archs.recommender.common import build_recommender_batch

        return (
            build_recommender_batch(spec.variant_config, batch_size, generator),
            torch.randint(0, 2, (batch_size,), generator=generator).float(),
        )

    if is_causal_lm_model_name(spec.base_model.name):
        sequence_length = spec.variant_config.example_input_shape[1]
        input_ids = build_text_input_ids(
            model,
            batch_size,
            sequence_length,
            generator,
        )
        attention_mask = torch.ones((batch_size, sequence_length), dtype=torch.long)
        return input_ids, attention_mask, input_ids.clone()

    if is_text_model_name(spec.base_model.name):
        output_classes = spec.variant_config.target_output_classes
        assert output_classes is not None
        sequence_length = spec.variant_config.example_input_shape[1]
        return (
            build_text_input_ids(model, batch_size, sequence_length, generator),
            torch.ones((batch_size, sequence_length), dtype=torch.long),
            torch.randint(0, output_classes, (batch_size,), generator=generator),
        )

    output_classes = spec.variant_config.target_output_classes
    assert output_classes is not None
    _, channels, height, width = spec.variant_config.example_input_shape
    return (
        torch.randn((batch_size, channels, height, width), generator=generator),
        torch.randint(
            0,
            output_classes,
            (batch_size,),
            generator=generator,
        ),
    )


def build_example_batch(
    variant_config: VariantConfig,
    model: nn.Module,
    is_text_model: bool,
) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(42)

    if is_text_model:
        batch_size, sequence_length = variant_config.example_input_shape
        if not is_t5_model(model):
            max_position_embeddings = get_model_config_int(
                model,
                "max_position_embeddings",
            )
            if sequence_length > max_position_embeddings:
                raise ValueError(
                    "example_input_shape sequence length exceeds "
                    "the model max_position_embeddings"
                )
        return {
            "input_ids": build_text_input_ids(
                model,
                batch_size,
                sequence_length,
                generator,
            ),
            "attention_mask": torch.ones(
                (batch_size, sequence_length),
                dtype=torch.long,
            ),
        }

    batch_size, channels, height, width = variant_config.example_input_shape
    return {
        "inputs": torch.randn(
            (batch_size, channels, height, width),
            generator=generator,
        )
    }


def is_t5_model(model: nn.Module) -> bool:
    return getattr(getattr(model, "config", None), "model_type", None) == "t5"


def build_text_input_ids(
    model: nn.Module,
    batch_size: int,
    sequence_length: int,
    generator: torch.Generator,
) -> torch.Tensor:
    vocab_size = get_model_config_int(model, "vocab_size")
    if is_t5_model(model):
        return build_t5_input_ids(batch_size, sequence_length, vocab_size, generator)
    return build_bert_like_text_input_ids(
        batch_size,
        sequence_length,
        vocab_size,
        generator,
    )


def get_model_config_int(model: nn.Module, field_name: str) -> int:
    config = getattr(model, "config", None)
    value = getattr(config, field_name, None)
    if not isinstance(value, int):
        text_config = getattr(config, "text_config", None)
        value = getattr(text_config, field_name, None)
    if not isinstance(value, int):
        raise ValueError(f"model config {field_name} must be an integer")
    return value


def build_bert_like_text_input_ids(
    batch_size: int,
    sequence_length: int,
    vocab_size: int,
    generator: torch.Generator,
) -> torch.Tensor:
    return torch.randint(
        0,
        vocab_size,
        (batch_size, sequence_length),
        generator=generator,
    )


def build_t5_input_ids(
    batch_size: int,
    sequence_length: int,
    vocab_size: int,
    generator: torch.Generator,
) -> torch.Tensor:
    input_ids = torch.randint(
        2,
        vocab_size,
        (batch_size, sequence_length),
        generator=generator,
    )
    input_ids[:, -1] = 1
    return input_ids


def forward_model(
    model: nn.Module,
    batch: dict[str, torch.Tensor],
    is_text_model: bool,
    is_recommender_model: bool = False,
) -> Any:
    if is_recommender_model:
        return model(batch)
    if is_text_model:
        return model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
        )
    return model(batch["inputs"])


def extract_logits(outputs: Any) -> torch.Tensor:
    if hasattr(outputs, "logits"):
        return outputs.logits
    if isinstance(outputs, tuple):
        return outputs[0]
    if isinstance(outputs, torch.Tensor):
        return outputs
    raise TypeError(f"unsupported model output type: {type(outputs)}")


def summarize_variant_results(variant_results: list[VariantResult]) -> dict[str, int]:
    return {
        "variant_count": len(variant_results),
        "training_count": sum(
            result.training is not None for result in variant_results
        ),
        "inference_count": sum(
            result.inference is not None for result in variant_results
        ),
        "prefill_count": sum(result.prefill is not None for result in variant_results),
        "decode_count": sum(result.decode is not None for result in variant_results),
        "onnx_export_count": sum(
            result.onnx_export is not None for result in variant_results
        ),
    }


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def build_time_window(started_at_ts: float, ended_at_ts: float) -> TimeWindow:
    return TimeWindow(
        started_at_ts=started_at_ts,
        ended_at_ts=ended_at_ts,
        started_at_text=format_timestamp(started_at_ts),
        ended_at_text=format_timestamp(ended_at_ts),
    )


def format_timestamp(timestamp: float) -> str:
    return datetime.fromtimestamp(timestamp, tz=UTC).isoformat()
