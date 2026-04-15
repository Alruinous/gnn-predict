from __future__ import annotations

import gc
import json
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

import onnx
import timm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from transformers import BertConfig, BertForSequenceClassification

from gnn_archs.config import is_text_model_name
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


class OnnxExportWrapper(nn.Module):
    def __init__(self, model: nn.Module, is_text_model: bool) -> None:
        super().__init__()
        self.model = model
        self.is_text_model = is_text_model

    def forward(self, *inputs: torch.Tensor) -> torch.Tensor:
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
    model: nn.Module | None = None
    onnx_result: OnnxExportResult | None = None
    training_result: TrainingResult | None = None
    inference_result: InferenceResult | None = None
    try:
        model_build_started_at = time.time()
        model = build_variant_model(spec)
        model = model.to(context.device)
        timings["model_build"] = build_time_window(model_build_started_at, time.time())

        validation_started_at = time.time()
        validation_metrics = validate_model(spec, model, context.device, is_text_model)
        timings["validation"] = build_time_window(validation_started_at, time.time())

        if spec.variant_config.export_onnx:
            onnx_started_at = time.time()
            onnx_result = export_onnx_model(spec, model, context, is_text_model)
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
            )
            timings["inference"] = inference_result.timings

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
            onnx_export=onnx_result,
            metadata={
                "device": str(context.device),
                "gpu_node": context.gpu_node,
                "model_kind": "text" if is_text_model else "image",
                "parameter_count": count_parameters(model),
                "validation_batch_size": validation_metrics["batch_size"],
                "validation_num_outputs": validation_metrics["num_outputs"],
                "pretrained_weights_loaded": False,
            },
        )
    finally:
        model = None
        cleanup_workload_boundary(context.device)


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
    assert spec.variant_config.target_output_classes is not None

    if is_text_model_name(spec.base_model.name):
        validate_mutation_types(spec.mutations, TEXT_MUTATION_TYPES, "text")
        config = build_bert_config(
            base_model=spec.base_model,
            variant_config=spec.variant_config,
        )
        config = apply_text_config_mutations(config, spec.mutations)
        return BertForSequenceClassification(config)

    validate_mutation_types(spec.mutations, IMAGE_MUTATION_TYPES, "image")
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
        variant_config.max_sequence_length,
        variant_config.example_input_shape[1],
    )
    return config


def validate_model(
    spec: ResolvedVariantSpec,
    model: nn.Module,
    device: torch.device,
    is_text_model: bool,
) -> dict[str, int]:
    model.eval()
    with torch.no_grad():
        batch = build_example_batch(spec.variant_config, model, is_text_model)
        batch = {name: tensor.to(device) for name, tensor in batch.items()}
        outputs = forward_model(model, batch, is_text_model)
        logits = extract_logits(outputs)

    expected_batch_size = spec.variant_config.example_input_shape[0]
    if logits.shape[0] != expected_batch_size:
        raise ValueError(
            "validation output batch size mismatch: "
            f"{logits.shape[0]} != {expected_batch_size}"
        )
    return {
        "batch_size": expected_batch_size,
        "num_outputs": int(logits.shape[-1]),
    }


def export_onnx_model(
    spec: ResolvedVariantSpec,
    model: nn.Module,
    context: RunContext,
    is_text_model: bool,
) -> OnnxExportResult:
    model.eval()
    export_wrapper = OnnxExportWrapper(model, is_text_model).to(context.device)
    batch = build_example_batch(spec.variant_config, model, is_text_model)
    batch = {name: tensor.to(context.device) for name, tensor in batch.items()}
    export_path = context.output_layout.onnx_models_dir / f"{spec.name}.onnx"
    opset_version = 14

    if is_text_model:
        args = (batch["input_ids"], batch["attention_mask"])
        input_names = ["input_ids", "attention_mask"]
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
    graph_info = {
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


def train_model(
    spec: ResolvedVariantSpec,
    model: nn.Module,
    device: torch.device,
) -> TrainingResult:
    if is_text_model_name(spec.base_model.name):
        if not spec.variant_config.use_fake_text_dataset:
            raise NotImplementedError(
                "real text dataset preparation is not migrated yet"
            )
    elif not spec.variant_config.use_fake_imagenet:
        raise NotImplementedError("real image dataset preparation is not migrated yet")

    batch_size = spec.variant_config.training_batch_sizes[0]
    dataset = build_training_dataset(spec, model)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    total_steps = 0
    last_loss = 0.0
    training_started_at = time.time()

    model.train()
    for _epoch_index in range(spec.variant_config.training_epochs):
        for batch in dataloader:
            optimizer.zero_grad(set_to_none=True)

            if is_text_model_name(spec.base_model.name):
                input_ids, attention_mask, labels = [
                    tensor.to(device) for tensor in batch
                ]
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

    return TrainingResult(
        hyperparameters={
            "batch_size": batch_size,
            "epochs": spec.variant_config.training_epochs,
            "dataset_size": len(dataset),
        },
        optimizer={
            "name": "AdamW",
            "lr": 1e-3,
        },
        metrics={
            "final_loss": last_loss,
            "total_steps": total_steps,
        },
        timings=build_time_window(training_started_at, time.time()),
    )


def run_inference(
    spec: ResolvedVariantSpec,
    model: nn.Module,
    device: torch.device,
    is_text_model: bool,
) -> InferenceResult:
    batch = build_example_batch(spec.variant_config, model, is_text_model)
    batch = {name: tensor.to(device) for name, tensor in batch.items()}
    measurement_min_seconds = spec.variant_config.inference_measurement_min_seconds
    iterations = 0

    model.eval()
    with torch.no_grad():
        for _ in range(2):
            _ = forward_model(model, batch, is_text_model)

        synchronize_device(device)

        started_at = time.time()
        ended_at = started_at
        outputs: Any | None = None
        while ended_at - started_at < measurement_min_seconds:
            outputs = forward_model(model, batch, is_text_model)
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
            "num_outputs": int(logits.shape[-1]),
        },
        timings=build_time_window(started_at, ended_at),
    )


def build_training_dataset(
    spec: ResolvedVariantSpec,
    model: nn.Module,
) -> TensorDataset:
    dataset_size = spec.variant_config.fake_dataset_size
    generator = torch.Generator().manual_seed(42)

    if is_text_model_name(spec.base_model.name):
        sequence_length = spec.variant_config.example_input_shape[1]
        vocab_size = int(model.config.vocab_size)  # type: ignore[attr-defined]
        labels = torch.randint(
            0,
            spec.variant_config.target_output_classes,
            (dataset_size,),
            generator=generator,
        )
        return TensorDataset(
            torch.randint(
                0,
                vocab_size,
                (dataset_size, sequence_length),
                generator=generator,
            ),
            torch.ones(
                (dataset_size, sequence_length),
                dtype=torch.long,
            ),
            labels,
        )

    _, channels, height, width = spec.variant_config.example_input_shape
    return TensorDataset(
        torch.randn((dataset_size, channels, height, width), generator=generator),
        torch.randint(
            0,
            spec.variant_config.target_output_classes,
            (dataset_size,),
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
        max_position_embeddings = int(model.config.max_position_embeddings)  # type: ignore[attr-defined]
        if sequence_length > max_position_embeddings:
            raise ValueError(
                "example_input_shape sequence length exceeds "
                "the model max_position_embeddings"
            )
        vocab_size = int(model.config.vocab_size)  # type: ignore[attr-defined]
        return {
            "input_ids": torch.randint(
                0,
                vocab_size,
                (batch_size, sequence_length),
                generator=generator,
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


def forward_model(
    model: nn.Module, batch: dict[str, torch.Tensor], is_text_model: bool
) -> Any:
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
