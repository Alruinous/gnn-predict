from __future__ import annotations

import errno
import gc
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Protocol, cast

import timm
import torch
import torch.nn as nn
from transformers import BertConfig, BertForSequenceClassification

from common.graph_artifact import clear_graph_capture_caches
from gnn_archs.config import (
    get_causal_lm_family,
    is_causal_lm_model_name,
    is_detection_model_name,
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
    GraphExportResult,
    InferenceResult,
    TimeWindow,
    TrainingResult,
    VariantFailure,
    VariantResult,
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
    fx_graphs_dir: Path


@dataclass(frozen=True)
class RunContext:
    config_path: Path
    output_layout: OutputLayout
    device: torch.device
    gpu_node: str
    logger: logging.Logger
    device_backend: str = "unknown"
    device_name: str = "unknown"
    stage_tracker: Callable[[str], None] | None = None


class CausalLMGenerator(Protocol):
    def generate(self, **kwargs: Any) -> torch.Tensor: ...


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
        fx_graphs_dir=config_output_root / "fx_graphs",
    )
    for directory in (
        layout.root,
        layout.logs_dir,
        layout.results_dir,
        layout.fx_graphs_dir,
    ):
        directory.mkdir(parents=True, exist_ok=True)
    return layout


def run_variant(spec: ResolvedVariantSpec, context: RunContext) -> VariantResult:
    run_started_at = time.time()
    model: nn.Module | None = None
    try:
        _record_stage(context, "model_build")
        model, model_build_timing = build_model_for_run(spec, context.device)
        return execute_variant(
            spec,
            context,
            model,
            model_build_timing=model_build_timing,
            run_started_at=run_started_at,
            model_build_reused=False,
        )
    finally:
        model = None
        try:
            cleanup_workload_boundary(context.device)
        except Exception:
            _record_stage(context, "cleanup")
            raise


def run_variants(
    specs: Sequence[ResolvedVariantSpec],
    context: RunContext,
    *,
    continue_on_error: bool = False,
    on_progress: (
        Callable[[list[VariantResult], list[VariantFailure]], None] | None
    ) = None,
) -> list[VariantResult]:
    results: list[VariantResult] = []
    failures: list[VariantFailure] = []
    reusable_model: nn.Module | None = None
    reusable_key: tuple[str, str, str] | None = None
    reusable_build_timing: TimeWindow | None = None
    try:
        for index, spec in enumerate(specs, start=1):
            context.logger.info(
                "processing variant %s/%s: %s",
                index,
                len(specs),
                spec.name,
            )
            stage = "variant_setup"

            def track_stage(value: str) -> None:
                nonlocal stage
                stage = value

            variant_context = replace(context, stage_tracker=track_stage)
            try:
                reuse_key = resolve_model_reuse_key(spec)
                if reuse_key is None:
                    if reusable_model is not None:
                        reusable_model = None
                        reusable_key = None
                        reusable_build_timing = None
                        track_stage("cleanup")
                        cleanup_workload_boundary(context.device)
                    result = run_variant(spec, variant_context)
                else:
                    model_build_reused = (
                        reusable_model is not None and reuse_key == reusable_key
                    )
                    if not model_build_reused:
                        if reusable_model is not None:
                            reusable_model = None
                            reusable_build_timing = None
                            track_stage("cleanup")
                            cleanup_workload_boundary(context.device)
                        run_started_at = time.time()
                        track_stage("model_build")
                        reusable_model, reusable_build_timing = build_model_for_run(
                            spec,
                            context.device,
                        )
                        reusable_key = reuse_key
                    else:
                        run_started_at = time.time()

                    assert reusable_model is not None
                    assert reusable_build_timing is not None
                    try:
                        result = execute_variant(
                            spec,
                            variant_context,
                            reusable_model,
                            model_build_timing=reusable_build_timing,
                            run_started_at=run_started_at,
                            model_build_reused=model_build_reused,
                        )
                    finally:
                        try:
                            cleanup_workload_boundary(context.device)
                        except Exception:
                            track_stage("cleanup")
                            raise
            except Exception as exc:
                failure = VariantFailure(
                    name=spec.name,
                    stage=stage,
                    error_type=type(exc).__name__,
                    message=str(exc),
                )
                failures.append(failure)
                context.logger.exception(
                    "variant %s failed during %s", spec.name, stage
                )
                # A reused model may be partially mutated after a failed variant.
                # Never carry it into the next variant.
                reusable_model = None
                reusable_key = None
                reusable_build_timing = None
                if on_progress is not None:
                    on_progress(list(results), list(failures))
                if not continue_on_error or _is_fatal_variant_error(exc):
                    raise
                # If the accelerator cannot synchronize, this is a device failure,
                # not an isolated variant failure; do not skip the remaining work.
                cleanup_workload_boundary(context.device)
                _verify_device_after_failure(context.device)
                continue

            results.append(result)
            if on_progress is not None:
                on_progress(list(results), list(failures))
        return results
    finally:
        reusable_model = None
        cleanup_workload_boundary(context.device)


def _record_stage(context: RunContext, stage: str) -> None:
    if context.stage_tracker is not None:
        context.stage_tracker(stage)


def _is_fatal_variant_error(exc: Exception) -> bool:
    return isinstance(exc, OSError) and exc.errno in {
        errno.EACCES,
        errno.ENOSPC,
        errno.EROFS,
    }


def _verify_device_after_failure(device: torch.device) -> None:
    if device.type != "cuda":
        return
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA-compatible device is unavailable after variant failure"
        )
    torch.cuda.synchronize(device)


def resolve_model_reuse_key(
    spec: ResolvedVariantSpec,
) -> tuple[str, str, str] | None:
    if spec.variant_config.run_training or spec.base_model.pretrained or spec.mutations:
        return None
    qwen3_config = spec.variant_config.qwen3_config
    if qwen3_config is not None:
        return (
            normalize_model_identifier(spec.base_model.name),
            "qwen3",
            qwen3_config.model_dump_json(),
        )
    gemma4_config = spec.variant_config.gemma4_config
    if gemma4_config is not None:
        return (
            normalize_model_identifier(spec.base_model.name),
            "gemma4",
            gemma4_config.model_dump_json(),
        )
    return None


def build_model_for_run(
    spec: ResolvedVariantSpec,
    device: torch.device,
) -> tuple[nn.Module, TimeWindow]:
    model_build_started_at = time.time()
    model = build_variant_model(spec, device)
    model = model.to(device)
    return model, build_time_window(model_build_started_at, time.time())


def execute_variant(
    spec: ResolvedVariantSpec,
    context: RunContext,
    model: nn.Module,
    *,
    model_build_timing: TimeWindow,
    run_started_at: float,
    model_build_reused: bool,
) -> VariantResult:
    timings = {"model_build": model_build_timing}
    is_text_model = is_text_model_name(spec.base_model.name)
    is_detection_model = is_detection_model_name(spec.base_model.name)
    graph_result: GraphExportResult | None = None
    decode_graph_result: GraphExportResult | None = None
    training_result: TrainingResult | None = None
    inference_result: InferenceResult | None = None
    prefill_result: InferenceResult | None = None
    decode_result: InferenceResult | None = None

    _record_stage(context, "validation")
    validation_started_at = time.time()
    validation_metrics = validate_model(
        spec,
        model,
        context.device,
        is_text_model,
    )
    timings["validation"] = build_time_window(validation_started_at, time.time())

    if spec.variant_config.export_graph:
        _record_stage(context, "graph_export")
        from gnn_archs.graph_export import (
            export_causal_lm_decode_graph,
            export_model_graph,
        )

        graph_started_at = time.time()
        graph_result = export_model_graph(
            spec,
            model,
            context,
            is_text_model=is_text_model,
        )
        if (
            is_causal_lm_model_name(spec.base_model.name)
            and spec.variant_config.run_decode
        ):
            decode_graph_result = export_causal_lm_decode_graph(
                spec,
                model,
                context,
            )
        timings["graph_export"] = build_time_window(
            graph_started_at,
            time.time(),
        )

    if spec.variant_config.run_training:
        _record_stage(context, "training")
        training_result = train_model(
            spec,
            model,
            context.device,
        )
        timings["training"] = training_result.timings

    if spec.variant_config.run_inference:
        _record_stage(context, "inference")
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

    if spec.variant_config.run_prefill:
        _record_stage(context, "prefill")
        if training_result is not None or inference_result is not None:
            wait_for_inference_cooldown(
                spec.variant_config.pre_prefill_cooldown_seconds,
                context.device,
            )
        prefill_result = run_prefill(spec, model, context.device)
        timings["prefill"] = prefill_result.timings

    if spec.variant_config.run_decode:
        _record_stage(context, "decode")
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

    _record_stage(context, "result_serialization")
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
        graph_export=graph_result,
        decode_graph_export=decode_graph_result,
        metadata={
            "device": str(context.device),
            "device_backend": context.device_backend,
            "device_name": context.device_name,
            "gpu_node": context.gpu_node,
            "model_kind": resolve_model_kind(
                spec.base_model.name,
                is_detection_model=is_detection_model,
                is_text_model=is_text_model,
            ),
            "parameter_count": count_parameters(model),
            "validation_batch_size": validation_metrics["batch_size"],
            "validation_num_outputs": validation_metrics["num_outputs"],
            "pretrained_weights_loaded": spec.base_model.pretrained,
            "model_build_reused": model_build_reused,
        },
    )


def resolve_model_kind(
    model_name: str,
    *,
    is_detection_model: bool,
    is_text_model: bool,
) -> str:
    if is_detection_model:
        return "detection"
    if is_causal_lm_model_name(model_name):
        return get_causal_lm_family(model_name)
    if is_text_model:
        return "text"
    return "image"


def cleanup_workload_boundary(device: torch.device) -> None:
    clear_graph_capture_caches()
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


def build_variant_model(
    spec: ResolvedVariantSpec,
    device: torch.device | None = None,
) -> nn.Module:
    if is_detection_model_name(spec.base_model.name):
        assert spec.variant_config.target_output_classes is not None
        from gnn_archs.yolo_builder import build_detection_model

        return build_detection_model(spec)

    if is_causal_lm_model_name(spec.base_model.name):
        from gnn_archs.causal_lm_builder import build_causal_lm_variant_model

        return build_causal_lm_variant_model(spec, device)

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


def train_model(
    spec: ResolvedVariantSpec,
    model: nn.Module,
    device: torch.device,
) -> TrainingResult:
    if is_detection_model_name(spec.base_model.name):
        from gnn_archs.yolo_builder import train_detection_model

        return train_detection_model(spec, model, device)

    if is_text_model_name(spec.base_model.name):
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
    criterion: nn.Module = nn.CrossEntropyLoss()

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

        if is_text_model:
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


def summarize_variant_results(
    variant_results: list[VariantResult],
) -> dict[str, int | float | str]:
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
        "graph_export_count": sum(
            result.graph_export is not None for result in variant_results
        ),
        "decode_graph_export_count": sum(
            result.decode_graph_export is not None for result in variant_results
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
    return datetime.fromtimestamp(timestamp, tz=timezone.utc).isoformat()
