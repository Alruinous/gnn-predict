from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
import torch.nn as nn
from torch.export import ExportedProgram
from transformers import T5ForSequenceClassification

from common.graph_artifact import (
    GraphArtifactMetadata,
    build_graph_info,
    capture_inference_graph,
    save_graph_artifact,
)
from gnn_archs.config import is_causal_lm_model_name, is_detection_model_name
from gnn_archs.result import GraphExportResult
from gnn_archs.t5_builder import T5SequenceClassificationGraph

if TYPE_CHECKING:
    from gnn_archs.config import ResolvedVariantSpec
    from gnn_archs.variant_runner import RunContext


class ModelInferenceGraph(nn.Module):
    def __init__(self, model: nn.Module, *, is_text_model: bool) -> None:
        super().__init__()
        self.model = model
        self.is_text_model = is_text_model

    def forward(self, *inputs: torch.Tensor) -> torch.Tensor:
        if self.is_text_model:
            outputs = self.model(input_ids=inputs[0], attention_mask=inputs[1])
        else:
            outputs = self.model(inputs[0])
        return extract_graph_logits(outputs)


def export_model_graph(
    spec: ResolvedVariantSpec,
    model: nn.Module,
    context: RunContext,
    *,
    is_text_model: bool,
) -> GraphExportResult:
    if is_detection_model_name(spec.base_model.name):
        from gnn_archs.yolo_builder import export_detection_graph

        return export_detection_graph(spec, model, context)
    if is_causal_lm_model_name(spec.base_model.name):
        return export_causal_lm_prefill_graph(spec, model, context)

    from gnn_archs.variant_runner import build_example_batch

    batch = build_example_batch(spec.variant_config, model, is_text_model)
    batch = {name: tensor.to(context.device) for name, tensor in batch.items()}
    if is_text_model:
        args = (batch["input_ids"], batch["attention_mask"])
        runtime_input_names = ["input_ids", "attention_mask"]
        export_model: nn.Module = (
            T5SequenceClassificationGraph(model)
            if isinstance(model, T5ForSequenceClassification)
            else ModelInferenceGraph(model, is_text_model=True)
        )
    else:
        args = (batch["inputs"],)
        runtime_input_names = ["inputs"]
        export_model = ModelInferenceGraph(model, is_text_model=False)
    exported_program = capture_inference_graph(
        export_model.to(context.device),
        args,
    )
    export_path = context.output_layout.fx_graphs_dir / f"{spec.name}.pt2"
    result = write_graph_export(exported_program, export_path, runtime_input_names)
    del exported_program, export_model, args, batch
    return result


def export_causal_lm_prefill_graph(
    spec: ResolvedVariantSpec,
    model: nn.Module,
    context: RunContext,
) -> GraphExportResult:
    from gnn_archs.causal_lm_builder import (
        CausalLMPrefillGraph,
        temporary_causal_lm_graph_mode,
    )
    from gnn_archs.variant_runner import build_example_batch

    batch = build_example_batch(spec.variant_config, model, True)
    batch = {name: tensor.to(context.device) for name, tensor in batch.items()}
    runtime_input_names = ["input_ids", "attention_mask"]
    export_model = CausalLMPrefillGraph(model).to(context.device)
    with temporary_causal_lm_graph_mode(model):
        exported_program = capture_inference_graph(
            export_model,
            (batch["input_ids"], batch["attention_mask"]),
        )
    export_path = context.output_layout.fx_graphs_dir / f"{spec.name}_prefill.pt2"
    result = write_graph_export(exported_program, export_path, runtime_input_names)
    del exported_program, export_model, batch
    return result


def export_causal_lm_decode_graph(
    spec: ResolvedVariantSpec,
    model: nn.Module,
    context: RunContext,
) -> GraphExportResult:
    from gnn_archs.causal_lm_builder import (
        CausalLMDecodeGraph,
        build_causal_lm_kv_input_names,
        collect_causal_lm_kv_pairs,
        run_causal_lm_dry_prefill,
        temporary_causal_lm_graph_mode,
    )
    from gnn_archs.variant_runner import build_text_input_ids

    batch_size, sequence_length = spec.variant_config.example_input_shape
    output_length = spec.variant_config.decode_max_output_length
    if output_length <= 0:
        raise ValueError(
            "decode graph export requires positive decode_max_output_length"
        )
    past_length = sequence_length + output_length - 1
    device = context.device
    generator = torch.Generator().manual_seed(42)
    dry_input_ids = build_text_input_ids(
        model,
        batch_size,
        past_length,
        generator,
    ).to(device)
    dry_attention_mask = torch.ones(
        (batch_size, past_length),
        dtype=torch.long,
        device=device,
    )
    past_cache = run_causal_lm_dry_prefill(model, dry_input_ids, dry_attention_mask)
    past_kv_pairs = collect_causal_lm_kv_pairs(past_cache)
    if not past_kv_pairs:
        raise ValueError("causal LM dry prefill produced no KV cache layers")

    decode_input_ids = torch.zeros((batch_size, 1), dtype=torch.long, device=device)
    decode_attention_mask = torch.ones(
        (batch_size, sequence_length + output_length),
        dtype=torch.long,
        device=device,
    )
    past_kv_flat = tuple(
        tensor.contiguous() for pair in past_kv_pairs for tensor in pair
    )
    runtime_input_names = [
        "input_ids",
        "attention_mask",
        *build_causal_lm_kv_input_names(len(past_kv_pairs)),
    ]
    export_model = CausalLMDecodeGraph(model).to(device)
    with temporary_causal_lm_graph_mode(model):
        exported_program = capture_inference_graph(
            export_model,
            (decode_input_ids, decode_attention_mask, *past_kv_flat),
        )
    export_path = context.output_layout.fx_graphs_dir / f"{spec.name}_decode.pt2"
    result = write_graph_export(exported_program, export_path, runtime_input_names)
    del exported_program, export_model, past_cache, past_kv_pairs, past_kv_flat
    del dry_input_ids, dry_attention_mask, decode_input_ids, decode_attention_mask
    return result


def write_graph_export(
    exported_program: ExportedProgram,
    export_path: Path,
    runtime_input_names: list[str],
) -> GraphExportResult:
    metadata = save_graph_artifact(
        exported_program,
        export_path,
        runtime_input_names=runtime_input_names,
    )
    return build_graph_export_result(
        exported_program,
        export_path,
        runtime_input_names,
        metadata,
    )


def build_graph_export_result(
    exported_program: ExportedProgram,
    export_path: Path,
    runtime_input_names: list[str],
    metadata: GraphArtifactMetadata,
) -> GraphExportResult:
    return GraphExportResult(
        path=str(export_path),
        artifact_schema_version=metadata.schema_version,
        torch_version=metadata.torch_version,
        file_size_bytes=export_path.stat().st_size,
        graph_info=build_graph_info(
            exported_program,
            runtime_input_names=runtime_input_names,
        ),
    )


def extract_graph_logits(outputs: Any) -> torch.Tensor:
    if hasattr(outputs, "logits"):
        return outputs.logits
    if isinstance(outputs, tuple):
        output = outputs[0]
        if isinstance(output, torch.Tensor):
            return output
    if isinstance(outputs, torch.Tensor):
        return outputs
    raise TypeError(f"unsupported model output type: {type(outputs)}")
