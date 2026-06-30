from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Protocol

import torch
from torch_geometric.data import Data

from gnn_archs.config import (
    BaseModelConfig,
    Gemma4TextConfigOverride,
    Qwen3ConfigOverride,
    ResolvedVariantSpec,
    VariantConfig,
    get_causal_lm_family,
    is_causal_lm_model_name,
    is_text_model_name,
    normalize_model_identifier,
)
from gnn_archs.variant_runner import (
    OutputLayout,
    RunContext,
    build_variant_model,
    export_onnx_model,
)
from gnn_model.data.onnx_graph import build_graph_data_from_onnx
from workflow.schema import WorkflowNodeConfig


class WorkflowModelExporter(Protocol):
    def export_onnx(
        self,
        node: WorkflowNodeConfig,
        output_dir: Path,
    ) -> Path: ...


class GnnArchsWorkflowModelExporter:
    def __init__(self, device: str = "cpu") -> None:
        self.device = torch.device(device)
        self.logger = logging.getLogger(__name__)

    def export_onnx(
        self,
        node: WorkflowNodeConfig,
        output_dir: Path,
    ) -> Path:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        spec = build_resolved_variant_spec(node)
        model = build_variant_model(spec).to(self.device)
        context = RunContext(
            config_path=output_path / "workflow_export.yaml",
            output_layout=OutputLayout(
                root=output_path,
                logs_dir=output_path,
                results_dir=output_path,
                onnx_models_dir=output_path,
            ),
            device=self.device,
            gpu_node="workflow",
            logger=self.logger,
        )
        is_text_model = is_text_model_name(spec.base_model.name)
        result = export_onnx_model(
            spec,
            model,
            context,
            is_text_model,
            False,
        )
        return Path(result.path)


def build_workflow_graph_from_onnx(
    node: WorkflowNodeConfig,
    onnx_path: Path,
    gpu_name: str,
) -> Data:
    assert node.runtime is not None
    return build_graph_data_from_onnx(
        onnx_path,
        batch_size=node.runtime.batch_size,
        gpu_name=gpu_name,
        phase=node.runtime.phase,
        sample_count=1,
        decode_output_length=node.runtime.decode_max_output_length or 0,
    )


def build_resolved_variant_spec(node: WorkflowNodeConfig) -> ResolvedVariantSpec:
    assert node.model is not None
    assert node.runtime is not None
    model_name = node.model.name
    parameters = dict(node.model.parameters)
    variant_config = build_variant_config(model_name, parameters, node)
    return ResolvedVariantSpec(
        name=build_variant_name(node.name),
        base_model=BaseModelConfig(name=model_name, pretrained=False),
        variant_config=variant_config,
        mutations=[],
        source="workflow",
        group_total_variants_defined=1,
    )


def build_variant_config(
    model_name: str,
    parameters: dict[str, Any],
    node: WorkflowNodeConfig,
) -> VariantConfig:
    assert node.runtime is not None
    normalized_name = normalize_model_identifier(model_name)
    payload: dict[str, Any] = {
        "example_input_shape": resolve_example_input_shape(node),
        "run_inference": node.runtime.phase == "inference",
        "run_prefill": node.runtime.phase == "prefill",
        "run_decode": node.runtime.phase == "decode",
        "decode_max_output_length": node.runtime.decode_max_output_length or 0,
        "export_onnx": True,
        "onnx_export_mode": "architecture_only",
        "batch_size": node.runtime.batch_size,
    }
    if not is_causal_lm_model_name(model_name):
        raise ValueError(f"workflow model export only supports causal LM: {model_name}")
    if normalized_name.startswith("qwen"):
        payload["qwen3_config"] = Qwen3ConfigOverride.model_validate(parameters)
        return VariantConfig.model_validate(payload)
    if normalized_name == "gemma4":
        payload["gemma4_config"] = Gemma4TextConfigOverride.model_validate(parameters)
        return VariantConfig.model_validate(payload)
    family = get_causal_lm_family(model_name)
    if parameters:
        raise ValueError(f"{family} workflow export does not accept model parameters")
    return VariantConfig.model_validate(payload)


def resolve_example_input_shape(node: WorkflowNodeConfig) -> list[int]:
    assert node.runtime is not None
    if node.runtime.input_shape is not None:
        return list(node.runtime.input_shape)
    if node.runtime.sequence_length is not None:
        return [node.runtime.batch_size, node.runtime.sequence_length]
    return [node.runtime.batch_size]


def build_variant_name(node_name: str) -> str:
    normalized_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", node_name.strip())
    if not normalized_name:
        raise ValueError("node name must produce a non-empty variant name")
    return normalized_name
