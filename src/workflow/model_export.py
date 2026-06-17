from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Protocol

import torch
from torch_geometric.data import Data

from gnn_archs.config import (
    BaseModelConfig,
    DCNConfigOverride,
    DCNv2ConfigOverride,
    DeepFMConfigOverride,
    EDCNConfigOverride,
    Gpt2ConfigOverride,
    Qwen3ConfigOverride,
    ResolvedVariantSpec,
    T5ConfigOverride,
    VariantConfig,
    is_causal_lm_model_name,
    is_recommender_model_name,
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

RECOMMENDER_CONFIG_BY_NAME = {
    "deepfm": ("deepfm_config", DeepFMConfigOverride),
    "dcn": ("dcn_config", DCNConfigOverride),
    "dcnv2": ("dcnv2_config", DCNv2ConfigOverride),
    "edcn": ("edcn_config", EDCNConfigOverride),
}


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
        is_recommender_model = is_recommender_model_name(spec.base_model.name)
        result = export_onnx_model(
            spec,
            model,
            context,
            is_text_model,
            is_recommender_model,
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

    if normalized_name == "gpt2":
        payload["target_input_channels"] = int(parameters.get("input_channels", 1))
        payload["target_output_classes"] = int(parameters["output_classes"])
        payload["gpt2_config"] = Gpt2ConfigOverride.model_validate(
            without_common_output_fields(parameters)
        )
        return VariantConfig.model_validate(payload)
    if normalized_name == "t5":
        payload["target_input_channels"] = int(parameters.get("input_channels", 1))
        payload["target_output_classes"] = int(parameters["output_classes"])
        payload["t5_config"] = T5ConfigOverride.model_validate(
            without_common_output_fields(parameters)
        )
        return VariantConfig.model_validate(payload)
    if normalized_name.startswith("qwen"):
        payload["qwen3_config"] = Qwen3ConfigOverride.model_validate(parameters)
        return VariantConfig.model_validate(payload)
    if is_causal_lm_model_name(model_name):
        return VariantConfig.model_validate(payload)
    if is_recommender_model_name(model_name):
        config_field, config_model = RECOMMENDER_CONFIG_BY_NAME[normalized_name]
        payload["target_output_classes"] = int(parameters.get("output_classes", 1))
        payload[config_field] = config_model.model_validate(parameters[config_field])
        return VariantConfig.model_validate(payload)

    if not is_text_model_name(model_name):
        payload["target_input_channels"] = resolve_input_channels(parameters, node)
    else:
        payload["target_input_channels"] = int(parameters.get("input_channels", 1))
    if "output_classes" in parameters:
        payload["target_output_classes"] = int(parameters["output_classes"])
    return VariantConfig.model_validate(payload)


def without_common_output_fields(parameters: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in parameters.items()
        if key not in {"input_channels", "output_classes"}
    }


def resolve_example_input_shape(node: WorkflowNodeConfig) -> list[int]:
    assert node.runtime is not None
    if node.runtime.input_shape is not None:
        return list(node.runtime.input_shape)
    if node.runtime.sequence_length is not None:
        return [node.runtime.batch_size, node.runtime.sequence_length]
    return [node.runtime.batch_size]


def resolve_input_channels(
    parameters: dict[str, Any],
    node: WorkflowNodeConfig,
) -> int:
    if "input_channels" in parameters:
        return int(parameters["input_channels"])
    assert node.runtime is not None
    if node.runtime.input_shape is None or len(node.runtime.input_shape) < 2:
        raise ValueError(f"input_channels is required: {node.name}")
    return int(node.runtime.input_shape[1])


def build_variant_name(node_name: str) -> str:
    normalized_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", node_name.strip())
    if not normalized_name:
        raise ValueError("node name must produce a non-empty variant name")
    return normalized_name
