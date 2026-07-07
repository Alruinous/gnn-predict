from __future__ import annotations

import json
from pathlib import Path

from pydantic import Field

from gnn_archs.config import StrictModel

JsonScalar = int | float | str | bool


class TimeWindow(StrictModel):
    started_at_ts: float | None = None
    ended_at_ts: float | None = None
    started_at_text: str | None = None
    ended_at_text: str | None = None


class TrainingResult(StrictModel):
    hyperparameters: dict[str, JsonScalar | list[int]] = Field(default_factory=dict)
    optimizer: dict[str, JsonScalar] = Field(default_factory=dict)
    metrics: dict[str, JsonScalar] = Field(default_factory=dict)
    timings: TimeWindow = Field(default_factory=TimeWindow)


class InferenceResult(StrictModel):
    metrics: dict[str, JsonScalar] = Field(default_factory=dict)
    timings: TimeWindow = Field(default_factory=TimeWindow)


class OnnxExportResult(StrictModel):
    path: str
    opset_version: int
    file_size_bytes: int | None = None
    graph_info: dict[str, int | float | str | list[str]] = Field(default_factory=dict)


class VariantResult(StrictModel):
    name: str
    base_model_name: str
    base_model_pretrained: bool
    source: str
    group_total_variants_defined: int
    variant_config: dict[str, object]
    mutations: list[dict[str, object]]
    timings: dict[str, TimeWindow] = Field(default_factory=dict)
    training: TrainingResult | None = None
    inference: InferenceResult | None = None
    prefill: InferenceResult | None = None
    decode: InferenceResult | None = None
    onnx_export: OnnxExportResult | None = None
    onnx_decode_export: OnnxExportResult | None = None
    metadata: dict[str, JsonScalar] = Field(default_factory=dict)


class ResultDocument(StrictModel):
    schema_version: str = "2.0.0"
    config_path: str
    gpu_node: str
    timings: dict[str, TimeWindow] = Field(default_factory=dict)
    variants: list[VariantResult]
    summary: dict[str, JsonScalar] = Field(default_factory=dict)


def write_result_document(path: Path, document: ResultDocument) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(document.model_dump(mode="json"), file, indent=2, ensure_ascii=False)
