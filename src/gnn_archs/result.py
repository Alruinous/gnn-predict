from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Literal

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


class GraphExportResult(StrictModel):
    path: str
    format: Literal["pt2"] = "pt2"
    artifact_schema_version: str
    torch_version: str
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
    graph_export: GraphExportResult | None = None
    decode_graph_export: GraphExportResult | None = None
    metadata: dict[str, JsonScalar] = Field(default_factory=dict)


class VariantFailure(StrictModel):
    name: str
    stage: str
    error_type: str
    message: str


class ResultDocument(StrictModel):
    schema_version: str = "3.0.0"
    config_path: str
    gpu_node: str
    timings: dict[str, TimeWindow] = Field(default_factory=dict)
    variants: list[VariantResult]
    failures: list[VariantFailure] = Field(default_factory=list)
    summary: dict[str, JsonScalar] = Field(default_factory=dict)


def write_result_document(path: Path, document: ResultDocument) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=output_path.parent,
            prefix=f".{output_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as file:
            temporary_path = Path(file.name)
            json.dump(
                document.model_dump(mode="json"),
                file,
                indent=2,
                ensure_ascii=False,
            )
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary_path, output_path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
