from __future__ import annotations

import json
from pathlib import Path

from pydantic import Field

from gnn_model.config import StrictModel

JsonScalar = int | float | str | bool


class TimeWindow(StrictModel):
    started_at_ts: float | None = None
    ended_at_ts: float | None = None
    started_at_text: str | None = None
    ended_at_text: str | None = None


class DatasetSummary(StrictModel):
    train_count: int
    val_count: int
    test_count: int


class TrainingSummary(StrictModel):
    batch_size: int
    num_epochs: int
    learning_rate: float
    weight_decay: float
    best_epoch: int
    best_val_loss: float
    last_train_loss: float
    checkpoint_path: str


class EvaluationSummary(StrictModel):
    split: str
    metrics: dict[str, float] = Field(default_factory=dict)


class ExperimentResult(StrictModel):
    schema_version: str = "1.0.0"
    experiment_name: str
    config_path: str
    device: str
    target_names: list[str]
    dataset: DatasetSummary
    timings: dict[str, TimeWindow] = Field(default_factory=dict)
    training: TrainingSummary
    evaluation: EvaluationSummary
    metadata: dict[str, JsonScalar] = Field(default_factory=dict)


def write_result_document(path: Path, document: ExperimentResult) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(document.model_dump(mode="json"), file, indent=2, ensure_ascii=False)
