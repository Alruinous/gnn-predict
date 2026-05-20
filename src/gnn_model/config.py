from __future__ import annotations

from pathlib import Path
from typing import Annotated, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class PreparedDataConfig(StrictModel):
    kind: Literal["prepared"] = "prepared"
    manifest_path: str

    @model_validator(mode="after")
    def validate_prepared_data_config(self) -> PreparedDataConfig:
        if not self.manifest_path.strip():
            raise ValueError("prepared manifest_path must not be empty")
        return self


class SplitDataConfig(StrictModel):
    kind: Literal["split"] = "split"
    data_dir: str = "data/scaled"
    target_names: list[str] = Field(
        default_factory=lambda: [
            "duration_sec_avg",
            "cpu_cores_p95",
            "memory_gb_p95",
            "memory_delta_gb_p95",
            "gpu_util_percent_p95",
            "gpu_sm_occupancy_percent_p95",
            "gpu_mem_used_mb_p95",
        ]
    )
    scaler_dir: str = "data/scalers"
    split_files: dict[str, str] = Field(
        default_factory=lambda: {
            "train": "train.pt",
            "val": "val.pt",
            "test": "test.pt",
        }
    )

    @model_validator(mode="after")
    def validate_split_data_config(self) -> SplitDataConfig:
        if not self.data_dir.strip():
            raise ValueError("split data_dir must not be empty")
        if not self.scaler_dir.strip():
            raise ValueError("split scaler_dir must not be empty")
        if not self.target_names:
            raise ValueError("split target_names must not be empty")
        if any(not target_name.strip() for target_name in self.target_names):
            raise ValueError("split target_names must not contain empty values")
        required_splits = {"train", "val", "test"}
        if set(self.split_files) != required_splits:
            raise ValueError("split split_files must contain train, val, and test")
        if any(not value.strip() for value in self.split_files.values()):
            raise ValueError("split split_files values must not be empty")
        return self


DataConfig = Annotated[
    PreparedDataConfig | SplitDataConfig,
    Field(discriminator="kind"),
]


class ModelConfig(StrictModel):
    hidden_dim: int = 64
    num_layers: int = 2
    num_heads: int = 4
    dropout_rate: float = 0.1

    @model_validator(mode="after")
    def validate_model_config(self) -> ModelConfig:
        if self.hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        if self.num_layers <= 0:
            raise ValueError("num_layers must be positive")
        if self.num_heads <= 0:
            raise ValueError("num_heads must be positive")
        if self.hidden_dim % self.num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")
        if not 0 <= self.dropout_rate < 1:
            raise ValueError("dropout_rate must be in [0, 1)")
        return self


class TrainingConfig(StrictModel):
    batch_size: int = 4
    num_epochs: int = 1
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    loss_weights: list[float] | None = None

    @model_validator(mode="after")
    def validate_training_config(self) -> TrainingConfig:
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.num_epochs <= 0:
            raise ValueError("num_epochs must be positive")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.weight_decay < 0:
            raise ValueError("weight_decay must be non-negative")
        if self.loss_weights is not None and any(
            weight <= 0 for weight in self.loss_weights
        ):
            raise ValueError("loss_weights must contain only positive values")
        return self


class ExperimentConfig(StrictModel):
    experiment_name: str = "gnn_model_scaled_run"
    data: DataConfig = Field(default_factory=SplitDataConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)

    @model_validator(mode="after")
    def validate_experiment_name(self) -> ExperimentConfig:
        if not self.experiment_name.strip():
            raise ValueError("experiment_name must not be empty")
        return self


def load_experiment_config(path: Path) -> ExperimentConfig:
    config_path = Path(path)
    with config_path.open(encoding="utf-8") as file:
        raw_config = yaml.safe_load(file)
    return ExperimentConfig.model_validate(raw_config)
