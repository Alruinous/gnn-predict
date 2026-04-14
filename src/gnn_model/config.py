from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class FakeDataConfig(StrictModel):
    kind: Literal["fake"] = "fake"
    dataset_size: int = 12
    val_ratio: float = 0.2
    test_ratio: float = 0.2
    random_seed: int = 7
    input_shape: list[int] = Field(default_factory=lambda: [1, 3, 32, 32])

    @model_validator(mode="after")
    def validate_fake_data_config(self) -> FakeDataConfig:
        if self.dataset_size < 3:
            raise ValueError("fake dataset_size must be at least 3")
        if len(self.input_shape) != 4:
            raise ValueError(
                "fake input_shape must be [batch, channels, height, width]"
            )
        if any(value <= 0 for value in self.input_shape):
            raise ValueError("fake input_shape values must be positive")
        if not 0 < self.val_ratio < 1:
            raise ValueError("val_ratio must be between 0 and 1")
        if not 0 < self.test_ratio < 1:
            raise ValueError("test_ratio must be between 0 and 1")
        if self.val_ratio + self.test_ratio >= 1:
            raise ValueError("val_ratio + test_ratio must be less than 1")
        return self


class PreparedDataConfig(StrictModel):
    kind: Literal["prepared"] = "prepared"
    manifest_path: str

    @model_validator(mode="after")
    def validate_prepared_data_config(self) -> PreparedDataConfig:
        if not self.manifest_path.strip():
            raise ValueError("prepared manifest_path must not be empty")
        return self


DataConfig = FakeDataConfig | PreparedDataConfig


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
    experiment_name: str = "gnn_model_fake_run"
    data: DataConfig = Field(default_factory=FakeDataConfig)
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
