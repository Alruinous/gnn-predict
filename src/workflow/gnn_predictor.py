from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Protocol, cast

import numpy as np
import torch
from pydantic import BaseModel, ConfigDict, Field, field_validator
from torch_geometric.data import Data

from gnn_model.config import PreparedDataConfig, SplitDataConfig, load_experiment_config
from gnn_model.data.extract import TARGET_FIELDS
from gnn_model.data.scaler import load_target_scalers
from gnn_model.evaluation.metrics import TargetScaler
from gnn_model.runner import build_model


class ToolPrediction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    node_name: str
    device_name: str
    gpu_name: str
    metrics: dict[str, float] = Field(default_factory=dict)

    def metric(self, name: str, default: float = 0.0) -> float:
        return float(self.metrics.get(name, default))


class WorkflowGnnPredictorConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    config_path: Path
    checkpoint_path: Path
    scaler_dir: Path
    device: str = "cpu"
    target_names: list[str] = Field(default_factory=lambda: list(TARGET_FIELDS))

    @field_validator("target_names")
    @classmethod
    def validate_target_names(cls, value: list[str]) -> list[str]:
        if not value:
            raise ValueError("target_names must not be empty")
        if any(not item.strip() for item in value):
            raise ValueError("target_names must not contain empty values")
        return value


class WorkflowPredictionProvider(Protocol):
    def predict_graph(
        self,
        *,
        node_name: str,
        device_name: str,
        gpu_name: str,
        graph: Data,
    ) -> ToolPrediction: ...


class WorkflowGnnPredictor:
    def __init__(self, config: WorkflowGnnPredictorConfig) -> None:
        self.config = config
        self.device = torch.device(config.device)
        experiment_config = load_experiment_config(config.config_path)
        target_names = resolve_target_names(experiment_config.data, config.target_names)
        self.target_names = target_names
        self.model = build_model(experiment_config, target_names).to(self.device)
        checkpoint = torch.load(
            config.checkpoint_path,
            map_location=self.device,
            weights_only=False,
        )
        state_dict = (
            checkpoint["model_state_dict"]
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint
            else checkpoint
        )
        self.model.load_state_dict(state_dict)
        self.model.eval()
        self.feature_scalers = load_feature_scalers(config.scaler_dir)
        self.target_scalers = load_target_scalers(config.scaler_dir, target_names)

    def predict_graph(
        self,
        *,
        node_name: str,
        device_name: str,
        gpu_name: str,
        graph: Data,
    ) -> ToolPrediction:
        scaled_graph = scale_graph_features(graph, self.feature_scalers)
        with torch.no_grad():
            prediction = self.model(scaled_graph.to(self.device)).cpu().numpy()
        metrics = inverse_transform_prediction(
            prediction,
            self.target_names,
            self.target_scalers,
        )
        return ToolPrediction(
            node_name=node_name,
            device_name=device_name,
            gpu_name=gpu_name,
            metrics=metrics,
        )


def resolve_target_names(
    data_config: PreparedDataConfig | SplitDataConfig,
    fallback_target_names: list[str],
) -> list[str]:
    if isinstance(data_config, SplitDataConfig):
        return list(data_config.target_names)
    return list(fallback_target_names)


def load_feature_scalers(scaler_dir: Path) -> dict[str, Any]:
    feature_scalers: dict[str, Any] = {}
    for scaler_name in (
        "node_feature_scaler",
        "edge_feature_scaler",
        "graph_feature_scaler",
    ):
        scaler_path = Path(scaler_dir) / f"{scaler_name}.pkl"
        if not scaler_path.exists():
            raise FileNotFoundError(f"feature scaler does not exist: {scaler_path}")
        with scaler_path.open("rb") as file:
            feature_scalers[scaler_name] = pickle.load(file)
    return feature_scalers


def scale_graph_features(data: Data, feature_scalers: dict[str, Any]) -> Data:
    scaled_data = data.clone()
    node_scaler = feature_scalers["node_feature_scaler"]
    scaled_data.x = torch.from_numpy(
        node_scaler.transform(data.x.cpu().numpy())
    ).float()

    graph_scaler = feature_scalers["graph_feature_scaler"]
    graph_features = getattr(data, "graph_features", None)
    assert isinstance(graph_features, torch.Tensor)
    scaled_data.graph_features = torch.from_numpy(
        graph_scaler.transform(graph_features.cpu().numpy())
    ).float()

    edge_attr = getattr(data, "edge_attr", None)
    if isinstance(edge_attr, torch.Tensor) and edge_attr.numel() > 0:
        edge_scaler = feature_scalers["edge_feature_scaler"]
        scaled_data.edge_attr = torch.from_numpy(
            edge_scaler.transform(edge_attr.cpu().numpy())
        ).float()
    return scaled_data


def inverse_transform_prediction(
    prediction: np.ndarray,
    target_names: list[str],
    target_scalers: dict[str, TargetScaler],
) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for index, target_name in enumerate(target_names):
        scaler = target_scalers[target_name]
        raw_value = scaler.inverse_transform(prediction[:, index : index + 1])
        metrics[target_name] = float(cast(np.ndarray, raw_value)[0, 0])
    return metrics
