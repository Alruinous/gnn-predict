from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, cast

import numpy as np
import torch


class TargetScaler(Protocol):
    def inverse_transform(self, values: np.ndarray) -> np.ndarray:
        ...


def compute_regression_metrics(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    target_names: Sequence[str],
) -> dict[str, float]:
    if predictions.shape != targets.shape:
        raise ValueError(
            f"prediction shape mismatch: {predictions.shape} != {targets.shape}"
        )
    if predictions.ndim != 2:
        raise ValueError("predictions and targets must be rank-2 tensors")

    absolute_error = (predictions - targets).abs()
    squared_error = (predictions - targets).pow(2)
    metrics = {
        "mae": float(absolute_error.mean().item()),
        "mse": float(squared_error.mean().item()),
        "rmse": float(squared_error.mean().sqrt().item()),
    }
    for index, target_name in enumerate(target_names):
        metrics[f"{target_name}_mae"] = float(absolute_error[:, index].mean().item())
        metrics[f"{target_name}_rmse"] = float(
            squared_error[:, index].mean().sqrt().item()
        )
    return metrics


def compute_original_scale_metrics(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    target_names: Sequence[str],
    target_scalers: dict[str, TargetScaler],
) -> dict[str, float]:
    original_predictions = inverse_transform_targets(
        predictions,
        target_names,
        target_scalers,
    )
    original_targets = inverse_transform_targets(
        targets,
        target_names,
        target_scalers,
    )
    metrics = compute_regression_metrics(
        original_predictions,
        original_targets,
        target_names,
    )
    return {
        f"original_scale_{metric_name}": metric_value
        for metric_name, metric_value in metrics.items()
    }


def inverse_transform_targets(
    values: torch.Tensor,
    target_names: Sequence[str],
    target_scalers: dict[str, TargetScaler],
) -> torch.Tensor:
    if values.ndim != 2:
        raise ValueError("target values must be rank-2 tensors")
    if values.size(1) != len(target_names):
        raise ValueError(
            f"target dim mismatch: {values.size(1)} != {len(target_names)}"
        )

    columns: list[torch.Tensor] = []
    values_array = values.detach().cpu().numpy()
    for index, target_name in enumerate(target_names):
        scaler = target_scalers.get(target_name)
        if scaler is None:
            raise ValueError(f"target scaler missing: {target_name}")
        transformed = scaler.inverse_transform(values_array[:, index : index + 1])
        column = torch.from_numpy(cast(np.ndarray, transformed)).float()
        columns.append(column)
    return torch.cat(columns, dim=1)
