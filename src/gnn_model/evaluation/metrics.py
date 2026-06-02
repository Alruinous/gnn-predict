from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

import numpy as np
import torch


class TargetScaler(Protocol):
    def inverse_transform(self, values: np.ndarray) -> np.ndarray: ...


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
    error_sum = absolute_error.sum()
    target_sum = targets.abs().sum()
    if float(target_sum.item()) == 0.0:
        if float(error_sum.item()) != 0.0:
            raise ValueError("WAPE denominator is zero for non-zero prediction error")
        wape = 0.0
    else:
        wape = float((error_sum / target_sum).item())
    metrics = {
        "mae": float(absolute_error.mean().item()),
        "mse": float(squared_error.mean().item()),
        "rmse": float(squared_error.mean().sqrt().item()),
        "r2": compute_r2(predictions, targets),
        "wape": wape,
        "max_abs_error": float(absolute_error.max().item()),
    }
    for index, target_name in enumerate(target_names):
        target_absolute_error = absolute_error[:, index]
        target_squared_error = squared_error[:, index]
        target_error_sum = target_absolute_error.sum()
        target_sum = targets[:, index].abs().sum()
        if float(target_sum.item()) == 0.0:
            if float(target_error_sum.item()) != 0.0:
                raise ValueError(
                    "WAPE denominator is zero for non-zero prediction error"
                )
            target_wape = 0.0
        else:
            target_wape = float((target_error_sum / target_sum).item())
        metrics[f"{target_name}_mae"] = float(target_absolute_error.mean().item())
        metrics[f"{target_name}_mse"] = float(target_squared_error.mean().item())
        metrics[f"{target_name}_rmse"] = float(
            target_squared_error.mean().sqrt().item()
        )
        metrics[f"{target_name}_r2"] = compute_r2(
            predictions[:, index],
            targets[:, index],
        )
        metrics[f"{target_name}_wape"] = target_wape
        metrics[f"{target_name}_max_abs_error"] = float(
            target_absolute_error.max().item()
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
        column = torch.from_numpy(transformed).float()
        columns.append(column)
    return torch.cat(columns, dim=1)


def compute_r2(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    residual_sum = (predictions - targets).pow(2).sum()
    total_sum = (targets - targets.mean()).pow(2).sum()
    if float(total_sum.item()) == 0.0:
        return 1.0 if float(residual_sum.item()) == 0.0 else 0.0
    return float((1.0 - residual_sum / total_sum).item())
