from __future__ import annotations

from collections.abc import Sequence

import torch


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
