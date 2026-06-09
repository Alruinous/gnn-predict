from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from gnn_model.evaluation.metrics import (
    TargetScaler,
    compute_original_scale_metrics,
    compute_regression_metrics,
)


class IdentityScaler:
    def inverse_transform(self, values: np.ndarray) -> np.ndarray:
        return values


def test_compute_regression_metrics_includes_wape_and_max_error() -> None:
    predictions = torch.tensor([[2.0, 4.0], [6.0, 8.0]])
    targets = torch.tensor([[1.0, 5.0], [3.0, 10.0]])

    metrics = compute_regression_metrics(
        predictions,
        targets,
        ("duration_sec_avg", "gpu_mem_used_mb_max"),
    )

    assert metrics["mae"] == pytest.approx(1.75)
    assert metrics["mse"] == pytest.approx(3.75)
    assert metrics["rmse"] == pytest.approx(math.sqrt(3.75))
    assert metrics["r2"] == pytest.approx(1.0 - 15.0 / 44.75)
    assert metrics["wape"] == pytest.approx(7.0 / 19.0)
    assert metrics["max_abs_error"] == pytest.approx(3.0)
    assert metrics["duration_sec_avg_mae"] == pytest.approx(2.0)
    assert metrics["duration_sec_avg_mse"] == pytest.approx(5.0)
    assert metrics["duration_sec_avg_rmse"] == pytest.approx(math.sqrt(5.0))
    assert metrics["duration_sec_avg_r2"] == pytest.approx(-4.0)
    assert metrics["duration_sec_avg_wape"] == pytest.approx(1.0)
    assert metrics["duration_sec_avg_max_abs_error"] == pytest.approx(3.0)
    assert metrics["gpu_mem_used_mb_max_mae"] == pytest.approx(1.5)
    assert metrics["gpu_mem_used_mb_max_mse"] == pytest.approx(2.5)
    assert metrics["gpu_mem_used_mb_max_rmse"] == pytest.approx(math.sqrt(2.5))
    assert metrics["gpu_mem_used_mb_max_r2"] == pytest.approx(0.6)
    assert metrics["gpu_mem_used_mb_max_wape"] == pytest.approx(0.2)
    assert metrics["gpu_mem_used_mb_max_max_abs_error"] == pytest.approx(2.0)


def test_compute_original_scale_metrics_prefixes_new_metrics() -> None:
    predictions = torch.tensor([[2.0, 4.0], [6.0, 8.0]])
    targets = torch.tensor([[1.0, 5.0], [3.0, 10.0]])
    target_scalers: dict[str, TargetScaler] = {
        "duration_sec_avg": IdentityScaler(),
        "gpu_mem_used_mb_max": IdentityScaler(),
    }

    metrics = compute_original_scale_metrics(
        predictions,
        targets,
        ("duration_sec_avg", "gpu_mem_used_mb_max"),
        target_scalers,
    )

    assert metrics["original_scale_wape"] == pytest.approx(7.0 / 19.0)
    assert metrics["original_scale_r2"] == pytest.approx(1.0 - 15.0 / 44.75)
    assert metrics["original_scale_max_abs_error"] == pytest.approx(3.0)
    assert metrics["original_scale_duration_sec_avg_wape"] == pytest.approx(1.0)
    assert metrics["original_scale_duration_sec_avg_r2"] == pytest.approx(-4.0)
    assert metrics["original_scale_duration_sec_avg_max_abs_error"] == pytest.approx(
        3.0
    )


def test_compute_regression_metrics_allows_zero_wape_denominator_without_error() -> None:
    predictions = torch.zeros((2, 2))
    targets = torch.zeros((2, 2))

    metrics = compute_regression_metrics(predictions, targets, ("latency", "memory"))

    assert metrics["wape"] == 0.0
    assert metrics["latency_wape"] == 0.0
    assert metrics["memory_wape"] == 0.0


def test_compute_regression_metrics_rejects_zero_wape_denominator_with_error() -> None:
    predictions = torch.tensor([[1.0, 0.0]])
    targets = torch.zeros((1, 2))

    with pytest.raises(ValueError, match="WAPE denominator is zero"):
        compute_regression_metrics(predictions, targets, ("latency", "memory"))
