"""Scoped point de-bias + one-sided upper-bound calibration primitives.

The serving-domain post-processing of the GNN predictor pipeline: a scoped
affine/linear de-bias that aligns the raw graph prediction to the measured
reference, and a scoped upper residual quantile that turns a point estimate into
the admission bound the scheduler gates on (paper Eq. calibrated-path-peak).
Shared by the production post-processor (`cache_postprocess.py`) and the offline
evaluation (`eval_prediction_caches.py`). Distinct from `calibration.py`, which
is unrelated capacity calibration.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import numpy as np

from workflow.types import WorkflowModelFeatureKey

Scope = tuple[str, str]  # (model_name, gpu_name)
MIN_LINEAR_ROWS = 8


def scope_of(key: WorkflowModelFeatureKey) -> Scope:
    return (key.model_name, key.gpu_name)


def vram_features(key: WorkflowModelFeatureKey, prediction: float) -> list[float]:
    """De-bias regressors: the raw prediction plus the shape knobs its error
    tracks (KV grows with context and batch)."""
    return [
        prediction,
        math.log2(key.sequence_length),
        math.log2(key.batch_size),
        math.log2(max(key.decode_output_length, 1)),
        1.0,
    ]


def fit_scoped_linear(
    rows_by_scope: Mapping[Scope, Sequence[tuple[Sequence[float], float]]],
) -> dict[Scope, np.ndarray]:
    """Least-squares `truth ≈ w·features` per scope; scopes with too few rows
    fall back to a pooled global fit so every scope stays covered."""
    pooled_x: list[Sequence[float]] = []
    pooled_y: list[float] = []
    for rows in rows_by_scope.values():
        for features, truth in rows:
            pooled_x.append(features)
            pooled_y.append(truth)
    if not pooled_x:
        raise ValueError("cannot fit calibration without samples")
    global_coef = _lstsq(
        np.array(pooled_x, dtype=float), np.array(pooled_y, dtype=float)
    )
    coefs: dict[Scope, np.ndarray] = {}
    for scope, rows in rows_by_scope.items():
        if len(rows) < MIN_LINEAR_ROWS:
            coefs[scope] = global_coef
            continue
        x = np.array([features for features, _ in rows], dtype=float)
        y = np.array([truth for _, truth in rows], dtype=float)
        coefs[scope] = _lstsq(x, y)
    return coefs


def apply_linear(coef: np.ndarray, features: Sequence[float]) -> float:
    return float(np.dot(coef, np.array(features, dtype=float)))


def fit_scoped_upper_quantile(
    residuals_by_scope: Mapping[Scope, Sequence[float]],
    *,
    alpha: float = 0.05,
    floor_by_scope: Mapping[Scope, float] | None = None,
    floor_fraction: float = 0.10,
) -> dict[Scope, float]:
    """One-sided upper residual quantile at level 1-alpha per scope, floored so
    a small sample never yields an unsafe (negative) reservation."""
    floors = floor_by_scope or {}
    quantiles: dict[Scope, float] = {}
    for scope, residuals in residuals_by_scope.items():
        if not residuals:
            raise ValueError(f"no residuals for scope {scope}")
        raw = float(np.quantile(np.array(residuals, dtype=float), 1.0 - alpha))
        quantiles[scope] = max(raw, floor_fraction * floors.get(scope, 0.0))
    return quantiles


def _lstsq(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    coef, _, _, _ = np.linalg.lstsq(x, y, rcond=None)
    return coef
