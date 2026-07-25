"""Iso-budget predictor comparison: accuracy, decision quality, and cost.

Every lookup/learned method sees the same real measurements SageRadar's runtime
calibration consumed (recorded in the GNN cache metrics: 5 per model-GPU cell,
all at output length 512); the analytical formula sees none. The profile cache is
the empirical upper bound. Scoring runs on the complement of the anchor set plus
a long-decode stratum whose output length no anchor covers.

Reuses the baseline builders in `predictor_baselines` and the scoped calibration
primitives in `prediction_calibration`; adds only the anchor bookkeeping, the
equal-budget analytical variant, and the decision/bound metrics the scheduler
actually cares about. See `docs/workflow/gnn_isobudget_experiment_20260725.md`.
"""

from __future__ import annotations

import itertools
import json
import math
import random
import statistics
import time
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from experiment.workflow.prediction_calibration import (
    Scope,
    apply_linear,
    fit_scoped_linear,
    scope_of,
    vram_features,
)
from experiment.workflow.predictor_baselines import (
    build_config_mean_cache,
    build_nearest_profile_cache,
    build_tabular_cache,
    stratified_anchor_sample,
)
from experiment.workflow.static_cache import (
    STATIC_ARCH,
    build_static_prediction_cache,
)
from workflow.artifacts import ResourceContract, ResourceContractCache
from workflow.types import WorkflowModelFeatureKey

RUN = "run_sec"
VRAM = "peak_vram_mb"
TARGETS = (RUN, VRAM)
LONG_DECODE_OUTPUT = 1024  # anchors stop at 512, so this stratum is extrapolation
ANCHORS_PER_CELL = 5
PAIR_SAMPLE_LIMIT = 60_000
PAIR_SEED = 7
BYTES_PER_MIB = 1024 * 1024
KV_BYTES_PER_TOKEN_HEAD = 2 * 2  # key+value, float16

Predictions = Mapping[WorkflowModelFeatureKey, float]
Truth = Mapping[WorkflowModelFeatureKey, ResourceContract]


def target_value(contract: ResourceContract, target: str) -> float:
    if target == RUN:
        return contract.predicted_run_sec
    assert target == VRAM, f"unsupported target {target}"
    return contract.predicted_peak_vram_mb


def index_cache(
    cache: ResourceContractCache,
) -> dict[WorkflowModelFeatureKey, ResourceContract]:
    return {entry.key: entry for entry in cache.entries}


def load_recorded_anchors(
    metrics_path: Path, truth: Truth
) -> tuple[WorkflowModelFeatureKey, ...]:
    """The exact measurements the frozen GNN cache spent its runtime budget on."""
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    calibration = payload["runtime_calibration"]
    keys = tuple(
        WorkflowModelFeatureKey(**record)
        for record in calibration["selected_sample_keys"]
    )
    missing = [key for key in keys if key not in truth]
    assert not missing, f"{len(missing)} recorded anchors absent from profile truth"
    return keys


def spread_anchors(
    truth: Truth, per_cell: int = ANCHORS_PER_CELL
) -> tuple[WorkflowModelFeatureKey, ...]:
    """Same measurement count spread over the whole grid — the baselines' best case."""
    return stratified_anchor_sample(tuple(truth), per_cell)


def held_out_keys(
    truth: Truth, anchors: Iterable[WorkflowModelFeatureKey]
) -> tuple[WorkflowModelFeatureKey, ...]:
    spent = set(anchors)
    return tuple(key for key in truth if key not in spent)


def long_decode_keys(
    keys: Iterable[WorkflowModelFeatureKey],
) -> tuple[WorkflowModelFeatureKey, ...]:
    return tuple(k for k in keys if k.decode_output_length > LONG_DECODE_OUTPUT)


@dataclass(frozen=True, slots=True)
class MethodPredictions:
    run_sec: Predictions
    peak_vram_mb: Predictions

    def of(self, target: str) -> Predictions:
        return self.run_sec if target == RUN else self.peak_vram_mb


def _from_cache(cache: ResourceContractCache) -> MethodPredictions:
    entries = cache.entries
    return MethodPredictions(
        run_sec={e.key: e.predicted_run_sec for e in entries},
        peak_vram_mb={e.key: e.predicted_peak_vram_mb for e in entries},
    )


def fit_cell_scale(
    truth: Truth,
    base: Predictions,
    anchors: Sequence[WorkflowModelFeatureKey],
    target: str,
) -> dict[Scope | None, float]:
    """Least-squares multiplier per model-GPU cell; `None` holds the pooled fallback."""
    per_cell: dict[Scope, list[float]] = defaultdict(lambda: [0.0, 0.0])
    pooled = [0.0, 0.0]
    for key in anchors:
        prediction = base[key]
        observed = target_value(truth[key], target)
        for accumulator in (per_cell[scope_of(key)], pooled):
            accumulator[0] += prediction * observed
            accumulator[1] += prediction * prediction
    assert pooled[1] > 0.0, "analytical baseline produced only zero predictions"
    scales: dict[Scope | None, float] = {None: pooled[0] / pooled[1]}
    for scope, accumulator in per_cell.items():
        if accumulator[1] > 0.0:
            scales[scope] = accumulator[0] / accumulator[1]
    return scales


def _rescaled(base: Predictions, scales: Mapping[Scope | None, float]) -> Predictions:
    fallback = scales[None]
    return {
        key: scales.get(scope_of(key), fallback) * value for key, value in base.items()
    }


def debias_vram(
    raw: Predictions, truth: Truth, anchors: Sequence[WorkflowModelFeatureKey]
) -> Predictions:
    """Scoped affine de-bias of the raw graph VRAM estimate, fit on the anchors only."""
    rows: dict[Scope, list[tuple[Sequence[float], float]]] = defaultdict(list)
    for key in anchors:
        rows[scope_of(key)].append(
            (vram_features(key, raw[key]), target_value(truth[key], VRAM))
        )
    coefficients = fit_scoped_linear(rows)
    pooled = fit_scoped_linear(
        {("pooled", "pooled"): [row for scoped in rows.values() for row in scoped]}
    )[("pooled", "pooled")]
    return {
        key: max(
            apply_linear(
                coefficients.get(scope_of(key), pooled), vram_features(key, value)
            ),
            1.0,
        )
        for key, value in raw.items()
    }


def build_sageradar_predictions(
    gnn: Mapping[WorkflowModelFeatureKey, ResourceContract],
    truth: Truth,
    anchors: Sequence[WorkflowModelFeatureKey],
) -> MethodPredictions:
    """Run time is the frozen calibrated cache; VRAM gets the same anchor budget."""
    raw_vram = {key: entry.predicted_peak_vram_mb for key, entry in gnn.items()}
    return MethodPredictions(
        run_sec={key: entry.predicted_run_sec for key, entry in gnn.items()},
        peak_vram_mb=debias_vram(raw_vram, truth, anchors),
    )


def _kv_cache_mib(key: WorkflowModelFeatureKey) -> float:
    arch = STATIC_ARCH[key.model_name]
    tokens = key.sequence_length + key.decode_output_length
    return (
        KV_BYTES_PER_TOKEN_HEAD
        * arch.num_layers
        * arch.num_kv_heads
        * arch.head_dim
        * tokens
        * key.batch_size
        / BYTES_PER_MIB
    )


def build_anchor_scaled_predictions(
    truth: Truth,
    anchors: Sequence[WorkflowModelFeatureKey],
    all_keys: Sequence[WorkflowModelFeatureKey],
) -> MethodPredictions:
    """Expert sparse-profiling heuristic: in-cell nearest anchor, then scale run
    time linearly in output length and shift VRAM by the analytic KV delta.

    Excluded from the paper's baseline set — it is a measurement method requiring
    a real run in every model-GPU cell, and it matches SageRadar on this grid.
    Kept for the internal record so the finding is not lost."""
    by_cell: dict[Scope, list[WorkflowModelFeatureKey]] = defaultdict(list)
    for key in anchors:
        by_cell[scope_of(key)].append(key)

    def nearest(key: WorkflowModelFeatureKey) -> WorkflowModelFeatureKey:
        candidates = by_cell[scope_of(key)]
        assert candidates, f"no anchor in cell {scope_of(key)}"
        return min(
            candidates,
            key=lambda anchor: (
                (math.log2(anchor.sequence_length) - math.log2(key.sequence_length))
                ** 2
                + (math.log2(anchor.batch_size) - math.log2(key.batch_size)) ** 2
            ),
        )

    donors = {key: nearest(key) for key in all_keys}
    return MethodPredictions(
        run_sec={
            key: target_value(truth[donor], RUN)
            * key.decode_output_length
            / donor.decode_output_length
            for key, donor in donors.items()
        },
        peak_vram_mb={
            key: max(
                target_value(truth[donor], VRAM)
                + _kv_cache_mib(key)
                - _kv_cache_mib(donor),
                1.0,
            )
            for key, donor in donors.items()
        },
    )


def build_method_predictions(
    truth: Truth,
    gnn: Mapping[WorkflowModelFeatureKey, ResourceContract],
    anchors: Sequence[WorkflowModelFeatureKey],
    *,
    include_heuristic: bool = False,
) -> dict[str, MethodPredictions]:
    all_keys = tuple(truth)
    analytical = _from_cache(build_static_prediction_cache(all_keys))
    methods = {
        "sageradar": build_sageradar_predictions(gnn, truth, anchors),
        "analytical": analytical,
        "analytical_cal": MethodPredictions(
            **{
                target: _rescaled(
                    analytical.of(target),
                    fit_cell_scale(truth, analytical.of(target), anchors, target),
                )
                for target in TARGETS
            }
        ),
        "nearest_profile": _from_cache(
            build_nearest_profile_cache(truth, anchors, all_keys)
        ),
        "config_mean": _from_cache(build_config_mean_cache(truth, anchors, all_keys)),
        "tabular": _from_cache(build_tabular_cache(truth, anchors, all_keys)),
    }
    if include_heuristic:
        methods["anchor_scaled"] = build_anchor_scaled_predictions(
            truth, anchors, all_keys
        )
    return methods


@dataclass(frozen=True, slots=True)
class Accuracy:
    count: int
    wape: float
    p95_relative_error: float
    underestimate_rate: float
    max_underestimate: float


def score_accuracy(
    predictions: Predictions,
    truth: Truth,
    keys: Sequence[WorkflowModelFeatureKey],
    target: str,
) -> Accuracy:
    assert keys, "cannot score an empty key set"
    observed = np.array([target_value(truth[key], target) for key in keys])
    predicted = np.array([predictions[key] for key in keys])
    absolute = np.abs(predicted - observed)
    shortfall = observed - predicted
    return Accuracy(
        count=len(keys),
        wape=float(absolute.sum() / observed.sum()),
        p95_relative_error=float(np.quantile(absolute / observed, 0.95)),
        underestimate_rate=float((predicted < observed).mean()),
        max_underestimate=float(shortfall.max()),
    )


def pairwise_order_accuracy(
    predictions: Predictions,
    truth: Truth,
    keys: Sequence[WorkflowModelFeatureKey],
    target: str = RUN,
    *,
    sample_limit: int = PAIR_SAMPLE_LIMIT,
    seed: int = PAIR_SEED,
) -> float:
    """Fraction of pairs ordered as the oracle would — the SRPT/ETA ordering
    SageWeaver consumes, which point error alone does not capture."""
    assert len(keys) >= 2, "pairwise ordering needs at least two keys"
    pairs = list(itertools.combinations(range(len(keys)), 2))
    random.Random(seed).shuffle(pairs)
    pairs = pairs[:sample_limit]
    observed = [target_value(truth[key], target) for key in keys]
    predicted = [predictions[key] for key in keys]
    agree = sum(
        1
        for left, right in pairs
        if (predicted[left] < predicted[right]) == (observed[left] < observed[right])
    )
    return agree / len(pairs)


@dataclass(frozen=True, slots=True)
class ThresholdDecision:
    threshold: float
    false_positive: int  # predicted to fit but does not — overrun / OOM
    false_negative: int  # predicted not to fit but would — lost opportunity
    accuracy: float
    asymmetric_cost: float


def threshold_decision(
    predictions: Predictions,
    truth: Truth,
    keys: Sequence[WorkflowModelFeatureKey],
    target: str,
    threshold: float,
    *,
    false_positive_weight: float = 5.0,
) -> ThresholdDecision:
    """`predicted <= threshold` is the admit/fit decision; a false admit costs a
    preemption plus reload, so it is weighted above a false reject."""
    assert keys, "cannot evaluate a decision on an empty key set"
    observed = np.array([target_value(truth[key], target) for key in keys])
    predicted = np.array([predictions[key] for key in keys])
    admits = predicted <= threshold
    feasible = observed <= threshold
    false_positive = int((admits & ~feasible).sum())
    false_negative = int((~admits & feasible).sum())
    return ThresholdDecision(
        threshold=threshold,
        false_positive=false_positive,
        false_negative=false_negative,
        accuracy=1.0 - (false_positive + false_negative) / len(keys),
        asymmetric_cost=(false_positive_weight * false_positive + false_negative)
        / len(keys),
    )


def decision_thresholds(
    truth: Truth,
    keys: Sequence[WorkflowModelFeatureKey],
    target: str,
    quantiles: Sequence[float] = (0.25, 0.5, 0.75),
) -> tuple[float, ...]:
    observed = [target_value(truth[key], target) for key in keys]
    return tuple(float(np.quantile(observed, q)) for q in quantiles)


@dataclass(frozen=True, slots=True)
class BoundQuality:
    quantile_level: float
    violation_rate: float
    mean_over_reservation: float
    p95_over_reservation: float


def bound_quality(
    predictions: Predictions,
    truth: Truth,
    anchors: Sequence[WorkflowModelFeatureKey],
    keys: Sequence[WorkflowModelFeatureKey],
    *,
    target_violation: float = 0.05,
    floor_fraction: float = 0.02,
) -> BoundQuality:
    """Turn each point estimate into an admission bound with the same recipe — a
    scoped upper residual quantile fit on the anchors — then compare safety
    against wasted reservation at matched anchor-side violation."""
    residuals: dict[Scope, list[float]] = defaultdict(list)
    for key in anchors:
        residuals[scope_of(key)].append(
            target_value(truth[key], VRAM) - predictions[key]
        )

    def bound_at(level: float, key: WorkflowModelFeatureKey) -> float:
        scoped = residuals.get(scope_of(key))
        margin = float(np.quantile(scoped, level)) if scoped else 0.0
        point = predictions[key]
        return max(point + margin, point * (1.0 + floor_fraction))

    level = 1.0
    for candidate in np.arange(0.50, 1.0001, 0.01):
        candidate = min(float(candidate), 1.0)
        breached = sum(
            1
            for key in anchors
            if target_value(truth[key], VRAM) > bound_at(candidate, key)
        )
        if breached / len(anchors) <= target_violation:
            level = candidate
            break

    observed = np.array([target_value(truth[key], VRAM) for key in keys])
    bounds = np.array([bound_at(level, key) for key in keys])
    slack = (bounds - observed) / observed
    return BoundQuality(
        quantile_level=level,
        violation_rate=float((observed > bounds).mean()),
        mean_over_reservation=float(slack.mean()),
        p95_over_reservation=float(np.quantile(slack, 0.95)),
    )


@dataclass(frozen=True, slots=True)
class ProfilingCost:
    config_count: int
    run_gpu_hours: float
    load_gpu_hours: float
    total_gpu_hours: float
    marginal_sec_per_config: float


def profiling_cost(
    truth: Truth, keys: Sequence[WorkflowModelFeatureKey]
) -> ProfilingCost:
    """Decode execution plus one model load per model-GPU cell touched."""
    assert keys, "cannot cost an empty campaign"
    run_sec = sum(target_value(truth[key], RUN) for key in keys)
    load_by_cell = {scope_of(key): truth[key].predicted_load_sec for key in keys}
    load_sec = sum(load_by_cell.values())
    total = run_sec + load_sec
    return ProfilingCost(
        config_count=len(keys),
        run_gpu_hours=run_sec / 3600.0,
        load_gpu_hours=load_sec / 3600.0,
        total_gpu_hours=total / 3600.0,
        marginal_sec_per_config=total / len(keys),
    )


def graph_forward_seconds(iterations: int = 20) -> float:
    """Median cost of one predictor forward pass — SageRadar's marginal cost."""
    import torch
    from torch_geometric.data import Data

    from gnn_model.data.constants import (
        EDGE_FEATURE_DIM,
        GRAPH_FEATURE_DIM,
        NODE_FEATURE_DIM,
        OP_TYPE_COUNT,
        OP_TYPE_EMBEDDING_DIM,
    )
    from gnn_model.models.predictor import IntelliGraphLargeModelPredictor

    node_count, edge_count = 800, 1600
    data = Data(
        x=torch.randn(node_count, NODE_FEATURE_DIM),
        edge_index=torch.randint(0, node_count, (2, edge_count)),
        edge_attr=torch.randn(edge_count, EDGE_FEATURE_DIM),
    )
    data.graph_features = torch.randn(1, GRAPH_FEATURE_DIM)
    data.op_type_ids = torch.randint(0, OP_TYPE_COUNT, (node_count,))
    model = IntelliGraphLargeModelPredictor(
        node_dim=NODE_FEATURE_DIM,
        edge_dim=EDGE_FEATURE_DIM,
        graph_dim=GRAPH_FEATURE_DIM,
        op_type_count=OP_TYPE_COUNT,
        op_type_embedding_dim=OP_TYPE_EMBEDDING_DIM,
        hidden_dim=128,
        targets=[f"t{index}" for index in range(9)],
        num_heads=8,
        num_layers=2,
        readout_mode="mean_sum_max",
        structural_context_mode="basic",
    ).eval()
    samples: list[float] = []
    with torch.no_grad():
        for _ in range(3):
            model(data)
        for _ in range(iterations):
            start = time.perf_counter()
            model(data)
            samples.append(time.perf_counter() - start)
    return statistics.median(samples)


def baseline_marginal_seconds(
    truth: Truth,
    anchors: Sequence[WorkflowModelFeatureKey],
) -> dict[str, float]:
    """Amortized build time per configuration for the CPU-only methods."""
    all_keys = tuple(truth)
    count = len(all_keys)
    timings: dict[str, float] = {}
    builders = {
        "analytical": lambda: build_static_prediction_cache(all_keys),
        "config_mean": lambda: build_config_mean_cache(truth, anchors, all_keys),
        "nearest_profile": lambda: build_nearest_profile_cache(
            truth, anchors, all_keys
        ),
        "tabular": lambda: build_tabular_cache(truth, anchors, all_keys),
    }
    for name, build in builders.items():
        start = time.perf_counter()
        build()
        timings[name] = (time.perf_counter() - start) / count
    timings["analytical_cal"] = timings["analytical"]
    return timings


METHOD_ORDER = (
    "sageradar",
    "analytical_cal",
    "analytical",
    "nearest_profile",
    "config_mean",
    "tabular",
    "anchor_scaled",
)
HELD_OUT = "held_out"
LONG_DECODE = "long_decode"
VRAM_CAPACITY_QUANTILES = (0.3, 0.5, 0.7)


def ordered_methods(names: Iterable[str]) -> tuple[str, ...]:
    present = set(names)
    return tuple(name for name in METHOD_ORDER if name in present)


@dataclass(frozen=True, slots=True)
class SchemeResult:
    anchor_count: int
    anchor_cost: ProfilingCost
    stratum_counts: Mapping[str, int]
    run_windows_sec: tuple[float, ...]
    vram_capacities_mb: tuple[float, ...]
    accuracy: Mapping[str, Mapping[str, Mapping[str, Accuracy]]]
    ordering: Mapping[str, Mapping[str, float]]
    decisions: Mapping[str, Mapping[str, tuple[ThresholdDecision, ...]]]
    bounds: Mapping[str, BoundQuality]


def evaluate_scheme(
    truth: Truth,
    gnn: Mapping[WorkflowModelFeatureKey, ResourceContract],
    anchors: Sequence[WorkflowModelFeatureKey],
    *,
    include_heuristic: bool = False,
) -> SchemeResult:
    """Score every method on the configurations the anchor budget did not buy."""
    methods = build_method_predictions(
        truth, gnn, anchors, include_heuristic=include_heuristic
    )
    held_out = held_out_keys(truth, anchors)
    strata = {HELD_OUT: held_out, LONG_DECODE: long_decode_keys(held_out)}
    run_windows = decision_thresholds(truth, held_out, RUN)
    capacities = decision_thresholds(truth, held_out, VRAM, VRAM_CAPACITY_QUANTILES)

    accuracy: dict[str, Mapping[str, Mapping[str, Accuracy]]] = {}
    ordering: dict[str, Mapping[str, float]] = {}
    decisions: dict[str, Mapping[str, tuple[ThresholdDecision, ...]]] = {}
    bounds: dict[str, BoundQuality] = {}
    for name in ordered_methods(methods):
        predictions = methods[name]
        accuracy[name] = {
            stratum: {
                target: score_accuracy(predictions.of(target), truth, keys, target)
                for target in TARGETS
            }
            for stratum, keys in strata.items()
        }
        ordering[name] = {
            stratum: pairwise_order_accuracy(predictions.run_sec, truth, keys)
            for stratum, keys in strata.items()
        }
        decisions[name] = {
            RUN: tuple(
                threshold_decision(predictions.run_sec, truth, held_out, RUN, window)
                for window in run_windows
            ),
            VRAM: tuple(
                threshold_decision(
                    predictions.peak_vram_mb, truth, held_out, VRAM, capacity
                )
                for capacity in capacities
            ),
        }
        bounds[name] = bound_quality(predictions.peak_vram_mb, truth, anchors, held_out)
    return SchemeResult(
        anchor_count=len(anchors),
        anchor_cost=profiling_cost(truth, anchors),
        stratum_counts={name: len(keys) for name, keys in strata.items()},
        run_windows_sec=run_windows,
        vram_capacities_mb=capacities,
        accuracy=accuracy,
        ordering=ordering,
        decisions=decisions,
        bounds=bounds,
    )
