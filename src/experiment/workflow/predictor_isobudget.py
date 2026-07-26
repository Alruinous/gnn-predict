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

from experiment.workflow.prediction_calibration import Scope, scope_of
from experiment.workflow.predictor_baselines import (
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
LOAD = "load_sec"
TARGETS = (LOAD, RUN, VRAM)
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
    if target == LOAD:
        return contract.predicted_load_sec
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
    load_sec: Predictions
    run_sec: Predictions
    peak_vram_mb: Predictions

    def of(self, target: str) -> Predictions:
        if target == RUN:
            return self.run_sec
        if target == LOAD:
            return self.load_sec
        assert target == VRAM, f"unsupported target {target}"
        return self.peak_vram_mb


def _from_cache(cache: ResourceContractCache) -> MethodPredictions:
    entries = cache.entries
    return MethodPredictions(
        load_sec={e.key: e.predicted_load_sec for e in entries},
        run_sec={e.key: e.predicted_run_sec for e in entries},
        peak_vram_mb={e.key: e.predicted_peak_vram_mb for e in entries},
    )


PREDICTION_FLOOR = 1e-6


def anchor_affine(
    base: Predictions,
    truth: Truth,
    anchors: Sequence[WorkflowModelFeatureKey],
    target: str,
) -> Predictions:
    """Per-cell affine correction (scale and intercept) fit on that cell's anchors.

    Two parameters per model-GPU cell is what a five-anchor budget supports, and
    it is the one recipe every method receives, so the comparison is between
    predictors rather than between calibration efforts. The richer scoped fit in
    `prediction_calibration` needs eight rows per scope, so at this budget it
    silently collapses to a single pooled correction."""
    per_cell: dict[Scope, list[WorkflowModelFeatureKey]] = defaultdict(list)
    for key in anchors:
        per_cell[scope_of(key)].append(key)

    def solve(keys: Sequence[WorkflowModelFeatureKey]) -> np.ndarray:
        design = np.array([[base[key], 1.0] for key in keys])
        observed = np.array([target_value(truth[key], target) for key in keys])
        return np.linalg.lstsq(design, observed, rcond=None)[0]

    coefficients = {scope: solve(keys) for scope, keys in per_cell.items()}
    pooled = solve(anchors)
    return {
        key: max(
            float(coefficients.get(scope_of(key), pooled) @ np.array([value, 1.0])),
            PREDICTION_FLOOR,
        )
        for key, value in base.items()
    }


def anchor_calibrated(
    base: MethodPredictions, truth: Truth, anchors: Sequence[WorkflowModelFeatureKey]
) -> MethodPredictions:
    """Apply the shared per-cell affine correction to every target."""
    return MethodPredictions(
        **{
            target: anchor_affine(base.of(target), truth, anchors, target)
            for target in TARGETS
        }
    )


def build_sageradar_predictions(
    gnn: Mapping[WorkflowModelFeatureKey, ResourceContract],
    truth: Truth,
    anchors: Sequence[WorkflowModelFeatureKey],
) -> MethodPredictions:
    """Run time is the frozen calibrated cache; VRAM gets the same anchor budget."""
    raw_vram = {key: entry.predicted_peak_vram_mb for key, entry in gnn.items()}
    return MethodPredictions(
        load_sec={key: entry.predicted_load_sec for key, entry in gnn.items()},
        run_sec={key: entry.predicted_run_sec for key, entry in gnn.items()},
        peak_vram_mb=anchor_affine(raw_vram, truth, anchors, VRAM),
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
        load_sec={
            key: target_value(truth[donor], LOAD) for key, donor in donors.items()
        },
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
    methods = {
        "sageradar": build_sageradar_predictions(gnn, truth, anchors),
        "analytical": anchor_calibrated(
            _from_cache(build_static_prediction_cache(all_keys)), truth, anchors
        ),
        "tabular": anchor_calibrated(
            _from_cache(build_tabular_cache(truth, anchors, all_keys)), truth, anchors
        ),
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
    alpha: float
    violation_rate: float
    mean_over_reservation: float
    p95_over_reservation: float


def bound_quality(
    predictions: Predictions,
    truth: Truth,
    anchors: Sequence[WorkflowModelFeatureKey],
    keys: Sequence[WorkflowModelFeatureKey],
    *,
    alpha: float = 0.05,
    floor_fraction: float = 0.02,
) -> BoundQuality:
    """Turn each point estimate into an admission bound with the same recipe — a
    split-conformal upper margin on the anchors — then compare safety against
    wasted reservation.

    Residuals are pooled across cells after dividing by the cell's measured VRAM
    scale. Pooling is what makes the target reachable: a distribution-free bound
    from n points cannot miscover less than 1/(n+1), so the five anchors in one
    cell floor at 17% while all fifty floor at 2%."""
    scale = _anchor_scale(truth, anchors)
    residuals = sorted(
        (target_value(truth[key], VRAM) - predictions[key]) / scale[scope_of(key)]
        for key in anchors
    )
    rank = min(math.ceil((len(residuals) + 1) * (1.0 - alpha)) - 1, len(residuals) - 1)
    margin = residuals[rank]
    pooled_scale = statistics.fmean(scale.values())

    observed = np.array([target_value(truth[key], VRAM) for key in keys])
    bounds = np.array(
        [
            max(
                predictions[key] + margin * scale.get(scope_of(key), pooled_scale),
                predictions[key] * (1.0 + floor_fraction),
            )
            for key in keys
        ]
    )
    slack = (bounds - observed) / observed
    return BoundQuality(
        alpha=alpha,
        violation_rate=float((observed > bounds).mean()),
        mean_over_reservation=float(slack.mean()),
        p95_over_reservation=float(np.quantile(slack, 0.95)),
    )


def _anchor_scale(
    truth: Truth, anchors: Sequence[WorkflowModelFeatureKey]
) -> dict[Scope, float]:
    """Per-cell VRAM magnitude, so residuals from a 2GB and a 31GB cell pool."""
    per_cell: dict[Scope, list[float]] = defaultdict(list)
    for key in anchors:
        per_cell[scope_of(key)].append(target_value(truth[key], VRAM))
    return {scope: statistics.fmean(values) for scope, values in per_cell.items()}


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
        "analytical": lambda: _from_cache(build_static_prediction_cache(all_keys)),
        "tabular": lambda: _from_cache(build_tabular_cache(truth, anchors, all_keys)),
    }
    for name, build in builders.items():
        start = time.perf_counter()
        anchor_calibrated(build(), truth, anchors)
        timings[name] = (time.perf_counter() - start) / count
    return timings


METHOD_ORDER = (
    "sageradar",
    "analytical",
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


CANONICAL_METRICS_PATH = Path("cache/canonical_metrics.json")
CANONICAL_SCHEMA = 1


def canonical_payload(
    result: SchemeResult,
    profile_cost: ProfilingCost,
    marginal_cost_sec: Mapping[str, float],
    *,
    sources: Mapping[str, str],
) -> dict[str, object]:
    """The frozen headline numbers every table and figure must quote.

    Presentation reads this file; it never recomputes. Regenerate it only with
    `eval_predictor_isobudget.py --write-canonical`."""
    windows = result.run_windows_sec
    median_window = windows[len(windows) // 2]
    methods: dict[str, dict[str, float]] = {}
    for name, per_stratum in result.accuracy.items():
        if name not in marginal_cost_sec:  # excluded heuristic: never presented
            continue
        decisions = result.decisions[name][RUN]
        fit = next(d for d in decisions if d.threshold == median_window)
        methods[name] = {
            "generate_wape": per_stratum[HELD_OUT][RUN].wape,
            "generate_wape_long_output": per_stratum[LONG_DECODE][RUN].wape,
            "generate_p95_relative_error": per_stratum[LONG_DECODE][
                RUN
            ].p95_relative_error,
            "vram_wape": per_stratum[HELD_OUT][VRAM].wape,
            "order_accuracy": result.ordering[name][HELD_OUT],
            "bubble_fit_accuracy": fit.accuracy,
            "bound_violation_rate": result.bounds[name].violation_rate,
            "bound_mean_over_reservation": result.bounds[name].mean_over_reservation,
            "bound_p95_over_reservation": result.bounds[name].p95_over_reservation,
            "marginal_cost_sec": marginal_cost_sec[name],
        }
    return {
        "schema": CANONICAL_SCHEMA,
        "experiment": "predictor_isobudget",
        "sources": dict(sources),
        "grid": {
            "configurations": profile_cost.config_count,
            "held_out": result.stratum_counts[HELD_OUT],
            "long_output": result.stratum_counts[LONG_DECODE],
            "long_output_threshold": LONG_DECODE_OUTPUT,
        },
        "full_profiling": {
            "gpu_hours": profile_cost.total_gpu_hours,
            "marginal_sec_per_config": profile_cost.marginal_sec_per_config,
        },
        "anchor_budget": {
            "measurements": result.anchor_count,
            "gpu_hours": result.anchor_cost.total_gpu_hours,
            "share_of_full_grid": result.anchor_cost.total_gpu_hours
            / profile_cost.total_gpu_hours,
        },
        "median_bubble_window_sec": median_window,
        "methods": methods,
    }


def load_canonical_metrics(path: Path = CANONICAL_METRICS_PATH) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["schema"] == CANONICAL_SCHEMA, (
        f"canonical metrics schema {payload['schema']} != {CANONICAL_SCHEMA}"
    )
    return payload
