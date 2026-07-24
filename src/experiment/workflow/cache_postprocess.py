"""Serving-domain post-processing of a GNN resource cache into a deployable v2.

Bridges raw graph predictions to what the scheduler actually consumes:
  raw GNN cache
    → scoped VRAM de-bias + calibrated upper-bound reservation (admission gate)
    → trace-driven warm-reload load calibration (prefetch / eviction timing)
    → deployable ResourceContractCache (base cache left immutable)

Load truth is the online vLLM `model_load_finished.payload.duration_sec` in
`output/serve*` traces. Cold (first load of a deployment in a run) is excluded;
only stable warm reloads calibrate the reload-cost estimate the scheduler needs.
Reuses `cache_replay` for trace parsing and `prediction_calibration` for the
VRAM primitives. See `docs/workflow/gnn_cache_postprocess_method_20260724.md`.
"""

from __future__ import annotations

import statistics
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from experiment.workflow.artifacts import file_sha256
from experiment.workflow.cache_replay import (
    LoadSample,
    TraceRun,
    discover_complete_trace_paths,
    read_trace_run,
)
from experiment.workflow.prediction_calibration import (
    Scope,
    apply_linear,
    fit_scoped_linear,
    fit_scoped_upper_quantile,
    scope_of,
    vram_features,
)
from workflow.artifacts import (
    ResourceContract,
    ResourceContractCache,
    ResourceEvidence,
    load_resource_contract_cache,
)
from workflow.types import WorkflowModelFeatureKey

MIN_CALIBRATION_RUNS = 3
MIN_WARM_SAMPLES = 5
VRAM_ALPHA = 0.05
VRAM_FLOOR_FRACTION = 0.10
ModelGpu = tuple[str, str]


@dataclass(frozen=True, slots=True)
class LoadTierValue:
    value: float
    run_count: int
    sample_count: int


@dataclass(frozen=True, slots=True)
class LoadCalibration:
    model_gpu: Mapping[ModelGpu, LoadTierValue]
    deployment: Mapping[ModelGpu, LoadTierValue]  # (model_key, gpu) — phase-2 / report
    gpu_scale: Mapping[str, float]
    trace_sha256: tuple[str, ...]
    warm_sample_total: int
    cold_sample_total: int


def _robust_medians(
    runs: Sequence[TraceRun],
    scope_key: str,
) -> dict[ModelGpu, LoadTierValue]:
    """`median_run(median_warm_in_run)` per scope. Cold = first load of each
    (model_key, gpu) within a run, marked by event order, never by threshold."""
    per_run_medians: dict[ModelGpu, list[float]] = defaultdict(list)
    sample_counts: dict[ModelGpu, int] = defaultdict(int)
    for run in runs:
        run_warm: dict[ModelGpu, list[float]] = defaultdict(list)
        cold_seen: set[ModelGpu] = set()
        for load in run.loads:
            deployment = (load.model_key, load.gpu_kind)
            if deployment not in cold_seen:
                cold_seen.add(deployment)
                continue
            scope = _scope(load, scope_key)
            run_warm[scope].append(load.duration_sec)
        for scope, values in run_warm.items():
            per_run_medians[scope].append(statistics.median(values))
            sample_counts[scope] += len(values)
    return {
        scope: LoadTierValue(
            value=statistics.median(medians),
            run_count=len(medians),
            sample_count=sample_counts[scope],
        )
        for scope, medians in per_run_medians.items()
    }


def _scope(load: LoadSample, scope_key: str) -> ModelGpu:
    if scope_key == "model_gpu":
        return (load.model_name, load.gpu_kind)
    return (load.model_key, load.gpu_kind)


def _cold_warm_totals(runs: Sequence[TraceRun]) -> tuple[int, int]:
    warm = cold = 0
    for run in runs:
        cold_seen: set[ModelGpu] = set()
        for load in run.loads:
            deployment = (load.model_key, load.gpu_kind)
            if deployment not in cold_seen:
                cold_seen.add(deployment)
                cold += 1
            else:
                warm += 1
    return warm, cold


def extract_load_calibration(
    trace_roots: Sequence[Path],
    base_cache: ResourceContractCache,
) -> LoadCalibration:
    paths = tuple(
        path for root in trace_roots for path in discover_complete_trace_paths(root)
    )
    if not paths:
        raise ValueError(f"no complete traces under {list(trace_roots)}")
    runs = tuple(read_trace_run(path) for path in paths)
    model_gpu = _robust_medians(runs, "model_gpu")
    deployment = _robust_medians(runs, "deployment")
    warm_total, cold_total = _cold_warm_totals(runs)
    original = _original_load_by_scope(base_cache)
    gpu_scale = _gpu_scale(model_gpu, original)
    return LoadCalibration(
        model_gpu=model_gpu,
        deployment=deployment,
        gpu_scale=gpu_scale,
        trace_sha256=tuple(file_sha256(path) for path in paths),
        warm_sample_total=warm_total,
        cold_sample_total=cold_total,
    )


def _original_load_by_scope(cache: ResourceContractCache) -> dict[ModelGpu, float]:
    values: dict[ModelGpu, float] = {}
    for entry in cache.entries:
        values.setdefault(
            (entry.key.model_name, entry.key.gpu_name), entry.predicted_load_sec
        )
    return values


def _gpu_scale(
    model_gpu: Mapping[ModelGpu, LoadTierValue],
    original: Mapping[ModelGpu, float],
) -> dict[str, float]:
    ratios: dict[str, list[float]] = defaultdict(list)
    for (model_name, gpu), tier in model_gpu.items():
        base = original.get((model_name, gpu))
        if base is None or base <= 0:
            continue
        if (
            tier.run_count >= MIN_CALIBRATION_RUNS
            and tier.sample_count >= MIN_WARM_SAMPLES
        ):
            ratios[gpu].append(tier.value / base)
    return {gpu: statistics.median(values) for gpu, values in ratios.items() if values}


def _calibrated_load(
    entry: ResourceContract, calibration: LoadCalibration
) -> tuple[float, str, int, int]:
    scope = (entry.key.model_name, entry.key.gpu_name)
    tier = calibration.model_gpu.get(scope)
    if (
        tier is not None
        and tier.run_count >= MIN_CALIBRATION_RUNS
        and tier.sample_count >= MIN_WARM_SAMPLES
    ):
        return tier.value, "warm_model_gpu", tier.run_count, tier.sample_count
    scale = calibration.gpu_scale.get(entry.key.gpu_name)
    if scale is not None:
        return entry.predicted_load_sec * scale, "gpu_scale", 0, 0
    return entry.predicted_load_sec, "original", 0, 0


@dataclass(frozen=True, slots=True)
class VramCalibration:
    coef_by_scope: Mapping[Scope, np.ndarray]
    quantile_by_scope: Mapping[Scope, float]


def fit_vram_calibration(
    base_cache: ResourceContractCache,
    reference: Mapping[WorkflowModelFeatureKey, ResourceContract],
) -> VramCalibration:
    rows: dict[Scope, list[tuple[list[float], float]]] = defaultdict(list)
    for entry in base_cache.entries:
        truth = reference.get(entry.key)
        if truth is None:
            continue
        scope = scope_of(entry.key)
        features = vram_features(entry.key, entry.predicted_peak_vram_mb)
        rows[scope].append((features, truth.predicted_peak_vram_mb))
    coefs = fit_scoped_linear(rows)
    residuals: dict[Scope, list[float]] = defaultdict(list)
    for entry in base_cache.entries:
        truth = reference.get(entry.key)
        if truth is None:
            continue
        scope = scope_of(entry.key)
        debiased = apply_linear(
            coefs[scope], vram_features(entry.key, entry.predicted_peak_vram_mb)
        )
        residuals[scope].append(truth.predicted_peak_vram_mb - debiased)
    quantiles = fit_scoped_upper_quantile(
        residuals, alpha=VRAM_ALPHA, floor_fraction=0.0
    )
    return VramCalibration(coef_by_scope=coefs, quantile_by_scope=quantiles)


def _calibrated_vram(
    entry: ResourceContract, calibration: VramCalibration
) -> tuple[float, float]:
    scope = scope_of(entry.key)
    debiased = apply_linear(
        calibration.coef_by_scope[scope],
        vram_features(entry.key, entry.predicted_peak_vram_mb),
    )
    debiased = max(debiased, 1.0)
    upper = max(
        debiased + calibration.quantile_by_scope[scope],
        debiased * (1.0 + VRAM_FLOOR_FRACTION),
    )
    return debiased, upper


def postprocess_cache(
    base_cache: ResourceContractCache,
    load_calibration: LoadCalibration,
    vram_calibration: VramCalibration | None,
    *,
    base_cache_sha256: str,
    reference_sha256: str | None,
) -> ResourceContractCache:
    """Rewrite load estimates from traces; when `vram_calibration` is given,
    also de-bias VRAM and reserve a calibrated upper bound. Load-only mode
    (`None`) leaves VRAM/run untouched — the right choice for the profile cache,
    whose empirical VRAM is itself the reference."""
    entries: list[ResourceContract] = []
    for entry in base_cache.entries:
        load_sec, load_tier, run_count, sample_count = _calibrated_load(
            entry, load_calibration
        )
        update: dict[str, object] = {"predicted_load_sec": max(load_sec, 1e-3)}
        if vram_calibration is not None:
            vram, vram_upper = _calibrated_vram(entry, vram_calibration)
            update["predicted_peak_vram_mb"] = vram
            update["peak_vram_mb_upper_bound"] = vram_upper
            update["peak_vram_mb_evidence"] = ResourceEvidence(
                method="repeated_sample_margin",
                sample_count=max(load_calibration.warm_sample_total, 1),
                margin_fraction=(vram_upper - vram) / vram,
            )
        metadata = dict(entry.predictor_metadata)
        metadata["postprocess"] = {
            "load_tier": load_tier,
            "load_run_count": run_count,
            "load_sample_count": sample_count,
            "load_source": (
                "trace_calibrated_vllm_warm"
                if load_tier != "original"
                else "empirical_profile"
            ),
            "vram_source": (
                "scoped_debias_plus_upper_quantile"
                if vram_calibration is not None
                else "unchanged_base"
            ),
        }
        update["predictor_metadata"] = metadata
        entries.append(entry.model_copy(update=update))
    return ResourceContractCache(
        version=base_cache.version,
        environment={
            **base_cache.environment,
            "postprocess_version": 2,
            "base_cache_sha256": base_cache_sha256,
            "vram_reference_sha256": reference_sha256,
            "vram_calibrated": vram_calibration is not None,
            "load_trace_sha256": list(load_calibration.trace_sha256),
            "load_source": "trace_calibrated_vllm_warm",
            "warm_sample_total": load_calibration.warm_sample_total,
            "cold_sample_total": load_calibration.cold_sample_total,
        },
        entries=tuple(entries),
    )


def build_v2_cache(
    base_cache_path: Path,
    trace_roots: Sequence[Path],
    reference_cache_path: Path | None = None,
    *,
    calibrate_vram: bool = True,
) -> tuple[ResourceContractCache, LoadCalibration]:
    base_cache = load_resource_contract_cache(base_cache_path)
    load_calibration = extract_load_calibration(trace_roots, base_cache)
    vram_calibration: VramCalibration | None = None
    reference_sha256: str | None = None
    if calibrate_vram:
        if reference_cache_path is None:
            raise ValueError("VRAM calibration requires a reference cache")
        reference = load_resource_contract_cache(reference_cache_path)
        reference_index = {entry.key: entry for entry in reference.entries}
        vram_calibration = fit_vram_calibration(base_cache, reference_index)
        reference_sha256 = file_sha256(reference_cache_path)
    v2 = postprocess_cache(
        base_cache,
        load_calibration,
        vram_calibration,
        base_cache_sha256=file_sha256(base_cache_path),
        reference_sha256=reference_sha256,
    )
    return v2, load_calibration
