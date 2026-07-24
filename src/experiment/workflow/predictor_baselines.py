"""Memorization / learned-tabular baselines built from the profile cache.

These share one deterministic train/test split so every method — static,
config-mean, nearest-profile, tabular-GBDT, and the frozen GNN cache — is scored
on the identical held-out keys against the empirical profile ground truth.
"""

from __future__ import annotations

import hashlib
import math
import random
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

from experiment.workflow.static_cache import STATIC_ARCH, STATIC_GPU
from workflow.artifacts import (
    ResourceContract,
    ResourceContractCache,
    ResourceContractSource,
    ResourceEvidence,
)
from workflow.types import WorkflowModelFeatureKey

TARGET_FIELDS = (
    "predicted_load_sec",
    "predicted_run_sec",
    "predicted_peak_vram_mb",
    "predicted_power_watts",
)
SPLIT_SEED = 7
VRAM_MARGIN_FRACTION = 0.10


def _key_order(key: WorkflowModelFeatureKey) -> tuple[str, str, str, int, int, int]:
    return (
        key.model_name,
        key.gpu_name,
        key.phase,
        key.batch_size,
        key.sequence_length,
        key.decode_output_length,
    )


def deterministic_split(
    keys: Iterable[WorkflowModelFeatureKey],
    *,
    test_fraction: float = 0.2,
    seed: int = SPLIT_SEED,
) -> tuple[tuple[WorkflowModelFeatureKey, ...], tuple[WorkflowModelFeatureKey, ...]]:
    ordered = sorted(keys, key=_key_order)
    shuffled = ordered[:]
    random.Random(seed).shuffle(shuffled)
    n_test = round(len(shuffled) * test_fraction)
    test = set(shuffled[:n_test])
    train = tuple(key for key in ordered if key not in test)
    test_keys = tuple(key for key in ordered if key in test)
    return train, test_keys


def stratified_anchor_sample(
    train_keys: Iterable[WorkflowModelFeatureKey],
    per_cell: int,
    *,
    seed: int = SPLIT_SEED,
) -> tuple[WorkflowModelFeatureKey, ...]:
    """Pick `per_cell` profiling anchors per (model, gpu) — the budget knob.

    Mirrors how the GNN cache was calibrated (5 runtime anchors per model_gpu):
    memorization baselines only see this many real profiles, so accuracy at a
    fixed anchor budget isolates generalization from grid memorization.
    """
    cells: dict[tuple[str, str], list[WorkflowModelFeatureKey]] = {}
    for key in sorted(train_keys, key=_key_order):
        cells.setdefault((key.model_name, key.gpu_name), []).append(key)
    anchors: list[WorkflowModelFeatureKey] = []
    for cell, cell_keys in sorted(cells.items()):
        # hashlib (not tuple.__hash__) so anchors are reproducible across processes.
        payload = f"{seed}|{cell[0]}|{cell[1]}".encode()
        cell_seed = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
        shuffled = cell_keys[:]
        random.Random(cell_seed).shuffle(shuffled)
        anchors.extend(shuffled[:per_cell])
    return tuple(anchors)


@dataclass(frozen=True, slots=True)
class TargetValues:
    load_sec: float
    run_sec: float
    peak_vram_mb: float
    power_watts: float


def profile_targets(contract: ResourceContract) -> TargetValues:
    return TargetValues(
        load_sec=contract.predicted_load_sec,
        run_sec=contract.predicted_run_sec,
        peak_vram_mb=contract.predicted_peak_vram_mb,
        power_watts=contract.predicted_power_watts or 0.0,
    )


def _contract(
    key: WorkflowModelFeatureKey, values: TargetValues, method: str
) -> ResourceContract:
    vram = max(values.peak_vram_mb, 1e-6)
    return ResourceContract(
        key=key,
        source=ResourceContractSource.SYNTHETIC_FIXTURE,
        predicted_load_sec=max(values.load_sec, 1e-6),
        predicted_run_sec=max(values.run_sec, 1e-6),
        predicted_peak_vram_mb=vram,
        peak_vram_mb_upper_bound=vram * (1.0 + VRAM_MARGIN_FRACTION),
        peak_vram_mb_evidence=ResourceEvidence(
            method="fixed_margin_fallback",
            sample_count=1,
            margin_fraction=VRAM_MARGIN_FRACTION,
        ),
        predicted_power_watts=max(values.power_watts, 0.0),
        predictor_metadata={"method": method},
    )


def _cache(entries: Sequence[ResourceContract], method: str) -> ResourceContractCache:
    return ResourceContractCache(
        version=2,
        environment={"source": method, "split_seed": SPLIT_SEED},
        entries=tuple(entries),
    )


def build_config_mean_cache(
    truth: Mapping[WorkflowModelFeatureKey, ResourceContract],
    train_keys: Sequence[WorkflowModelFeatureKey],
    all_keys: Sequence[WorkflowModelFeatureKey],
) -> ResourceContractCache:
    sums: dict[tuple[str, str], list[float]] = {}
    counts: dict[tuple[str, str], int] = {}
    for key in train_keys:
        cell = (key.model_name, key.gpu_name)
        values = profile_targets(truth[key])
        acc = sums.setdefault(cell, [0.0, 0.0, 0.0, 0.0])
        acc[0] += values.load_sec
        acc[1] += values.run_sec
        acc[2] += values.peak_vram_mb
        acc[3] += values.power_watts
        counts[cell] = counts.get(cell, 0) + 1
    means = {
        cell: TargetValues(
            load_sec=acc[0] / counts[cell],
            run_sec=acc[1] / counts[cell],
            peak_vram_mb=acc[2] / counts[cell],
            power_watts=acc[3] / counts[cell],
        )
        for cell, acc in sums.items()
    }
    entries = [
        _contract(key, means[(key.model_name, key.gpu_name)], "config_mean")
        for key in all_keys
    ]
    return _cache(entries, "config_mean")


def _nn_coords(key: WorkflowModelFeatureKey) -> tuple[float, float, float]:
    return (
        math.log2(key.batch_size),
        math.log2(key.sequence_length),
        math.log2(max(key.decode_output_length, 1)),
    )


def build_nearest_profile_cache(
    truth: Mapping[WorkflowModelFeatureKey, ResourceContract],
    train_keys: Sequence[WorkflowModelFeatureKey],
    all_keys: Sequence[WorkflowModelFeatureKey],
) -> ResourceContractCache:
    by_cell: dict[tuple[str, str], list[WorkflowModelFeatureKey]] = {}
    for key in train_keys:
        by_cell.setdefault((key.model_name, key.gpu_name), []).append(key)
    entries: list[ResourceContract] = []
    for key in all_keys:
        candidates = by_cell.get((key.model_name, key.gpu_name))
        assert candidates, f"no train profile for {key.model_name}/{key.gpu_name}"
        target = _nn_coords(key)
        nearest = min(
            candidates,
            key=lambda cand: sum(
                (a - b) ** 2 for a, b in zip(_nn_coords(cand), target, strict=True)
            ),
        )
        entries.append(
            _contract(key, profile_targets(truth[nearest]), "nearest_profile")
        )
    return _cache(entries, "nearest_profile")


def _tabular_features(key: WorkflowModelFeatureKey) -> list[float]:
    arch = STATIC_ARCH[key.model_name]
    gpu = STATIC_GPU[key.gpu_name]
    return [
        arch.params / 1e9,
        float(arch.num_layers),
        gpu.mem_bandwidth_bytes_s / 1e12,
        float(key.batch_size),
        math.log2(key.sequence_length),
        math.log2(max(key.decode_output_length, 1)),
    ]


def build_tabular_cache(
    truth: Mapping[WorkflowModelFeatureKey, ResourceContract],
    train_keys: Sequence[WorkflowModelFeatureKey],
    all_keys: Sequence[WorkflowModelFeatureKey],
) -> ResourceContractCache:
    x_train = np.array([_tabular_features(key) for key in train_keys], dtype=float)
    target_names = ("load_sec", "run_sec", "peak_vram_mb", "power_watts")
    models: dict[str, HistGradientBoostingRegressor] = {}
    for name, field in zip(target_names, TARGET_FIELDS, strict=True):
        y = np.array(
            [getattr(truth[key], field) or 0.0 for key in train_keys], dtype=float
        )
        model = HistGradientBoostingRegressor(random_state=SPLIT_SEED, max_iter=300)
        model.fit(x_train, y)
        models[name] = model
    x_all = np.array([_tabular_features(key) for key in all_keys], dtype=float)
    preds = {name: models[name].predict(x_all) for name in target_names}
    entries = [
        _contract(
            key,
            TargetValues(
                load_sec=float(preds["load_sec"][idx]),
                run_sec=float(preds["run_sec"][idx]),
                peak_vram_mb=float(preds["peak_vram_mb"][idx]),
                power_watts=float(preds["power_watts"][idx]),
            ),
            "tabular_hgbr",
        )
        for idx, key in enumerate(all_keys)
    ]
    return _cache(entries, "tabular_hgbr")
