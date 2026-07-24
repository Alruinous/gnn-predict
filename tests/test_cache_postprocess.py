from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from typing import cast

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from experiment.workflow.cache_postprocess import (
    LoadCalibration,
    LoadTierValue,
    _calibrated_load,
    _robust_medians,
    fit_vram_calibration,
    postprocess_cache,
)
from experiment.workflow.cache_replay import LoadSample, TraceRun
from workflow.artifacts import (
    ResourceContract,
    ResourceContractCache,
    ResourceContractSource,
    ResourceEvidence,
)
from workflow.types import WorkflowModelFeatureKey


def _load(model: str, key: str, gpu: str, duration: float) -> LoadSample:
    return LoadSample(
        model_name=model,
        model_key=key,
        gpu_kind=gpu,
        duration_sec=duration,
        cold_accelerator=False,
    )


def _run(loads: list[LoadSample]) -> TraceRun:
    # _robust_medians only reads `.loads`; a light stand-in keeps the test focused.
    return cast(TraceRun, SimpleNamespace(loads=tuple(loads)))


def test_cold_load_excluded_from_warm_median() -> None:
    run = _run(
        [
            _load("Qwen3-4B", "dep-a", "v100", 190.0),  # cold: first of dep-a
            _load("Qwen3-4B", "dep-a", "v100", 48.0),
            _load("Qwen3-4B", "dep-a", "v100", 52.0),
        ]
    )
    result = _robust_medians([run], "model_gpu")
    tier = result[("Qwen3-4B", "v100")]
    assert tier.sample_count == 2  # the 190 cold sample is dropped
    assert tier.value == 50.0  # median(48, 52), not median(190, 48, 52)


def test_robust_median_is_per_run_then_cross_run() -> None:
    run_a = _run(
        [
            _load("m", "dep", "v100", 0.0),  # cold
            _load("m", "dep", "v100", 40.0),
            _load("m", "dep", "v100", 60.0),
        ]
    )
    run_b = _run(
        [
            _load("m", "dep", "v100", 0.0),  # cold
            _load("m", "dep", "v100", 70.0),
        ]
    )
    result = _robust_medians([run_a, run_b], "model_gpu")
    tier = result[("m", "v100")]
    assert tier.run_count == 2
    assert tier.value == 60.0  # cross-run median(median(40,60)=50, 70) = 60


def _calibration(
    model_gpu: dict[tuple[str, str], LoadTierValue],
    gpu_scale: dict[str, float],
) -> LoadCalibration:
    return LoadCalibration(
        model_gpu=model_gpu,
        deployment={},
        gpu_scale=gpu_scale,
        trace_sha256=(),
        warm_sample_total=10,
        cold_sample_total=2,
    )


def _key(model: str, gpu: str, seq: int = 1024) -> WorkflowModelFeatureKey:
    return WorkflowModelFeatureKey(
        model_name=model,
        phase="decode",
        gpu_name=gpu,
        batch_size=1,
        sequence_length=seq,
        decode_output_length=128,
    )


def _contract(
    key: WorkflowModelFeatureKey, load: float, vram: float
) -> ResourceContract:
    return ResourceContract(
        key=key,
        source=ResourceContractSource.GNN_PREDICTED,
        predicted_load_sec=load,
        predicted_run_sec=5.0,
        predicted_peak_vram_mb=vram,
        peak_vram_mb_upper_bound=vram,
        peak_vram_mb_evidence=ResourceEvidence(
            method="point_estimate_only", sample_count=1
        ),
    )


def test_calibrated_load_tier_fallback() -> None:
    entry = _contract(_key("Qwen3-4B", "v100"), 190.0, 9000.0)
    warm = _calibration({("Qwen3-4B", "v100"): LoadTierValue(50.0, 4, 8)}, {})
    assert _calibrated_load(entry, warm)[:2] == (50.0, "warm_model_gpu")
    # too few runs -> fall to gpu scale
    thin = _calibration(
        {("Qwen3-4B", "v100"): LoadTierValue(50.0, 1, 2)}, {"v100": 0.25}
    )
    value, tier, _, _ = _calibrated_load(entry, thin)
    assert tier == "gpu_scale" and value == 190.0 * 0.25
    # no coverage -> original
    assert _calibrated_load(entry, _calibration({}, {}))[:2] == (190.0, "original")


def test_postprocess_is_nondestructive_and_debiases_vram() -> None:
    # GNN over-predicts vram by 1.3x vs the profile reference.
    keys = [_key("Qwen3-4B", "v100", seq=1024 + i) for i in range(12)]
    truths = [1000.0 * (1 + 0.05 * i) for i in range(12)]
    base = ResourceContractCache(
        version=2,
        environment={},
        entries=tuple(
            _contract(key, 190.0, truth * 1.3)
            for key, truth in zip(keys, truths, strict=True)
        ),
    )
    reference = {
        key: _contract(key, 190.0, truth)
        for key, truth in zip(keys, truths, strict=True)
    }
    vram_cal = fit_vram_calibration(base, reference)
    load_cal = _calibration({("Qwen3-4B", "v100"): LoadTierValue(50.0, 4, 8)}, {})
    v2 = postprocess_cache(
        base,
        load_cal,
        vram_cal,
        base_cache_sha256="base",
        reference_sha256="ref",
    )
    for entry in v2.entries:
        truth = reference[entry.key].predicted_peak_vram_mb
        assert abs(entry.predicted_peak_vram_mb - truth) < 0.1 * truth  # debiased
        assert entry.peak_vram_mb_upper_bound >= entry.predicted_peak_vram_mb
        assert entry.predicted_load_sec == 50.0
        postprocess = entry.predictor_metadata["postprocess"]
        assert isinstance(postprocess, dict)
        assert postprocess["load_tier"] == "warm_model_gpu"
    # base cache object is untouched
    assert base.entries[0].predicted_load_sec == 190.0


def test_load_only_calibrates_load_but_keeps_vram() -> None:
    key = _key("Qwen3-4B", "v100")
    base = ResourceContractCache(
        version=2, environment={}, entries=(_contract(key, 190.0, 9000.0),)
    )
    load_cal = _calibration({("Qwen3-4B", "v100"): LoadTierValue(50.0, 4, 8)}, {})
    v2 = postprocess_cache(
        base, load_cal, None, base_cache_sha256="b", reference_sha256=None
    )
    entry = v2.entries[0]
    assert entry.predicted_load_sec == 50.0  # load calibrated from traces
    assert entry.predicted_peak_vram_mb == 9000.0  # empirical VRAM untouched
    assert entry.peak_vram_mb_upper_bound == 9000.0
    postprocess = entry.predictor_metadata["postprocess"]
    assert isinstance(postprocess, dict)
    assert postprocess["vram_source"] == "unchanged_base"
    assert v2.environment["vram_calibrated"] is False
