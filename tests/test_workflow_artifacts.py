from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from workflow.artifacts import (
    AcceleratorConfig,
    DeploymentProfile,
    DeploymentProfileEntry,
    GpuKind,
    PredictionCache,
    PredictionEntry,
    SchedulerConfig,
    load_deployment_profile,
    load_prediction_cache,
)
from workflow.types import WorkflowModelFeatureKey


def prediction_key(
    *,
    model_name: str = "test-model",
    gpu_name: GpuKind = "v100",
    sequence_length: int = 2048,
    decode_output_length: int = 512,
) -> WorkflowModelFeatureKey:
    return WorkflowModelFeatureKey(
        model_name=model_name,
        phase="decode",
        gpu_name=gpu_name,
        batch_size=1,
        sequence_length=sequence_length,
        decode_output_length=decode_output_length,
    )


def prediction_entry(
    key: WorkflowModelFeatureKey | None = None,
) -> PredictionEntry:
    return PredictionEntry(
        key=key or prediction_key(),
        predicted_run_sec=3.5,
        predicted_peak_vram_mb=12_000,
    )


def test_prediction_cache_indexes_the_existing_feature_key() -> None:
    key = prediction_key()
    entry = prediction_entry(key)
    cache = PredictionCache(version=1, entries=(entry,))

    assert isinstance(cache.entries[0].key, WorkflowModelFeatureKey)
    assert cache.lookup(key) == entry
    assert (
        cache.lookup_decode(
            model_name="test-model",
            gpu_kind="v100",
            sequence_length=2048,
            decode_output_length=512,
        )
        == entry
    )


def test_prediction_cache_rejects_duplicate_keys() -> None:
    entry = prediction_entry()

    with pytest.raises(ValidationError, match="duplicate prediction key"):
        PredictionCache(version=1, entries=(entry, entry))


def test_prediction_loader_is_strict(tmp_path: Path) -> None:
    path = tmp_path / "predictions.yaml"
    path.write_text(
        """
version: 1
unknown: rejected
entries:
  - key:
      model_name: test-model
      phase: decode
      gpu_name: v100
      batch_size: 1
      sequence_length: 2048
      decode_output_length: 512
    predicted_run_sec: 3.5
    predicted_peak_vram_mb: 12000
""".strip(),
        encoding="utf-8",
    )

    with pytest.raises(ValidationError, match="unknown"):
        load_prediction_cache(path)


def test_prediction_entry_rejects_unknown_nested_key_fields() -> None:
    payload = {
        "key": {
            "model_name": "test-model",
            "phase": "decode",
            "gpu_name": "v100",
            "batch_size": 1,
            "sequence_length": 2048,
            "decode_output_length": 512,
            "unknown": "rejected",
        },
        "predicted_run_sec": 3.5,
        "predicted_peak_vram_mb": 12_000,
    }

    with pytest.raises(ValidationError, match="unknown prediction key fields"):
        PredictionEntry.model_validate(payload)


def test_prediction_loader_rejects_unknown_nested_key_fields(tmp_path: Path) -> None:
    path = tmp_path / "predictions.yaml"
    path.write_text(
        """
version: 1
entries:
  - key:
      model_name: test-model
      phase: decode
      gpu_name: v100
      batch_size: 1
      sequence_length: 2048
      decode_output_length: 512
      unknown: rejected
    predicted_run_sec: 3.5
    predicted_peak_vram_mb: 12000
""".strip(),
        encoding="utf-8",
    )

    with pytest.raises(ValidationError, match="unknown prediction key fields"):
        load_prediction_cache(path)


def test_deployment_profile_uses_p95_then_median_then_default() -> None:
    profile = DeploymentProfile(
        version=1,
        entries=(
            DeploymentProfileEntry(
                model_name="model-a",
                model_path="/models/a",
                dtype="float16",
                gpu_kind="v100",
                load_sec_median=10,
                load_sec_p95=12,
            ),
            DeploymentProfileEntry(
                model_name="model-b",
                model_path="/models/b",
                dtype="bfloat16",
                gpu_kind="a100",
                load_sec_median=7,
            ),
        ),
    )

    assert profile.load_cost_sec("model-a", "/models/a", "float16", "v100", 30) == 12
    assert profile.load_cost_sec("model-b", "/models/b", "bfloat16", "a100", 30) == 7
    assert (
        profile.load_cost_sec("missing", "/models/missing", "float16", "v100", 30) == 30
    )


def test_deployment_profile_rejects_duplicate_keys() -> None:
    entry = DeploymentProfileEntry(
        model_name="model-a",
        model_path="/models/a",
        dtype="float16",
        gpu_kind="v100",
        load_sec_median=10,
    )

    with pytest.raises(ValidationError, match="duplicate deployment profile key"):
        DeploymentProfile(version=1, entries=(entry, entry))


def test_deployment_profile_loader_accepts_the_manifest_shape(tmp_path: Path) -> None:
    path = tmp_path / "profile.yaml"
    path.write_text(
        """
version: 1
environment:
  storage_kind: shared_model_dir
  profile_scope: gpu_kind
entries:
  - model_name: model-a
    model_path: /models/a
    dtype: float16
    gpu_kind: v100
    samples: 3
    load_sec_median: 10
    load_sec_p95: 12
    idle_vram_mb_median: 8000
""".strip(),
        encoding="utf-8",
    )

    profile = load_deployment_profile(path)

    assert profile.load_cost_sec("model-a", "/models/a", "float16", "v100", 30) == 12


def test_accelerator_identity_and_scheduler_defaults_are_deterministic() -> None:
    accelerator = AcceleratorConfig(
        hostname="gpu-node-0",
        gpu_kind="v100",
        local_index=1,
        total_mem_mb=16_000,
    )

    config = SchedulerConfig(accelerators=(accelerator,))

    assert accelerator.accelerator_id == "gpu-node-0/v100:1"
    assert config.eps_mem_mb == 512.0
    assert config.hit_limit_rate_threshold == 0.1


def test_scheduler_config_allows_no_accelerators() -> None:
    assert SchedulerConfig().accelerators == ()


def test_scheduler_config_rejects_conflicting_host_local_indexes() -> None:
    first = AcceleratorConfig(
        hostname="gpu-node-0",
        gpu_kind="v100",
        local_index=0,
        total_mem_mb=16_000,
    )
    conflicting = AcceleratorConfig(
        hostname="gpu-node-0",
        gpu_kind="a100",
        local_index=0,
        total_mem_mb=40_000,
    )

    with pytest.raises(ValidationError, match="host/local accelerator indexes"):
        SchedulerConfig(accelerators=(first, conflicting))


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("eps_mem_mb", -1),
        ("eps_time_sec", -1),
        ("max_tick_interval_sec", 0),
        ("acquire_timeout_sec", 0),
        ("grant_poll_interval_sec", 0),
        ("eviction_timeout_sec", 0),
        ("default_load_sec", 0),
        ("history_ema_alpha", 0),
        ("history_ema_alpha", 1.1),
        ("hit_limit_rate_threshold", -0.1),
        ("hit_limit_rate_threshold", 1.1),
    ],
)
def test_scheduler_config_rejects_invalid_numeric_bounds(
    field: str, value: float
) -> None:
    with pytest.raises(ValidationError):
        SchedulerConfig.model_validate({field: value})
