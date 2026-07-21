from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from workflow.artifacts import (
    AcceleratorConfig,
    GpuKind,
    ResourceContract,
    ResourceContractCache,
    ResourceContractSource,
    ResourceEvidence,
    SchedulerConfig,
    load_resource_contract_cache,
)
from workflow.types import WorkflowModelFeatureKey


def prediction_key(
    *,
    model_name: str = "test-model",
    gpu_name: GpuKind = "v100",
    batch_size: int = 1,
    sequence_length: int = 2048,
    decode_output_length: int = 512,
) -> WorkflowModelFeatureKey:
    return WorkflowModelFeatureKey(
        model_name=model_name,
        phase="decode",
        gpu_name=gpu_name,
        batch_size=batch_size,
        sequence_length=sequence_length,
        decode_output_length=decode_output_length,
    )


def prediction_entry(
    key: WorkflowModelFeatureKey | None = None,
) -> ResourceContract:
    return ResourceContract(
        key=key or prediction_key(),
        source=ResourceContractSource.SYNTHETIC_FIXTURE,
        predicted_load_sec=5.0,
        predicted_run_sec=3.5,
        predicted_peak_vram_mb=12_000,
        peak_vram_mb_upper_bound=12_000,
        peak_vram_mb_evidence=ResourceEvidence(
            method="point_estimate_only", sample_count=1
        ),
    )


def test_prediction_cache_indexes_the_existing_feature_key() -> None:
    key = prediction_key()
    entry = prediction_entry(key)
    cache = ResourceContractCache(
        version=1,
        environment={"source": "gnn"},
        entries=(entry,),
    )

    assert cache.environment == {"source": "gnn"}
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
        ResourceContractCache(version=1, entries=(entry, entry))


def test_prediction_cache_reports_missing_contiguous_batch_keys() -> None:
    batch_one = prediction_entry(prediction_key(batch_size=1))
    batch_three = prediction_entry(prediction_key(batch_size=3))
    cache = ResourceContractCache(version=1, entries=(batch_one, batch_three))

    assert (
        cache.lookup_decode(
            model_name="test-model",
            gpu_kind="v100",
            batch_size=3,
            sequence_length=2048,
            decode_output_length=512,
        )
        == batch_three
    )
    missing = cache.missing_decode_batch_keys("test-model", "v100", 3)

    assert len(missing) == 1
    assert missing[0].batch_size == 2


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
    source: synthetic_fixture
    predicted_load_sec: 5.0
    predicted_run_sec: 3.5
    predicted_peak_vram_mb: 12000
    peak_vram_mb_upper_bound: 12000
    peak_vram_mb_evidence:
      method: point_estimate_only
      sample_count: 1
""".strip(),
        encoding="utf-8",
    )

    with pytest.raises(ValidationError, match="unknown"):
        load_resource_contract_cache(path)


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
        ResourceContract.model_validate(payload)


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
    predicted_load_sec: 5.0
    predicted_run_sec: 3.5
    predicted_peak_vram_mb: 12000
""".strip(),
        encoding="utf-8",
    )

    with pytest.raises(ValidationError, match="unknown prediction key fields"):
        load_resource_contract_cache(path)


def test_prediction_entry_requires_load_prediction() -> None:
    payload = prediction_entry().model_dump(mode="python")
    del payload["predicted_load_sec"]

    with pytest.raises(ValidationError, match="predicted_load_sec"):
        ResourceContract.model_validate(payload)


@pytest.mark.parametrize(
    "field", ["source", "peak_vram_mb_upper_bound", "peak_vram_mb_evidence"]
)
def test_prediction_entry_requires_resource_contract_fields(field: str) -> None:
    payload = prediction_entry().model_dump(mode="python")
    del payload[field]

    with pytest.raises(ValidationError, match=field):
        ResourceContract.model_validate(payload)


def test_prediction_entry_run_sec_bound_and_evidence_are_optional() -> None:
    entry = prediction_entry()

    assert entry.run_sec_upper_bound is None
    assert entry.run_sec_evidence is None


def test_prediction_entry_rejects_upper_bound_below_point_estimate() -> None:
    payload = prediction_entry().model_dump(mode="python")
    payload["peak_vram_mb_upper_bound"] = payload["predicted_peak_vram_mb"] - 1.0

    with pytest.raises(ValidationError, match="peak_vram_mb_upper_bound"):
        ResourceContract.model_validate(payload)


def test_prediction_entry_rejects_run_sec_upper_bound_below_point_estimate() -> None:
    payload = prediction_entry().model_dump(mode="python")
    payload["run_sec_upper_bound"] = payload["predicted_run_sec"] - 1.0

    with pytest.raises(ValidationError, match="run_sec_upper_bound"):
        ResourceContract.model_validate(payload)


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
    assert config.policy == "cache"


def test_prediction_and_accelerator_contracts_accept_generic_gpu_kinds() -> None:
    key = prediction_key(gpu_name="p100")
    accelerator = AcceleratorConfig(
        hostname="gpu-node-0",
        gpu_kind="p100",
        local_index=0,
        total_mem_mb=16_000,
    )

    assert key.gpu_name == "p100"
    assert accelerator.accelerator_id == "gpu-node-0/p100:0"


def test_scheduler_config_allows_no_accelerators() -> None:
    assert SchedulerConfig().accelerators == ()


def test_scheduler_config_preserves_virtualenv_python_symlink(tmp_path: Path) -> None:
    interpreter = tmp_path / "python3.12"
    interpreter.touch()
    virtualenv_python = tmp_path / "python"
    virtualenv_python.symlink_to(interpreter)

    config = SchedulerConfig(vllm_python_executable=str(virtualenv_python))
    assert config.vllm_python_executable is not None
    result = Path(config.vllm_python_executable)

    assert result == virtualenv_python.absolute()
    assert result.is_symlink()


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
        ("history_ema_alpha", 0),
        ("history_ema_alpha", 1.1),
    ],
)
def test_scheduler_config_rejects_invalid_numeric_bounds(
    field: str, value: float
) -> None:
    with pytest.raises(ValidationError):
        SchedulerConfig.model_validate({field: value})
