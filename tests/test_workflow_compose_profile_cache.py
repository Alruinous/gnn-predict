from __future__ import annotations

from pathlib import Path

from scripts.workflow.compose_profile_cache import (
    ProfileSource,
    compose_prediction_cache,
    parse_profile_source,
    write_prediction_cache,
)
from workflow.artifacts import PredictionCache, PredictionEntry
from workflow.types import WorkflowModelFeatureKey


def prediction(gpu_kind: str, run_sec: float) -> PredictionEntry:
    return PredictionEntry(
        key=WorkflowModelFeatureKey(
            model_name="test-model",
            phase="decode",
            gpu_name=gpu_kind,
            batch_size=1,
            sequence_length=128,
            decode_output_length=32,
        ),
        predicted_load_sec=2.0,
        predicted_run_sec=run_sec,
        predicted_peak_vram_mb=1024.0,
    )


def write_source(path: Path, entries: tuple[PredictionEntry, ...]) -> None:
    write_prediction_cache(
        path,
        PredictionCache(version=2, environment={}, entries=entries),
    )


def test_profile_source_accepts_generic_gpu_kind() -> None:
    source = parse_profile_source(" P100 = cache/p100.yaml ")

    assert source == ProfileSource("p100", Path("cache/p100.yaml"))


def test_compose_selects_only_the_declared_kind_from_each_source(
    tmp_path: Path,
) -> None:
    legacy_v100 = tmp_path / "legacy-v100.yaml"
    measured_a100 = tmp_path / "measured-a100.yaml"
    write_source(
        legacy_v100,
        (prediction("v100", 4.0), prediction("a100", 4.0)),
    )
    write_source(measured_a100, (prediction("a100", 2.0),))

    cache = compose_prediction_cache(
        [
            ProfileSource("v100", legacy_v100),
            ProfileSource("a100", measured_a100),
        ]
    )

    assert [(entry.key.gpu_name, entry.predicted_run_sec) for entry in cache.entries] == [
        ("a100", 2.0),
        ("v100", 4.0),
    ]
