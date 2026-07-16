from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from experiment.workflow.artifacts import file_sha256, write_json_exclusive
from workflow.artifacts import PredictionCache, PredictionEntry
from workflow.types import WorkflowModelFeatureKey

CACHE_GENERATOR_VERSION = 1


@dataclass(frozen=True, slots=True)
class SyntheticModelProfile:
    model_name: str
    sequence_lengths: tuple[int, ...]
    output_lengths: tuple[int, ...]
    load_sec: float
    base_vram_mb: float
    power_watts: float
    seconds_per_input_token: float
    seconds_per_output_token: float


SYNTHETIC_PROFILES = (
    SyntheticModelProfile(
        model_name="Qwen3-4B",
        sequence_lengths=(512, 1024, 1536, 2048, 4096, 7168),
        output_lengths=(384, 512),
        load_sec=20.0,
        base_vram_mb=31_145.0,
        power_watts=220.0,
        seconds_per_input_token=0.0010,
        seconds_per_output_token=0.018,
    ),
    SyntheticModelProfile(
        model_name="Qwen3-8B",
        sequence_lengths=(512, 1024, 1536, 2048, 3584),
        output_lengths=(384, 512),
        load_sec=34.0,
        base_vram_mb=31_340.0,
        power_watts=250.0,
        seconds_per_input_token=0.0017,
        seconds_per_output_token=0.030,
    ),
    SyntheticModelProfile(
        model_name="Qwen3-14B",
        sequence_lengths=(512,),
        output_lengths=(512,),
        load_sec=52.0,
        base_vram_mb=31_630.0,
        power_watts=280.0,
        seconds_per_input_token=0.0030,
        seconds_per_output_token=0.052,
    ),
)


def build_synthetic_prediction_cache() -> PredictionCache:
    entries = tuple(
        _synthetic_entry(profile, batch_size, sequence_length, output_length)
        for profile in SYNTHETIC_PROFILES
        for batch_size in (1, 2, 3)
        for sequence_length in profile.sequence_lengths
        for output_length in profile.output_lengths
    )
    return PredictionCache(
        version=CACHE_GENERATOR_VERSION,
        environment={
            "source": "frozen_synthetic",
            "gpu_kind": "v100",
            "purpose": "workflow_scheduler_mechanism_test",
        },
        entries=entries,
    )


def write_synthetic_prediction_cache(path: Path) -> str:
    cache = build_synthetic_prediction_cache()
    write_json_exclusive(path, cache)
    return file_sha256(path)


def _synthetic_entry(
    profile: SyntheticModelProfile,
    batch_size: int,
    sequence_length: int,
    output_length: int,
) -> PredictionEntry:
    batch_factor = 1.0 + 0.65 * (batch_size - 1)
    return PredictionEntry(
        key=WorkflowModelFeatureKey(
            model_name=profile.model_name,
            phase="decode",
            gpu_name="v100",
            batch_size=batch_size,
            sequence_length=sequence_length,
            decode_output_length=output_length,
        ),
        predicted_load_sec=profile.load_sec,
        predicted_run_sec=batch_factor
        * (
            sequence_length * profile.seconds_per_input_token
            + output_length * profile.seconds_per_output_token
        ),
        predicted_peak_vram_mb=profile.base_vram_mb + 16.0 * (batch_size - 1),
        predicted_power_watts=profile.power_watts,
        predictor_metadata={
            "source": "frozen_synthetic",
            "generator_version": CACHE_GENERATOR_VERSION,
        },
    )
