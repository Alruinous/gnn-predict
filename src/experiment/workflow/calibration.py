from __future__ import annotations

import statistics
from pathlib import Path
from typing import Literal, cast

from pydantic import BaseModel, ConfigDict, Field, JsonValue, PositiveFloat, PositiveInt

from experiment.workflow.analysis import (
    steady_state_completion_slope,
    summarize_trace,
)
from experiment.workflow.artifacts import (
    EnvironmentManifest,
    canonical_json,
    file_sha256,
    read_json,
    validate_completion,
)
from experiment.workflow.config import ExperimentConfig, Scenario
from experiment.workflow.environment import ServingEnvironment
from experiment.workflow.telemetry import TelemetryConfig

CalibrationRepetition = Literal[1, 2, 3]


class CalibrationModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class CalibrationRunManifest(CalibrationModel):
    version: PositiveInt = 1
    calibration_key: str
    scenario: Scenario
    gpu_count: Literal[1, 2, 3]
    repetition: CalibrationRepetition
    seed: int
    session_count: PositiveInt = 180
    concurrency: PositiveInt = 60
    sample_manifest_sha256: str
    prediction_cache_sha256: str
    workflow_config: dict[str, JsonValue]
    scheduler_config: dict[str, JsonValue]
    serving_environment: ServingEnvironment
    telemetry_config: TelemetryConfig
    environment: EnvironmentManifest
    created_at: float = Field(ge=0)


class CalibrationRunReference(CalibrationModel):
    repetition: CalibrationRepetition
    trace_sha256: str
    completion_slope_sessions_per_sec: PositiveFloat


class CapacitySummary(CalibrationModel):
    version: PositiveInt = 1
    calibration_key: str
    scenario: Scenario
    gpu_count: Literal[1, 2, 3]
    steady_state_start_fraction: float
    steady_state_end_fraction: float
    capacity_sessions_per_sec: PositiveFloat
    runs: tuple[
        CalibrationRunReference, CalibrationRunReference, CalibrationRunReference
    ]


def calibration_key(scenario: Scenario, gpu_count: int) -> str:
    valid = (scenario == "qmsum" and gpu_count == 2) or (
        scenario == "mbpp" and gpu_count in (1, 2, 3)
    )
    if not valid:
        raise ValueError("unsupported formal capacity calibration")
    return f"{scenario}__wf-cache__g{gpu_count}"


def calibration_run_dir(
    config: ExperimentConfig,
    scenario: Scenario,
    gpu_count: int,
    repetition: int,
) -> Path:
    if repetition not in (1, 2, 3):
        raise ValueError("calibration repetition must be one, two, or three")
    return (
        config.output_root.resolve()
        / "calibration"
        / calibration_key(scenario, gpu_count)
        / f"rep{repetition}"
    )


def capacity_path(
    config: ExperimentConfig,
    scenario: Scenario,
    gpu_count: int,
) -> Path:
    return (
        config.output_root.resolve()
        / "calibration"
        / calibration_key(scenario, gpu_count)
        / "capacity.json"
    )


def finalize_capacity(
    config: ExperimentConfig,
    scenario: Scenario,
    gpu_count: int,
) -> CapacitySummary:
    references: list[CalibrationRunReference] = []
    for repetition in (1, 2, 3):
        run_dir = calibration_run_dir(config, scenario, gpu_count, repetition)
        validate_completion(run_dir)
        trace_path = run_dir / "workflow_trace.jsonl"
        summary = summarize_trace(trace_path)
        timestamps = summary["completion_timestamps"]
        if not isinstance(timestamps, list):
            raise TypeError("trace summary has invalid completion timestamps")
        completion_timestamps: list[float] = []
        for value in timestamps:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError("trace summary has invalid completion timestamps")
            completion_timestamps.append(float(value))
        slope = steady_state_completion_slope(
            completion_timestamps,
            start_fraction=config.steady_state_start_fraction,
            end_fraction=config.steady_state_end_fraction,
        )
        references.append(
            CalibrationRunReference(
                repetition=repetition,
                trace_sha256=file_sha256(trace_path),
                completion_slope_sessions_per_sec=slope,
            )
        )
    capacity = CapacitySummary(
        calibration_key=calibration_key(scenario, gpu_count),
        scenario=scenario,
        gpu_count=cast(Literal[1, 2, 3], gpu_count),
        steady_state_start_fraction=config.steady_state_start_fraction,
        steady_state_end_fraction=config.steady_state_end_fraction,
        capacity_sessions_per_sec=statistics.median(
            run.completion_slope_sessions_per_sec for run in references
        ),
        runs=(references[0], references[1], references[2]),
    )
    _write_or_match(capacity_path(config, scenario, gpu_count), capacity)
    return capacity


def load_capacity(
    config: ExperimentConfig,
    scenario: Scenario,
    gpu_count: int,
) -> CapacitySummary:
    saved = CapacitySummary.model_validate(
        read_json(capacity_path(config, scenario, gpu_count))
    )
    if saved.calibration_key != calibration_key(scenario, gpu_count):
        raise ValueError("capacity calibration identity does not match")
    for run in saved.runs:
        run_dir = calibration_run_dir(config, scenario, gpu_count, run.repetition)
        validate_completion(run_dir)
        trace_path = run_dir / "workflow_trace.jsonl"
        if file_sha256(trace_path) != run.trace_sha256:
            raise ValueError("capacity calibration trace hash changed")
    return saved


def _write_or_match(path: Path, value: BaseModel) -> None:
    payload = f"{canonical_json(value)}\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if not path.is_file() or path.read_text(encoding="utf-8") != payload:
            raise ValueError(f"capacity artifact content mismatch: {path}")
        return
    with path.open("x", encoding="utf-8") as file:
        file.write(payload)
        file.flush()
