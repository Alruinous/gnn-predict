from __future__ import annotations

import time
from collections.abc import Mapping
from pathlib import Path
from typing import Literal, cast

import ray
from pydantic import JsonValue

from experiment.workflow.analysis import write_trial_analysis
from experiment.workflow.artifacts import (
    COMPLETION_MANIFEST,
    IncompleteTrialError,
    complete_trial,
    environment_manifest,
    file_sha256,
    json_mapping,
    read_json,
    validate_completion,
    write_json_exclusive,
    write_jsonl_exclusive,
)
from experiment.workflow.calibration import (
    CalibrationRepetition,
    CalibrationRunManifest,
    calibration_key,
    calibration_run_dir,
)
from experiment.workflow.config import ExperimentConfig, Scenario, TrialSpec
from experiment.workflow.environment import (
    inspect_serving_environment,
    selected_gpu_ids,
    validate_runtime_paths,
    validate_visible_devices,
)
from experiment.workflow.execution import (
    SessionInvocation,
    build_scheduler_config,
    initialize_local_ray,
    load_scenario_bundle,
    run_workflow_calibration,
)
from experiment.workflow.mbpp import MBPP_MANIFEST_PATH
from experiment.workflow.plan import session_permutation
from experiment.workflow.prepare import load_prepared_experiment
from experiment.workflow.qmsum import QMSUM_MANIFEST_PATH
from experiment.workflow.telemetry import (
    NvmlSampler,
    PeriodicRecorder,
    TelemetryConfig,
)

CALIBRATION_SESSION_COUNT = 180
CALIBRATION_CONCURRENCY = 60
PARENT_CALIBRATION_SAMPLE_COUNT = 60


def run_calibration(
    config: ExperimentConfig,
    scenario: Scenario,
    gpu_count: Literal[1, 2, 3],
    repetition: CalibrationRepetition,
) -> Literal["completed", "skipped"]:
    calibration_key(scenario, gpu_count)
    prepared = load_prepared_experiment(config)
    directory = calibration_run_dir(config, scenario, gpu_count, repetition)
    existing = _existing_calibration(
        directory,
        scenario,
        gpu_count,
        repetition,
    )
    if existing is not None:
        return existing
    validate_runtime_paths(config)
    trial = TrialSpec(
        scenario=scenario,
        strategy="wf-cache",
        gpu_count=gpu_count,
        workload="burst",
        repetition=cast(Literal[1, 2, 3, 4, 5], repetition),
    )
    sample_manifest_path = _calibration_sample_manifest_path(scenario)
    bundle = load_scenario_bundle(
        config,
        trial,
        sample_manifest_path,
        expected_session_count=PARENT_CALIBRATION_SAMPLE_COUNT,
    )
    sessions = _calibration_sessions(
        bundle.inputs_by_sample_id,
        config.seed,
        repetition,
    )
    gpu_ids = selected_gpu_ids(gpu_count)
    validate_visible_devices(gpu_ids)
    serving_environment = inspect_serving_environment(
        config.vllm_python,
        gpu_ids,
    ).assert_idle(config.gpu_idle_memory_limit_mb)
    telemetry_config = TelemetryConfig(
        interval_sec=config.telemetry_interval_sec,
        physical_gpu_ids=gpu_ids,
        energy_baseline_method=config.telemetry_baseline_method,
        memory_baseline_method=config.telemetry_baseline_method,
    )
    hostname = initialize_local_ray(gpu_count)
    try:
        scheduler_config = build_scheduler_config(config, trial, hostname)
        manifest = CalibrationRunManifest(
            calibration_key=calibration_key(scenario, gpu_count),
            scenario=scenario,
            gpu_count=gpu_count,
            repetition=repetition,
            seed=config.seed,
            sample_manifest_sha256=file_sha256(sample_manifest_path),
            prediction_cache_sha256=prepared.synthetic_cache_sha256,
            workflow_config=json_mapping(bundle.workflow.model_dump(mode="json")),
            scheduler_config=json_mapping(scheduler_config.model_dump(mode="json")),
            serving_environment=serving_environment,
            telemetry_config=telemetry_config,
            environment=environment_manifest(),
            created_at=time.time(),
        )
        action = _prepare_calibration_directory(directory, manifest)
        if action == "skip":
            return "skipped"
        write_jsonl_exclusive(
            directory / "calibration_sessions.jsonl",
            (session.arrival_row() for session in sessions),
        )
        write_json_exclusive(
            directory / "workflow_config.json",
            bundle.workflow,
        )
        write_json_exclusive(
            directory / "scheduler_config.json",
            scheduler_config,
        )
        write_json_exclusive(
            directory / "serving_environment.json",
            serving_environment,
        )
        write_json_exclusive(
            directory / "telemetry_config.json",
            telemetry_config,
        )
        gpu_recorder = PeriodicRecorder(
            directory / "gpu_telemetry.jsonl",
            NvmlSampler(gpu_ids),
            config.telemetry_interval_sec,
        )
        gpu_recorder.start()
        try:
            run_workflow_calibration(
                bundle,
                sessions,
                scheduler_config=scheduler_config,
                prediction_path=prepared.synthetic_cache_path,
                calibration_dir=directory,
                run_id=(
                    f"{config.experiment_id}/calibration/"
                    f"{calibration_key(scenario, gpu_count)}/rep{repetition}"
                ),
                concurrency=CALIBRATION_CONCURRENCY,
                timeout_sec=config.trial_timeout_sec,
                telemetry_interval_sec=config.telemetry_interval_sec,
            )
        finally:
            gpu_recorder.stop()
        write_trial_analysis(directory)
        complete_trial(
            directory,
            directory.name,
            (
                "calibration_manifest.json",
                "calibration_sessions.jsonl",
                "experiment_summary.json",
                "gpu_telemetry.jsonl",
                "queue_telemetry.jsonl",
                "run_summary.json",
                "scheduler_config.json",
                "serving_environment.json",
                "session_results.jsonl",
                "telemetry_config.json",
                "workflow_config.json",
                "workflow_trace.jsonl",
            ),
            completed_at=time.time(),
        )
        return "completed"
    finally:
        ray.shutdown()


def _calibration_sample_manifest_path(scenario: Scenario) -> Path:
    return QMSUM_MANIFEST_PATH if scenario == "qmsum" else MBPP_MANIFEST_PATH


def _calibration_sessions(
    inputs_by_sample_id: Mapping[str, dict[str, JsonValue]],
    seed: int,
    repetition: int,
) -> tuple[SessionInvocation, ...]:
    sample_ids = tuple(inputs_by_sample_id)
    sessions: list[SessionInvocation] = []
    for cycle in range(3):
        permutation = session_permutation(
            sample_ids,
            seed=seed,
            repetition=repetition * 10 + cycle,
        )
        for sample_id in permutation:
            position = len(sessions)
            sessions.append(
                SessionInvocation(
                    position=position,
                    sample_id=sample_id,
                    session_id=f"cal-{position:03d}-{sample_id}",
                    arrival_offset_sec=0.0,
                    inputs=dict(inputs_by_sample_id[sample_id]),
                )
            )
    if len(sessions) != CALIBRATION_SESSION_COUNT:
        raise ValueError("capacity calibration requires 180 sessions")
    return tuple(sessions)


def _prepare_calibration_directory(
    directory: Path,
    manifest: CalibrationRunManifest,
) -> Literal["run", "skip"]:
    manifest_path = directory / "calibration_manifest.json"
    if not directory.exists():
        directory.mkdir(parents=True)
        write_json_exclusive(manifest_path, manifest)
        return "run"
    if not directory.is_dir() or not manifest_path.is_file():
        raise IncompleteTrialError(f"calibration directory is incomplete: {directory}")
    saved = CalibrationRunManifest.model_validate(read_json(manifest_path))
    if saved != manifest:
        raise ValueError(f"calibration manifest mismatch: {directory}")
    if not (directory / COMPLETION_MANIFEST).is_file():
        raise IncompleteTrialError(f"calibration has no completion marker: {directory}")
    validate_completion(directory)
    return "skip"


def _existing_calibration(
    directory: Path,
    scenario: Scenario,
    gpu_count: int,
    repetition: int,
) -> Literal["skipped"] | None:
    if not directory.exists():
        return None
    manifest_path = directory / "calibration_manifest.json"
    completion_path = directory / COMPLETION_MANIFEST
    if not directory.is_dir() or not manifest_path.is_file():
        raise IncompleteTrialError(f"calibration directory is incomplete: {directory}")
    manifest = CalibrationRunManifest.model_validate(read_json(manifest_path))
    expected = (calibration_key(scenario, gpu_count), scenario, gpu_count, repetition)
    actual = (
        manifest.calibration_key,
        manifest.scenario,
        manifest.gpu_count,
        manifest.repetition,
    )
    if actual != expected:
        raise ValueError(f"calibration manifest identity mismatch: {directory}")
    if not completion_path.is_file():
        raise IncompleteTrialError(f"calibration has no completion marker: {directory}")
    validate_completion(directory)
    return "skipped"
