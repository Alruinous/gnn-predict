from __future__ import annotations

import hashlib
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

import ray

from experiment.workflow.analysis import write_trial_analysis
from experiment.workflow.artifacts import (
    COMPLETION_MANIFEST,
    TRIAL_MANIFEST,
    IncompleteTrialError,
    TrialManifest,
    canonical_json,
    complete_trial,
    file_sha256,
    json_mapping,
    prepare_trial_directory,
    read_json,
    validate_completion,
    write_json_exclusive,
    write_jsonl_exclusive,
)
from experiment.workflow.calibration import CapacitySummary, capacity_path
from experiment.workflow.config import ExperimentConfig, TrialSpec
from experiment.workflow.environment import (
    inspect_serving_environment,
    selected_gpu_ids,
    validate_runtime_paths,
    validate_visible_devices,
)
from experiment.workflow.execution import (
    ScenarioBundle,
    SessionInvocation,
    build_scheduler_config,
    build_session_invocations,
    initialize_local_ray,
    load_scenario_bundle,
    normalized_output_rows,
    run_langgraph_baseline,
    run_workflow_runtime,
)
from experiment.workflow.prepare import (
    ArrivalTraceArtifact,
    PreparedExperiment,
    load_arrival_trace,
    load_prepared_experiment,
)
from experiment.workflow.quality import (
    summarize_mbpp_quality,
    summarize_qmsum_quality,
    write_quality_report,
)
from experiment.workflow.telemetry import (
    NvmlSampler,
    PeriodicRecorder,
    TelemetryConfig,
)
from workflow.artifacts import SchedulerConfig


@dataclass(frozen=True, slots=True)
class TrialArrival:
    payload: object
    sha256: str
    absolute_sessions_per_sec: float | None
    sample_ids: tuple[str, ...]
    offsets_sec: tuple[float, ...]
    source_manifest_sha256: str | None = None
    capacity_sha256: str | None = None


@dataclass(frozen=True, slots=True)
class TrialRunOutcome:
    trial_id: str
    action: Literal["completed", "skipped"]
    trial_dir: Path


def _existing_trial(
    prepared: PreparedExperiment,
    trial: TrialSpec,
) -> TrialRunOutcome | None:
    trial_dir = prepared.root / "trials" / trial.trial_id
    if not trial_dir.exists():
        return None
    manifest_path = trial_dir / TRIAL_MANIFEST
    completion_path = trial_dir / COMPLETION_MANIFEST
    if not trial_dir.is_dir() or not manifest_path.is_file():
        raise IncompleteTrialError(f"trial directory is incomplete: {trial_dir}")
    manifest = TrialManifest.model_validate(read_json(manifest_path))
    if manifest.experiment_id != prepared.manifest.experiment_id:
        raise ValueError(f"trial experiment id mismatch: {trial_dir}")
    if manifest.trial != trial:
        raise ValueError(f"trial manifest identity mismatch: {trial_dir}")
    if not completion_path.is_file():
        raise IncompleteTrialError(f"trial has no completion marker: {trial_dir}")
    validate_completion(trial_dir)
    return TrialRunOutcome(trial.trial_id, "skipped", trial_dir)


def run_trial(config: ExperimentConfig, trial: TrialSpec) -> TrialRunOutcome:
    prepared = load_prepared_experiment(config)
    if trial not in prepared.trial_matrix:
        raise ValueError(f"trial is not in the formal matrix: {trial.trial_id}")
    existing = _existing_trial(prepared, trial)
    if existing is not None:
        return existing
    validate_runtime_paths(config)
    arrival = _trial_arrival(config, prepared, trial)
    bundle = load_scenario_bundle(
        config,
        trial,
        prepared.sample_manifest_path(trial.scenario),
        expected_session_count=config.session_count,
    )
    sessions = build_session_invocations(
        bundle,
        arrival.sample_ids,
        arrival.offsets_sec,
    )
    gpu_ids = selected_gpu_ids(trial.gpu_count)
    validate_visible_devices(gpu_ids)
    serving_environment = inspect_serving_environment(
        config.vllm_python,
        gpu_ids,
    ).assert_idle(config.gpu_idle_memory_limit_mb)
    telemetry_config = _telemetry_config(config, gpu_ids)
    hostname = initialize_local_ray(trial.gpu_count)
    try:
        scheduler_config = (
            None
            if trial.strategy == "lg-batch"
            else build_scheduler_config(config, trial, hostname)
        )
        manifest = _trial_manifest(
            config,
            prepared,
            trial,
            arrival,
            bundle.workflow.model_dump(mode="json"),
            scheduler_config,
            serving_environment.model_dump(mode="json"),
            telemetry_config,
        )
        action, trial_dir = prepare_trial_directory(prepared.root, manifest)
        if action == "skip":
            return TrialRunOutcome(trial.trial_id, "skipped", trial_dir)
        _write_trial_inputs(
            prepared,
            trial,
            trial_dir,
            arrival,
            bundle.workflow.model_dump(mode="json"),
            scheduler_config,
            serving_environment.model_dump(mode="json"),
            telemetry_config,
        )
        gpu_recorder = PeriodicRecorder(
            trial_dir / "gpu_telemetry.jsonl",
            NvmlSampler(gpu_ids),
            config.telemetry_interval_sec,
        )
        gpu_recorder.start()
        try:
            outputs = _execute_trial(
                config,
                prepared,
                trial,
                trial_dir,
                bundle,
                sessions,
                scheduler_config,
            )
        finally:
            gpu_recorder.stop()
        write_jsonl_exclusive(
            trial_dir / "normalized_session_outputs.jsonl",
            normalized_output_rows(sessions, outputs),
        )
        _write_quality(bundle, sessions, outputs, trial_dir)
        write_trial_analysis(trial_dir)
        artifact_names = _trial_artifact_names(trial)
        complete_trial(
            trial_dir,
            trial.trial_id,
            artifact_names,
            completed_at=time.time(),
        )
        return TrialRunOutcome(trial.trial_id, "completed", trial_dir)
    finally:
        ray.shutdown()


def _execute_trial(
    config: ExperimentConfig,
    prepared: PreparedExperiment,
    trial: TrialSpec,
    trial_dir: Path,
    bundle: ScenarioBundle,
    sessions: tuple[SessionInvocation, ...],
    scheduler_config: SchedulerConfig | None,
) -> dict[str, str]:
    run_id = f"{config.experiment_id}/{trial.trial_id}"
    if trial.strategy == "lg-batch":
        return run_langgraph_baseline(
            bundle,
            sessions,
            vllm_python=config.vllm_python,
            trial_dir=trial_dir,
            run_id=run_id,
            timeout_sec=config.trial_timeout_sec,
        )
    if scheduler_config is None:
        raise RuntimeError("workflow strategy requires a scheduler configuration")
    return run_workflow_runtime(
        bundle,
        sessions,
        scheduler_config=scheduler_config,
        prediction_path=prepared.synthetic_cache_path,
        trial_dir=trial_dir,
        run_id=run_id,
        timeout_sec=config.trial_timeout_sec,
        telemetry_interval_sec=config.telemetry_interval_sec,
    )


def _trial_arrival(
    config: ExperimentConfig,
    prepared: PreparedExperiment,
    trial: TrialSpec,
) -> TrialArrival:
    if trial.workload == "burst":
        permutation = prepared.burst_permutation(
            trial.scenario,
            trial.repetition,
        )
        payload = {
            "version": 1,
            "trace_key": (
                f"{trial.scenario}__g{trial.gpu_count}__burst__rep{trial.repetition}"
            ),
            "scenario": trial.scenario,
            "gpu_count": trial.gpu_count,
            "workload": "burst",
            "repetition": trial.repetition,
            "absolute_sessions_per_sec": None,
            "arrivals": [
                {
                    "position": position,
                    "sample_id": sample_id,
                    "offset_sec": 0.0,
                }
                for position, sample_id in enumerate(permutation.sample_ids)
            ],
        }
        return TrialArrival(
            payload=payload,
            sha256=_json_file_sha256(payload),
            absolute_sessions_per_sec=None,
            sample_ids=permutation.sample_ids,
            offsets_sec=(0.0,) * len(permutation.sample_ids),
        )
    capacity_file = capacity_path(config, trial.scenario, trial.gpu_count)
    capacity = CapacitySummary.model_validate(read_json(capacity_file))
    artifact = load_arrival_trace(config, trial)
    expected_rate = (
        capacity.capacity_sessions_per_sec * cast(int, trial.load_percent) / 100
    )
    if artifact.trace.absolute_sessions_per_sec != expected_rate:
        raise ValueError("arrival trace was generated from another capacity result")
    return _open_loop_arrival(
        artifact,
        file_sha256(capacity_file),
    )


def _open_loop_arrival(
    artifact: ArrivalTraceArtifact,
    capacity_sha256: str,
) -> TrialArrival:
    return TrialArrival(
        payload=artifact.trace,
        sha256=artifact.sha256,
        absolute_sessions_per_sec=artifact.trace.absolute_sessions_per_sec,
        sample_ids=tuple(point.sample_id for point in artifact.trace.arrivals),
        offsets_sec=tuple(point.offset_sec for point in artifact.trace.arrivals),
        source_manifest_sha256=artifact.manifest_sha256,
        capacity_sha256=capacity_sha256,
    )


def _trial_manifest(
    config: ExperimentConfig,
    prepared: PreparedExperiment,
    trial: TrialSpec,
    arrival: TrialArrival,
    workflow_config: dict[str, object],
    scheduler_config: SchedulerConfig | None,
    serving_environment: dict[str, object],
    telemetry_config: TelemetryConfig,
) -> TrialManifest:
    return TrialManifest(
        experiment_id=config.experiment_id,
        trial=trial,
        created_at=time.time(),
        absolute_arrival_rate=arrival.absolute_sessions_per_sec,
        sample_manifest_sha256=prepared.sample_manifest_sha256[trial.scenario],
        prediction_cache_sha256=(
            None if trial.strategy == "lg-batch" else prepared.synthetic_cache_sha256
        ),
        arrival_trace_sha256=arrival.sha256,
        arrival_manifest_sha256=arrival.source_manifest_sha256,
        capacity_calibration_sha256=arrival.capacity_sha256,
        workflow_config=json_mapping(workflow_config),
        scheduler_config=(
            None
            if scheduler_config is None
            else json_mapping(scheduler_config.model_dump(mode="json"))
        ),
        serving_environment=json_mapping(serving_environment),
        telemetry_config=json_mapping(telemetry_config.model_dump(mode="json")),
        environment=prepared.environment,
    )


def _write_trial_inputs(
    prepared: PreparedExperiment,
    trial: TrialSpec,
    trial_dir: Path,
    arrival: TrialArrival,
    workflow_config: dict[str, object],
    scheduler_config: SchedulerConfig | None,
    serving_environment: dict[str, object],
    telemetry_config: TelemetryConfig,
) -> None:
    _copy_exclusive(
        prepared.sample_manifest_path(trial.scenario),
        trial_dir / "sample_manifest.jsonl",
    )
    write_json_exclusive(trial_dir / "arrival_trace.json", arrival.payload)
    if _json_file_sha256(arrival.payload) != arrival.sha256:
        raise ValueError("trial arrival trace hash changed while writing")
    write_json_exclusive(trial_dir / "workflow_config.json", workflow_config)
    write_json_exclusive(
        trial_dir / "serving_environment.json",
        serving_environment,
    )
    write_json_exclusive(
        trial_dir / "telemetry_config.json",
        telemetry_config,
    )
    if scheduler_config is not None:
        write_json_exclusive(
            trial_dir / "scheduler_config.json",
            scheduler_config,
        )


def _write_quality(
    bundle: ScenarioBundle,
    sessions: tuple[SessionInvocation, ...],
    outputs: dict[str, str],
    trial_dir: Path,
) -> None:
    sample_sessions = {session.sample_id: session.session_id for session in sessions}
    if bundle.scenario == "qmsum":
        report = summarize_qmsum_quality(
            bundle.samples,
            sample_sessions,
            outputs,
        )
    else:
        report = summarize_mbpp_quality(
            bundle.samples,
            sample_sessions,
            outputs,
        )
    write_quality_report(
        report,
        rows_path=trial_dir / "quality_rows.jsonl",
        summary_path=trial_dir / "quality_summary.json",
    )


def _trial_artifact_names(trial: TrialSpec) -> tuple[str, ...]:
    common = [
        TRIAL_MANIFEST,
        "arrival_trace.json",
        "experiment_summary.json",
        "gpu_telemetry.jsonl",
        "normalized_session_outputs.jsonl",
        "quality_rows.jsonl",
        "quality_summary.json",
        "run_summary.json",
        "sample_manifest.jsonl",
        "serving_environment.json",
        "session_results.jsonl",
        "telemetry_config.json",
        "workflow_config.json",
        "workflow_trace.jsonl",
    ]
    if trial.strategy != "lg-batch":
        common.extend(("queue_telemetry.jsonl", "scheduler_config.json"))
    return tuple(common)


def _copy_exclusive(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("xb") as output, source.open("rb") as input_file:
        shutil.copyfileobj(input_file, output)
        output.flush()


def _telemetry_config(
    config: ExperimentConfig,
    gpu_ids: tuple[int, ...],
) -> TelemetryConfig:
    return TelemetryConfig(
        interval_sec=config.telemetry_interval_sec,
        physical_gpu_ids=gpu_ids,
        energy_baseline_method=config.telemetry_baseline_method,
        memory_baseline_method=config.telemetry_baseline_method,
    )


def _json_file_sha256(value: object) -> str:
    payload = f"{canonical_json(value)}\n".encode()
    return hashlib.sha256(payload).hexdigest()
