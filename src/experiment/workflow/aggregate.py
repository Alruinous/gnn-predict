from __future__ import annotations

import os
import shutil
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast
from uuid import uuid4

from experiment.workflow.analysis import (
    TELEMETRY_BASELINE_METHOD,
    confidence_interval_95,
    summarize_trial,
)
from experiment.workflow.artifacts import (
    COMPLETION_MANIFEST,
    TRIAL_MANIFEST,
    TrialManifest,
    canonical_json,
    file_sha256,
    read_json,
    stable_digest,
    validate_completion,
)

TRIAL_METRICS = "trial_metrics.jsonl"
GROUP_METRICS = "group_metrics.jsonl"


@dataclass(frozen=True, slots=True)
class AnalysisArtifacts:
    trial_metrics_path: Path
    group_metrics_path: Path
    completed_trial_count: int
    aligned_group_count: int


def scan_completed_trials(output_root: Path) -> tuple[Path, ...]:
    trials_dir = output_root / "trials"
    if not trials_dir.is_dir():
        raise FileNotFoundError(trials_dir)
    completed = []
    for trial_dir in sorted(trials_dir.iterdir()):
        if not trial_dir.is_dir() or not (trial_dir / COMPLETION_MANIFEST).is_file():
            continue
        marker = validate_completion(trial_dir)
        required = {TRIAL_MANIFEST, "workflow_trace.jsonl"}
        if not required <= marker.artifact_sha256.keys():
            raise ValueError(f"completion marker omits analysis artifacts: {trial_dir}")
        completed.append(trial_dir)
    return tuple(completed)


def build_trial_metrics(
    trial_dir: Path,
    *,
    steady_state_start_fraction: float = 0.2,
    steady_state_end_fraction: float = 0.8,
) -> dict[str, object]:
    completion = validate_completion(trial_dir)
    manifest = TrialManifest.model_validate(read_json(trial_dir / TRIAL_MANIFEST))
    trial = manifest.trial
    required = {
        TRIAL_MANIFEST,
        "workflow_trace.jsonl",
        "gpu_telemetry.jsonl",
        "telemetry_config.json",
    }
    if trial.strategy != "lg-batch":
        required.add("queue_telemetry.jsonl")
    if not required <= completion.artifact_sha256.keys():
        raise ValueError(f"completion marker omits analysis artifacts: {trial_dir}")
    if trial.trial_id != trial_dir.name or completion.trial_id != trial.trial_id:
        raise ValueError(f"trial identity mismatch: {trial_dir}")
    telemetry_config = _mapping(read_json(trial_dir / "telemetry_config.json"))
    if dict(telemetry_config) != manifest.telemetry_config:
        raise ValueError(f"telemetry config does not match manifest: {trial_dir}")
    for name in ("energy_baseline_method", "memory_baseline_method"):
        if telemetry_config.get(name) != TELEMETRY_BASELINE_METHOD:
            raise ValueError(f"unsupported telemetry baseline method: {trial_dir}")
    metrics = summarize_trial(
        trial_dir,
        steady_state_start_fraction=steady_state_start_fraction,
        steady_state_end_fraction=steady_state_end_fraction,
    )
    pairing = _mapping(metrics["trace_pairing"])
    if any(_integer(value) != 0 for value in pairing.values()):
        raise ValueError(f"completed trial trace has unmatched events: {trial_dir}")
    sessions_per_min = _numeric(metrics["sessions_per_min"])
    metrics["sessions_per_min_per_gpu"] = sessions_per_min / trial.gpu_count
    if manifest.absolute_arrival_rate is not None:
        metrics["offered_arrival_rate_sessions_per_sec"] = (
            manifest.absolute_arrival_rate
        )
    alignment = _alignment_metadata(manifest)
    return {
        "trial_id": trial.trial_id,
        "experiment_id": manifest.experiment_id,
        **trial.model_dump(mode="json"),
        "completed_at": completion.completed_at,
        "alignment_fingerprint": stable_digest(alignment),
        "alignment": alignment,
        "arrival_trace_sha256": manifest.arrival_trace_sha256,
        "metrics": metrics,
    }


def aggregate_trial_metrics(
    trial_rows: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    grouped: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    dimensions_by_key: dict[str, dict[str, object]] = {}
    for row in trial_rows:
        dimensions = _group_dimensions(row)
        key = canonical_json(dimensions)
        grouped[key].append(row)
        dimensions_by_key[key] = dimensions

    results = []
    for key in sorted(grouped):
        rows = grouped[key]
        repetitions = [_integer(row.get("repetition")) for row in rows]
        if len(repetitions) != len(set(repetitions)):
            raise ValueError("trial group contains duplicate repetitions")
        if set(repetitions) != {1, 2, 3, 4, 5}:
            continue
        metrics = [_mapping(row.get("metrics")) for row in rows]
        aggregated = _aggregate_numeric_tree(metrics)
        dimensions = dimensions_by_key[key]
        alignment = dict(_mapping(rows[0].get("alignment")))
        results.append(
            {
                "group_id": stable_digest(dimensions),
                **dimensions,
                "trial_count": 5,
                "trial_ids": sorted(_string(row.get("trial_id")) for row in rows),
                "alignment": alignment,
                "metrics": aggregated,
            }
        )
    return results


def regenerate_analysis(
    output_root: Path,
    *,
    analysis_dir: Path | None = None,
) -> AnalysisArtifacts:
    output_root = output_root.resolve()
    target_dir = (
        (output_root / "analysis") if analysis_dir is None else analysis_dir.resolve()
    )
    _validate_analysis_directory(output_root, target_dir)
    start_fraction, end_fraction = _steady_state_fractions(output_root)
    trial_rows = [
        build_trial_metrics(
            trial_dir,
            steady_state_start_fraction=start_fraction,
            steady_state_end_fraction=end_fraction,
        )
        for trial_dir in scan_completed_trials(output_root)
    ]
    group_rows = aggregate_trial_metrics(trial_rows)
    trial_path, group_path = _publish_analysis(
        target_dir,
        trial_rows,
        group_rows,
    )
    return AnalysisArtifacts(
        trial_metrics_path=trial_path,
        group_metrics_path=group_path,
        completed_trial_count=len(trial_rows),
        aligned_group_count=len(group_rows),
    )


def _group_dimensions(row: Mapping[str, object]) -> dict[str, object]:
    names = (
        "experiment_id",
        "scenario",
        "strategy",
        "gpu_count",
        "workload",
        "max_num_seqs",
        "queue_capacity",
        "load_percent",
        "alignment_fingerprint",
    )
    missing = [name for name in names if name not in row]
    if missing:
        raise ValueError(f"trial row lacks group dimensions: {missing}")
    return {name: row[name] for name in names}


def _alignment_metadata(manifest: TrialManifest) -> dict[str, object]:
    return {
        "experiment_id": manifest.experiment_id,
        "sample_manifest_sha256": manifest.sample_manifest_sha256,
        "prediction_cache_sha256": manifest.prediction_cache_sha256,
        "capacity_calibration_sha256": manifest.capacity_calibration_sha256,
        "absolute_arrival_rate": manifest.absolute_arrival_rate,
        "workflow_config_sha256": stable_digest(manifest.workflow_config),
        "scheduler_config_sha256": stable_digest(manifest.scheduler_config),
        "serving_environment_sha256": stable_digest(
            _stable_serving_environment(manifest.serving_environment)
        ),
        "telemetry_config_sha256": stable_digest(manifest.telemetry_config),
        "environment_sha256": stable_digest(manifest.environment),
    }


def _stable_serving_environment(
    environment: Mapping[str, object],
) -> dict[str, object]:
    normalized = dict(environment)
    gpus = normalized.get("gpus")
    if not isinstance(gpus, list):
        return normalized
    stable_gpus = []
    for value in gpus:
        gpu = dict(_mapping(value))
        gpu.pop("used_memory_mb", None)
        gpu.pop("compute_process_pids", None)
        gpu.pop("graphics_process_pids", None)
        stable_gpus.append(gpu)
    normalized["gpus"] = stable_gpus
    return normalized


def _aggregate_numeric_tree(
    values: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    if len(values) != 5:
        raise ValueError("formal aggregation requires five trial metric rows")
    common_keys = set(values[0])
    for value in values[1:]:
        common_keys &= value.keys()
    result: dict[str, object] = {}
    for key in sorted(common_keys):
        children = [value[key] for value in values]
        if all(_is_numeric(child) for child in children):
            result[key] = confidence_interval_95(
                [_numeric(child) for child in children]
            )
            continue
        if all(isinstance(child, Mapping) for child in children):
            nested = _aggregate_numeric_tree([_mapping(child) for child in children])
            if nested:
                result[key] = nested
    return result


def _steady_state_fractions(output_root: Path) -> tuple[float, float]:
    manifest_path = output_root / "experiment_manifest.json"
    if not manifest_path.is_file():
        return 0.2, 0.8
    manifest = _mapping(read_json(manifest_path))
    config = _mapping(manifest.get("config"))
    start = _numeric(config.get("steady_state_start_fraction"))
    end = _numeric(config.get("steady_state_end_fraction"))
    if not 0 <= start < end <= 1:
        raise ValueError("experiment manifest has invalid steady-state fractions")
    return start, end


def _validate_analysis_directory(output_root: Path, analysis_dir: Path) -> None:
    try:
        analysis_dir.relative_to(output_root)
    except ValueError:
        raise ValueError(
            "analysis directory must be inside the experiment root"
        ) from None
    try:
        analysis_dir.relative_to(output_root / "trials")
    except ValueError:
        return
    raise ValueError("analysis directory cannot be inside immutable trial data")


def _publish_analysis(
    analysis_dir: Path,
    trial_rows: Sequence[object],
    group_rows: Sequence[object],
) -> tuple[Path, Path]:
    generation_id = stable_digest(
        {"trial_metrics": trial_rows, "group_metrics": group_rows}
    )
    generations_dir = analysis_dir / "generations"
    generation_dir = generations_dir / generation_id
    created = False
    generations_dir.mkdir(parents=True, exist_ok=True)
    try:
        if generation_dir.exists():
            _validate_generation(generation_dir, generation_id)
        else:
            generation_dir.mkdir()
            created = True
            trial_generation_path = generation_dir / TRIAL_METRICS
            group_generation_path = generation_dir / GROUP_METRICS
            _write_jsonl_unpublished(trial_generation_path, trial_rows)
            _write_jsonl_unpublished(group_generation_path, group_rows)
            _write_generation_manifest(
                generation_dir,
                generation_id,
                trial_generation_path,
                group_generation_path,
                len(trial_rows),
                len(group_rows),
            )
        _ensure_stable_link(analysis_dir / TRIAL_METRICS, TRIAL_METRICS)
        _ensure_stable_link(analysis_dir / GROUP_METRICS, GROUP_METRICS)
        _switch_current_generation(analysis_dir, generation_id)
    except Exception:
        if created:
            shutil.rmtree(generation_dir, ignore_errors=True)
        raise
    return analysis_dir / TRIAL_METRICS, analysis_dir / GROUP_METRICS


def _write_jsonl_unpublished(path: Path, rows: Sequence[object]) -> None:
    with path.open("x", encoding="utf-8") as file:
        for row in rows:
            file.write(canonical_json(row))
            file.write("\n")
        file.flush()
        os.fsync(file.fileno())


def _write_generation_manifest(
    generation_dir: Path,
    generation_id: str,
    trial_path: Path,
    group_path: Path,
    trial_count: int,
    group_count: int,
) -> None:
    value = {
        "version": 1,
        "generation_id": generation_id,
        "trial_count": trial_count,
        "group_count": group_count,
        "artifact_sha256": {
            TRIAL_METRICS: file_sha256(trial_path),
            GROUP_METRICS: file_sha256(group_path),
        },
    }
    path = generation_dir / "generation_manifest.json"
    with path.open("x", encoding="utf-8") as file:
        file.write(canonical_json(value))
        file.write("\n")
        file.flush()
        os.fsync(file.fileno())


def _validate_generation(generation_dir: Path, generation_id: str) -> None:
    manifest = _mapping(read_json(generation_dir / "generation_manifest.json"))
    if manifest.get("generation_id") != generation_id:
        raise ValueError(f"analysis generation identity mismatch: {generation_dir}")
    artifacts = _mapping(manifest.get("artifact_sha256"))
    for name in (TRIAL_METRICS, GROUP_METRICS):
        expected = _string(artifacts.get(name))
        if file_sha256(generation_dir / name) != expected:
            raise ValueError(f"analysis generation artifact is invalid: {name}")


def _ensure_stable_link(path: Path, artifact_name: str) -> None:
    expected = Path("current") / artifact_name
    if path.is_symlink():
        if Path(os.readlink(path)) != expected:
            raise ValueError(f"analysis artifact link has the wrong target: {path}")
        return
    if path.exists():
        raise ValueError(f"analysis artifact path is not a symlink: {path}")
    os.symlink(expected, path)


def _switch_current_generation(analysis_dir: Path, generation_id: str) -> None:
    current = analysis_dir / "current"
    if current.exists() and not current.is_symlink():
        raise ValueError("analysis current path is not a symlink")
    temporary = analysis_dir / f".current.{uuid4().hex}.tmp"
    try:
        os.symlink(Path("generations") / generation_id, temporary)
        os.replace(temporary, current)
    finally:
        temporary.unlink(missing_ok=True)


def _mapping(value: object) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError("expected a mapping")
    return cast(Mapping[str, object], value)


def _numeric(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("expected a number")
    return float(value)


def _integer(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("expected an integer")
    return value


def _string(value: object) -> str:
    if not isinstance(value, str):
        raise TypeError("expected a string")
    return value


def _is_numeric(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)
