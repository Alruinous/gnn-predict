from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import socket
import subprocess
import sys
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    NonNegativeFloat,
    PositiveInt,
)

from experiment.workflow.config import ExperimentConfig, TrialSpec

TRIAL_MANIFEST = "trial_manifest.json"
COMPLETION_MANIFEST = "completed.json"


class IncompleteTrialError(RuntimeError):
    pass


class ArtifactModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class EnvironmentManifest(ArtifactModel):
    hostname: str
    python_version: str
    platform: str
    git_commit: str
    git_dirty: bool
    git_diff_sha256: str
    package_versions: dict[str, str]


class TrialManifest(ArtifactModel):
    version: PositiveInt = 1
    experiment_id: str
    trial: TrialSpec
    created_at: NonNegativeFloat
    absolute_arrival_rate: NonNegativeFloat | None = None
    sample_manifest_sha256: str
    prediction_cache_sha256: str | None = None
    arrival_trace_sha256: str
    arrival_manifest_sha256: str | None = None
    capacity_calibration_sha256: str | None = None
    workflow_config: dict[str, JsonValue]
    scheduler_config: dict[str, JsonValue] | None = None
    serving_environment: dict[str, JsonValue]
    telemetry_config: dict[str, JsonValue] = Field(default_factory=dict)
    environment: EnvironmentManifest


class CompletionManifest(ArtifactModel):
    version: PositiveInt = 1
    trial_id: str
    completed_at: NonNegativeFloat
    artifact_sha256: dict[str, str] = Field(min_length=1)


def canonical_json(value: object) -> str:
    if isinstance(value, BaseModel):
        value = value.model_dump(mode="json")
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def stable_digest(value: object) -> str:
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json_exclusive(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as file:
        file.write(canonical_json(value))
        file.write("\n")
        file.flush()


def write_jsonl_exclusive(path: Path, rows: Iterable[object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as file:
        for row in rows:
            file.write(canonical_json(row))
            file.write("\n")
        file.flush()


def read_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def prepare_experiment_root(config: ExperimentConfig) -> None:
    root = config.output_root
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "experiment_manifest.json"
    payload = {
        "version": 1,
        "experiment_id": config.experiment_id,
        "config": config.model_dump(mode="json"),
    }
    if manifest_path.exists():
        if read_json(manifest_path) != payload:
            raise ValueError("experiment manifest does not match the requested config")
        return
    write_json_exclusive(manifest_path, payload)


def prepare_trial_directory(
    output_root: Path,
    manifest: TrialManifest,
) -> tuple[Literal["run", "skip"], Path]:
    trial_dir = output_root / "trials" / manifest.trial.trial_id
    manifest_path = trial_dir / TRIAL_MANIFEST
    completion_path = trial_dir / COMPLETION_MANIFEST
    if not trial_dir.exists():
        trial_dir.mkdir(parents=True)
        write_json_exclusive(manifest_path, manifest)
        return "run", trial_dir
    if not trial_dir.is_dir() or not manifest_path.is_file():
        raise IncompleteTrialError(f"trial directory is incomplete: {trial_dir}")
    saved = TrialManifest.model_validate(read_json(manifest_path))
    if saved != manifest:
        raise ValueError(f"trial manifest mismatch: {trial_dir}")
    if not completion_path.is_file():
        raise IncompleteTrialError(f"trial has no completion marker: {trial_dir}")
    validate_completion(trial_dir)
    return "skip", trial_dir


def complete_trial(
    trial_dir: Path,
    trial_id: str,
    artifact_names: Iterable[str],
    *,
    completed_at: float,
) -> CompletionManifest:
    names = tuple(sorted(set(artifact_names)))
    if not names:
        raise ValueError("a completed trial must contain artifacts")
    hashes: dict[str, str] = {}
    for name in names:
        path = trial_dir / name
        if not path.is_file():
            raise FileNotFoundError(path)
        hashes[name] = file_sha256(path)
    completion = CompletionManifest(
        trial_id=trial_id,
        completed_at=completed_at,
        artifact_sha256=hashes,
    )
    write_json_exclusive(trial_dir / COMPLETION_MANIFEST, completion)
    return completion


def validate_completion(trial_dir: Path) -> CompletionManifest:
    completion = CompletionManifest.model_validate(
        read_json(trial_dir / COMPLETION_MANIFEST)
    )
    if completion.trial_id != trial_dir.name:
        raise ValueError("completion marker has the wrong trial id")
    for name, expected in completion.artifact_sha256.items():
        path = trial_dir / name
        if not path.is_file() or file_sha256(path) != expected:
            raise ValueError(f"completed trial artifact is invalid: {path}")
    return completion


def environment_manifest() -> EnvironmentManifest:
    packages = (
        "langchain",
        "langchain-core",
        "langgraph",
        "nvidia-ml-py",
        "pydantic",
        "ray",
        "rouge-score",
        "torch",
        "transformers",
    )
    status = _git_output("status", "--porcelain=v1")
    return EnvironmentManifest(
        hostname=socket.gethostname(),
        python_version=sys.version,
        platform=platform.platform(),
        git_commit=_git_output("rev-parse", "HEAD"),
        git_dirty=bool(status),
        git_diff_sha256=_repository_diff_sha256(),
        package_versions={name: importlib.metadata.version(name) for name in packages},
    )


def json_mapping(value: Mapping[str, object]) -> dict[str, JsonValue]:
    normalized = json.loads(canonical_json(value))
    if not isinstance(normalized, dict):
        raise TypeError("JSON mapping normalization returned a non-object")
    return normalized


def _git_output(*args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _repository_diff_sha256() -> str:
    digest = hashlib.sha256()
    tracked_diff = subprocess.run(
        ["git", "diff", "HEAD", "--binary"],
        check=True,
        capture_output=True,
    ).stdout
    digest.update(tracked_diff)
    untracked = subprocess.run(
        ["git", "ls-files", "--others", "--exclude-standard", "-z"],
        check=True,
        capture_output=True,
    ).stdout.split(b"\0")
    for raw_path in sorted(path for path in untracked if path):
        digest.update(b"\0untracked\0")
        digest.update(raw_path)
        path = Path(os.fsdecode(raw_path))
        if path.is_symlink():
            digest.update(b"\0symlink\0")
            digest.update(os.readlink(path).encode())
        elif path.is_file():
            digest.update(b"\0content\0")
            with path.open("rb") as file:
                for chunk in iter(lambda: file.read(1024 * 1024), b""):
                    digest.update(chunk)
        else:
            digest.update(b"\0directory\0")
    return digest.hexdigest()
