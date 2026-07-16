from __future__ import annotations

from pathlib import Path

import pytest

from experiment.workflow.artifacts import (
    IncompleteTrialError,
    TrialManifest,
    complete_trial,
    file_sha256,
    prepare_trial_directory,
    stable_digest,
    write_json_exclusive,
)
from experiment.workflow.config import TrialSpec


def trial_manifest(created_at: float = 1.0) -> TrialManifest:
    from experiment.workflow.artifacts import EnvironmentManifest

    return TrialManifest(
        experiment_id="system_20260713",
        trial=TrialSpec(
            scenario="qmsum",
            strategy="wf-cache",
            gpu_count=2,
            workload="burst",
            repetition=1,
        ),
        created_at=created_at,
        sample_manifest_sha256="sample-sha",
        prediction_cache_sha256="cache-sha",
        arrival_trace_sha256="arrival-sha",
        workflow_config={"nodes": []},
        scheduler_config={"policy": "cache"},
        serving_environment={"vllm": "0.10.2"},
        environment=EnvironmentManifest(
            hostname="host",
            python_version="3.12",
            platform="linux",
            git_commit="commit",
            git_dirty=False,
            git_diff_sha256="diff",
            package_versions={"ray": "test"},
        ),
    )


def test_trial_resume_requires_a_complete_valid_directory(tmp_path: Path) -> None:
    manifest = trial_manifest()
    action, trial_dir = prepare_trial_directory(tmp_path, manifest)
    assert action == "run"

    with pytest.raises(IncompleteTrialError, match="completion"):
        prepare_trial_directory(tmp_path, manifest)

    write_json_exclusive(trial_dir / "run_summary.json", {"completed": 60})
    complete_trial(
        trial_dir,
        manifest.trial.trial_id,
        ["trial_manifest.json", "run_summary.json"],
        completed_at=2.0,
    )

    action, resumed_dir = prepare_trial_directory(tmp_path, manifest)
    assert action == "skip"
    assert resumed_dir == trial_dir


def test_trial_resume_rejects_manifest_or_artifact_changes(tmp_path: Path) -> None:
    manifest = trial_manifest()
    _, trial_dir = prepare_trial_directory(tmp_path, manifest)
    write_json_exclusive(trial_dir / "run_summary.json", {"completed": 60})
    complete_trial(
        trial_dir,
        manifest.trial.trial_id,
        ["trial_manifest.json", "run_summary.json"],
        completed_at=2.0,
    )

    with pytest.raises(ValueError, match="manifest mismatch"):
        prepare_trial_directory(tmp_path, trial_manifest(created_at=3.0))

    (trial_dir / "run_summary.json").write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="artifact"):
        prepare_trial_directory(tmp_path, manifest)


def test_artifact_digests_are_content_stable(tmp_path: Path) -> None:
    first = {"b": [2, 1], "a": "value"}
    second = {"a": "value", "b": [2, 1]}
    path = tmp_path / "value.json"
    write_json_exclusive(path, first)

    assert stable_digest(first) == stable_digest(second)
    assert file_sha256(path) == file_sha256(path)
