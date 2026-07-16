from __future__ import annotations

from pathlib import Path

import pytest

from experiment.workflow.artifacts import file_sha256
from experiment.workflow.config import ExperimentConfig, TrialSpec
from experiment.workflow.prepare import (
    load_arrival_trace,
    load_or_create_arrival_trace,
    load_prepared_experiment,
    prepare_experiment,
    validate_prepared_experiment,
)


def experiment_config(tmp_path: Path) -> ExperimentConfig:
    return ExperimentConfig(output_root=tmp_path / "workflow-output")


def open_loop_spec(strategy: str) -> TrialSpec:
    return TrialSpec.model_validate(
        {
            "scenario": "qmsum",
            "strategy": strategy,
            "gpu_count": 2,
            "workload": "open-loop",
            "repetition": 3,
            "load_percent": 75,
        }
    )


def test_prepare_freezes_and_validates_all_shared_inputs(tmp_path: Path) -> None:
    config = experiment_config(tmp_path)
    prepared = prepare_experiment(config)
    repeated = prepare_experiment(config)

    assert prepared == repeated == load_prepared_experiment(config)
    assert prepared == validate_prepared_experiment(config)
    assert prepared.root == config.output_root.resolve()
    assert len(prepared.trial_matrix) == 75
    assert prepared.trial_matrix_sha256 == file_sha256(prepared.trial_matrix_path)
    assert set(prepared.sample_manifest_paths) == {"qmsum", "mbpp"}
    assert prepared.synthetic_cache_sha256 == file_sha256(prepared.synthetic_cache_path)
    assert prepared.cache_generation_manifest_sha256 == file_sha256(
        prepared.cache_generation_manifest_path
    )
    assert prepared.parent_preflight_path.is_file()
    assert prepared.capacity_reuse_path.is_file()
    assert prepared.manifest.environment == repeated.manifest.environment
    assert len(prepared.manifest.sample_selection_manifests) == 2
    assert len(list((prepared.root / "prepared/arrivals").glob("*.manifest.json"))) == 10

    for scenario in ("qmsum", "mbpp"):
        permutations = [
            prepared.burst_permutation(scenario, repetition)
            for repetition in range(1, 6)
        ]
        assert len({permutation.sample_ids for permutation in permutations}) == 5
        assert all(len(permutation.sample_ids) == 24 for permutation in permutations)
        assert all(
            prepared.burst_permutation_path(scenario, repetition).is_relative_to(
                prepared.root
            )
            for repetition in range(1, 6)
        )

    paths = (
        prepared.trial_matrix_path,
        prepared.synthetic_cache_path,
        prepared.cache_generation_manifest_path,
        *prepared.sample_manifest_paths.values(),
    )
    assert all(path.is_relative_to(prepared.root) for path in paths)


def test_prepare_rejects_existing_content_mismatch(tmp_path: Path) -> None:
    config = experiment_config(tmp_path)
    prepared = prepare_experiment(config)
    prepared.trial_matrix_path.write_text("{}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="content mismatch"):
        prepare_experiment(config)

    assert prepared.trial_matrix_path.read_text(encoding="utf-8") == "{}\n"


def test_prepare_rejects_artifact_path_outside_output_root(tmp_path: Path) -> None:
    config = experiment_config(tmp_path)
    config.output_root.mkdir(parents=True)
    outside = tmp_path / "outside"
    outside.mkdir()
    (config.output_root / "prepared").symlink_to(outside, target_is_directory=True)

    with pytest.raises(ValueError, match="escapes experiment output root"):
        prepare_experiment(config)

    assert not (outside / "trial_matrix.jsonl").exists()


def test_arrival_trace_is_immutable_and_shared_across_strategies(
    tmp_path: Path,
) -> None:
    config = experiment_config(tmp_path)
    prepare_experiment(config)
    baseline = open_loop_spec("lg-batch")
    cache = open_loop_spec("wf-cache")

    first = load_arrival_trace(config, baseline)
    shared = load_arrival_trace(config, cache)
    loaded = load_arrival_trace(config, baseline)

    assert first == shared == loaded
    assert first.trace.absolute_sessions_per_sec == pytest.approx(0.0581119652)
    assert len(first.trace.arrivals) == 24
    assert first.trace.arrivals[0].offset_sec == 0
    assert first.sha256 == file_sha256(first.path)
    assert first.manifest_sha256 == file_sha256(first.manifest_path)
    assert first.path.is_relative_to(config.output_root.resolve())
    assert first.manifest_path.is_relative_to(config.output_root.resolve())

    with pytest.raises(ValueError, match="content mismatch"):
        load_or_create_arrival_trace(config, baseline, 3.0)
