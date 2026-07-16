from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, cast

import pytest

import experiment.workflow.aggregate as aggregate_module
from experiment.workflow.aggregate import (
    aggregate_trial_metrics,
    regenerate_analysis,
    scan_completed_trials,
)
from experiment.workflow.analysis import read_jsonl
from experiment.workflow.artifacts import (
    EnvironmentManifest,
    TrialManifest,
    complete_trial,
    file_sha256,
    write_json_exclusive,
    write_jsonl_exclusive,
)
from experiment.workflow.config import TrialSpec


def test_group_metrics_require_five_aligned_repetitions() -> None:
    rows = [
        {
            "trial_id": f"trial-{repetition}",
            "experiment_id": "system_20260713",
            "scenario": "qmsum",
            "strategy": "wf-cache",
            "gpu_count": 2,
            "workload": "burst",
            "repetition": repetition,
            "max_num_seqs": 3,
            "queue_capacity": 16,
            "load_percent": None,
            "alignment_fingerprint": "aligned",
            "alignment": {"sample_manifest_sha256": "sample"},
            "metrics": {
                "makespan_sec": float(repetition),
                "latency": {"p95": float(repetition * 2)},
                "timestamps": [repetition],
            },
        }
        for repetition in range(1, 6)
    ]

    assert aggregate_trial_metrics(rows[:4]) == []

    groups = aggregate_trial_metrics(rows)

    assert len(groups) == 1
    assert groups[0]["trial_count"] == 5
    metrics = cast(dict[str, Any], groups[0]["metrics"])
    assert metrics["makespan_sec"]["mean"] == 3.0
    assert metrics["latency"]["p95"]["mean"] == 6.0
    assert "timestamps" not in metrics

    rows[-1]["alignment_fingerprint"] = "drifted"
    assert aggregate_trial_metrics(rows) == []


def test_analysis_generation_switch_is_atomic(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    analysis_dir = tmp_path / "analysis"
    trial_path, group_path = aggregate_module._publish_analysis(
        analysis_dir,
        [{"generation": 1}],
        [{"generation": 1}],
    )
    original_trial = trial_path.read_text(encoding="utf-8")
    original_group = group_path.read_text(encoding="utf-8")
    write = aggregate_module._write_jsonl_unpublished
    calls = 0

    def fail_second(path: Path, rows: list[object]) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("injected publish failure")
        write(path, rows)

    monkeypatch.setattr(aggregate_module, "_write_jsonl_unpublished", fail_second)

    with pytest.raises(OSError, match="injected"):
        aggregate_module._publish_analysis(
            analysis_dir,
            [{"generation": 2}],
            [{"generation": 2}],
        )

    assert trial_path.read_text(encoding="utf-8") == original_trial
    assert group_path.read_text(encoding="utf-8") == original_group


def test_alignment_ignores_dynamic_pre_run_gpu_state() -> None:
    first = {
        "driver_version": "driver",
        "gpus": [
            {
                "uuid": "gpu",
                "total_memory_mb": 32768.0,
                "used_memory_mb": 273.0,
                "compute_process_pids": [],
                "graphics_process_pids": [],
            }
        ],
    }
    second = {
        "driver_version": "driver",
        "gpus": [
            {
                "uuid": "gpu",
                "total_memory_mb": 32768.0,
                "used_memory_mb": 280.0,
                "compute_process_pids": [42],
                "graphics_process_pids": [],
            }
        ],
    }

    assert aggregate_module._stable_serving_environment(first) == (
        aggregate_module._stable_serving_environment(second)
    )


def test_completed_workflow_rejects_unmatched_trace_pairs(tmp_path: Path) -> None:
    output_root = tmp_path / "experiment"
    _completed_trial(output_root, 1, unmatched_acquire=True)

    with pytest.raises(ValueError, match="unmatched events"):
        regenerate_analysis(output_root)


def test_regeneration_validates_completed_trials_without_mutating_raw_data(
    tmp_path: Path,
) -> None:
    output_root = tmp_path / "experiment"
    trial_dirs = [
        _completed_trial(output_root, repetition) for repetition in range(1, 6)
    ]
    (output_root / "trials" / "incomplete").mkdir()
    raw_hashes = {
        path: file_sha256(path)
        for trial_dir in trial_dirs
        for path in trial_dir.iterdir()
        if path.is_file()
    }

    first = regenerate_analysis(output_root)
    first_trial_metrics = first.trial_metrics_path.read_text(encoding="utf-8")
    first_group_metrics = first.group_metrics_path.read_text(encoding="utf-8")
    second = regenerate_analysis(output_root)

    assert first.completed_trial_count == 5
    assert first.aligned_group_count == 1
    assert second.trial_metrics_path.read_text(encoding="utf-8") == (
        first_trial_metrics
    )
    assert second.group_metrics_path.read_text(encoding="utf-8") == (
        first_group_metrics
    )
    assert len(read_jsonl(first.trial_metrics_path)) == 5
    assert len(read_jsonl(first.group_metrics_path)) == 1
    assert raw_hashes == {path: file_sha256(path) for path in raw_hashes}
    assert not list(first.trial_metrics_path.parent.glob("*.tmp"))

    trace_path = trial_dirs[0] / "workflow_trace.jsonl"
    trace_path.write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="artifact"):
        scan_completed_trials(output_root)


def _completed_trial(
    output_root: Path,
    repetition: int,
    *,
    unmatched_acquire: bool = False,
) -> Path:
    trial = TrialSpec(
        scenario="qmsum",
        strategy="wf-cache",
        gpu_count=2,
        workload="burst",
        repetition=cast(Literal[1, 2, 3, 4, 5], repetition),
    )
    trial_dir = output_root / "trials" / trial.trial_id
    trial_dir.mkdir(parents=True)
    manifest = TrialManifest(
        experiment_id="system_20260713",
        trial=trial,
        created_at=1.0,
        sample_manifest_sha256="sample",
        prediction_cache_sha256="cache",
        arrival_trace_sha256="arrival",
        workflow_config={"nodes": []},
        scheduler_config={"policy": "cache"},
        serving_environment={"vllm_version": "test"},
        telemetry_config={
            "interval_sec": 0.2,
            "physical_gpu_ids": [0],
            "raw_samples_retained": True,
            "energy_baseline_method": ("per_gpu_first_sample_clamped_subtraction"),
            "memory_baseline_method": ("per_gpu_first_sample_clamped_subtraction"),
        },
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
    write_json_exclusive(trial_dir / "trial_manifest.json", manifest)
    write_json_exclusive(
        trial_dir / "telemetry_config.json",
        manifest.telemetry_config,
    )
    trace = [
        _event("run_started", 0.0),
        _event("session_completed", 1.0, latency_sec=1.0),
        _event("session_completed", 2.0, latency_sec=2.0),
    ]
    if unmatched_acquire:
        trace.append(
            {
                "event_type": "acquire_requested",
                "ts": 2.0,
                "acquire_id": "unmatched",
                "payload": {},
            }
        )
    trace.append(_event("run_finished", float(repetition + 3)))
    write_jsonl_exclusive(trial_dir / "workflow_trace.jsonl", trace)
    write_jsonl_exclusive(
        trial_dir / "gpu_telemetry.jsonl",
        (
            {
                "gpu_index": 0,
                "ts": 0.0,
                "power_watts": 50.0,
                "memory_used_mb": 1024.0,
                "gpu_utilization_percent": 0.0,
                "memory_utilization_percent": 0.0,
            },
            {
                "gpu_index": 0,
                "ts": 2.0,
                "power_watts": 100.0,
                "memory_used_mb": 2048.0,
                "gpu_utilization_percent": 50.0,
                "memory_utilization_percent": 25.0,
            },
        ),
    )
    write_jsonl_exclusive(
        trial_dir / "queue_telemetry.jsonl",
        (
            {"node_id": "stage", "ts": 0.0, "queue_size": 0},
            {"node_id": "stage", "ts": 2.0, "queue_size": 1},
        ),
    )
    complete_trial(
        trial_dir,
        trial.trial_id,
        (
            "gpu_telemetry.jsonl",
            "queue_telemetry.jsonl",
            "telemetry_config.json",
            "trial_manifest.json",
            "workflow_trace.jsonl",
        ),
        completed_at=10.0,
    )
    return trial_dir


def _event(
    event_type: str,
    timestamp: float,
    *,
    latency_sec: float | None = None,
) -> dict[str, object]:
    payload = {} if latency_sec is None else {"latency_sec": latency_sec}
    return {"event_type": event_type, "ts": timestamp, "payload": payload}
