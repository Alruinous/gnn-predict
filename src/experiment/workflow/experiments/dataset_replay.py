"""QMSum + MBPP dataset replay: feed samples through workflows, write trace."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from pydantic import JsonValue

from dataset.schema import TaskSample
from experiment.workflow.experiments.driver import (
    SessionInvocation,
    submit_sessions,
    wait_terminal,
)
from experiment.workflow.mbpp import (
    build_mbpp_session_inputs,
    load_mbpp_experiment_samples,
)
from experiment.workflow.qmsum import (
    build_qmsum_session_inputs,
    load_qmsum_experiment_samples,
)
from workflow.fleet import WorkflowFleet

SampleLoader = Callable[[str], tuple[TaskSample, ...]]
InputBuilder = Callable[[TaskSample], dict[str, JsonValue]]

_SCENARIOS: dict[str, tuple[SampleLoader, InputBuilder]] = {
    "qmsum": (load_qmsum_experiment_samples, build_qmsum_session_inputs),
    "mbpp": (load_mbpp_experiment_samples, build_mbpp_session_inputs),
}


def run(
    fleet: WorkflowFleet,
    *,
    output_dir: Path,
    run_id: str,
    config: Mapping[str, Any],
) -> None:
    sessions: list[SessionInvocation] = []
    for scenario in config["scenarios"]:
        sessions.extend(_scenario_sessions(scenario))
    submit_sessions(fleet, sessions)
    wait_terminal(fleet, sessions, float(config.get("timeout_sec", 1800.0)))


def _scenario_sessions(scenario: Mapping[str, Any]) -> list[SessionInvocation]:
    dataset = scenario["dataset"]
    if dataset not in _SCENARIOS:
        raise ValueError(f"unknown dataset: {dataset}")
    workflow_name = scenario.get("workflow_name", dataset)
    arrival_gap_sec = float(scenario.get("arrival_gap_sec", 0.0))
    load_samples, build_inputs = _SCENARIOS[dataset]
    samples = load_samples(scenario["dataset_path"])
    sample_count = scenario.get("sample_count")
    if sample_count is not None:
        samples = samples[:sample_count]
    return [
        SessionInvocation(
            workflow_name=workflow_name,
            session_id=f"{workflow_name}-{index:04d}",
            inputs=build_inputs(sample),
            arrival_offset_sec=index * arrival_gap_sec,
        )
        for index, sample in enumerate(samples)
    ]
