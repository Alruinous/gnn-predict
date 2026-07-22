"""Arrival-offset schedule for dataset_replay: poisson open-loop + backward-compat gap."""

from __future__ import annotations

from itertools import pairwise

import pytest

from experiment.workflow.experiments import dataset_replay
from experiment.workflow.plan import poisson_arrival_offsets


def test_poisson_offsets_match_shared_generator_and_are_reproducible() -> None:
    scenario = {
        "dataset": "qmsum",
        "workflow_name": "qmsum1",
        "arrival_process": "poisson",
        "arrival_rate_per_sec": 0.02,
    }
    offsets = dataset_replay._arrival_offsets(scenario, 40, seed=42)
    assert offsets == poisson_arrival_offsets(
        40, sessions_per_sec=0.02, seed=42, trace_key="qmsum1"
    )
    assert offsets[0] == 0.0
    assert len(offsets) == 40
    assert all(left < right for left, right in pairwise(offsets))
    assert dataset_replay._arrival_offsets(scenario, 40, seed=42) == offsets


def test_poisson_streams_are_independent_per_workflow() -> None:
    base = {
        "dataset": "qmsum",
        "arrival_process": "poisson",
        "arrival_rate_per_sec": 0.02,
    }
    first = dataset_replay._arrival_offsets(
        {**base, "workflow_name": "qmsum1"}, 30, seed=42
    )
    second = dataset_replay._arrival_offsets(
        {**base, "workflow_name": "qmsum2"}, 30, seed=42
    )
    assert first != second


def test_missing_arrival_process_defaults_to_burst_gap() -> None:
    scenario = {"dataset": "mbpp", "arrival_gap_sec": 0.0}
    assert dataset_replay._arrival_offsets(scenario, 5, seed=42) == (0.0,) * 5


def test_uniform_gap_produces_evenly_spaced_offsets() -> None:
    scenario = {"dataset": "mbpp", "arrival_process": "uniform", "arrival_gap_sec": 2.0}
    assert dataset_replay._arrival_offsets(scenario, 4, seed=42) == (0.0, 2.0, 4.0, 6.0)


def test_burst_process_ignores_gap() -> None:
    scenario = {"dataset": "mbpp", "arrival_process": "burst", "arrival_gap_sec": 5.0}
    assert dataset_replay._arrival_offsets(scenario, 3, seed=42) == (0.0, 0.0, 0.0)


def test_unknown_arrival_process_fails_fast() -> None:
    scenario = {"dataset": "mbpp", "arrival_process": "weibull"}
    with pytest.raises(ValueError, match="unknown arrival_process"):
        dataset_replay._arrival_offsets(scenario, 3, seed=42)


def test_scenario_sessions_threads_poisson_offsets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(
        dataset_replay._SCENARIOS,
        "qmsum",
        (lambda _path: tuple(range(50)), lambda sample: {"value": sample}),
    )
    scenario = {
        "dataset": "qmsum",
        "workflow_name": "qmsum1",
        "dataset_path": "ignored",
        "sample_count": 20,
        "arrival_process": "poisson",
        "arrival_rate_per_sec": 0.02,
    }
    sessions = dataset_replay._scenario_sessions(scenario, seed=7)
    expected = poisson_arrival_offsets(
        20, sessions_per_sec=0.02, seed=7, trace_key="qmsum1"
    )
    assert [session.arrival_offset_sec for session in sessions] == list(expected)
    assert [session.session_id for session in sessions] == [
        f"qmsum1-{index:04d}" for index in range(20)
    ]
    assert sessions[0].workflow_name == "qmsum1"
