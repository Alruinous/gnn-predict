from __future__ import annotations

from collections import Counter
from itertools import pairwise

import pytest

from experiment.workflow.config import EXPERIMENT_ID, ExperimentConfig, TrialSpec
from experiment.workflow.plan import (
    LOAD_PERCENTS,
    build_trial_matrix,
    poisson_arrival_offsets,
    session_permutation,
)


def test_formal_trial_matrix_matches_documented_counts() -> None:
    trials = build_trial_matrix()

    assert len(trials) == 75
    assert len({trial.trial_id for trial in trials}) == 75
    assert Counter((trial.scenario, trial.workload) for trial in trials) == {
        ("qmsum", "burst"): 25,
        ("qmsum", "open-loop"): 20,
        ("mbpp", "burst"): 30,
    }
    groups = {
        trial.model_copy(update={"repetition": 1}).trial_id
        for trial in trials
    }
    assert len(groups) == 15


def test_formal_trial_matrix_is_evidence_grouped_and_repetition_blocked() -> None:
    trials = build_trial_matrix()

    assert [trial.repetition for trial in trials[:15]] == [
        repetition for repetition in range(1, 6) for _ in range(3)
    ]
    assert [trial.strategy for trial in trials[:3]] == [
        "lg-batch",
        "wf-fifo",
        "wf-cache",
    ]
    assert [
        (trial.max_num_seqs, trial.queue_capacity) for trial in trials[15:17]
    ] == [(1, 16), (3, 1)]
    assert [
        (trial.load_percent, trial.strategy) for trial in trials[25:29]
    ] == [
        (75, "lg-batch"),
        (75, "wf-cache"),
        (125, "lg-batch"),
        (125, "wf-cache"),
    ]
    assert [
        (trial.strategy, trial.gpu_count) for trial in trials[45:49]
    ] == [
        ("lg-batch", 3),
        ("wf-cache", 1),
        ("wf-cache", 2),
        ("wf-cache", 3),
    ]
    assert [trial.strategy for trial in trials[65:67]] == [
        "wf-fifo",
        "wf-history",
    ]


def test_formal_config_matches_revised_budget() -> None:
    config = ExperimentConfig()

    assert EXPERIMENT_ID == "system_20260713_v2"
    assert config.session_count == 24
    assert config.trial_timeout_sec == 3_600.0
    assert config.campaign_timeout_sec == 86_400.0
    assert LOAD_PERCENTS == (75, 125)


def test_trial_id_contains_every_experimental_dimension() -> None:
    spec = TrialSpec(
        scenario="qmsum",
        strategy="wf-cache",
        gpu_count=2,
        workload="open-loop",
        repetition=3,
        max_num_seqs=2,
        queue_capacity=4,
        load_percent=75,
    )

    assert spec.trial_id == (
        "qmsum__wf-cache__g2__open-loop__load075__seq2__queue4__rep3"
    )
    assert spec.scheduler_policy == "cache"


def test_trial_workload_contract_is_strict() -> None:
    with pytest.raises(ValueError, match="open-loop"):
        TrialSpec(
            scenario="qmsum",
            strategy="wf-cache",
            gpu_count=2,
            workload="burst",
            repetition=1,
            load_percent=75,
        )


def test_permutations_and_arrivals_are_deterministic() -> None:
    sample_ids = [f"sample-{index}" for index in range(24)]

    first_order = session_permutation(sample_ids, seed=42, repetition=1)
    second_order = session_permutation(sample_ids, seed=42, repetition=1)
    other_order = session_permutation(sample_ids, seed=42, repetition=2)
    first_arrivals = poisson_arrival_offsets(
        24,
        sessions_per_sec=2.5,
        seed=42,
        trace_key="qmsum-g2-load75-rep1",
    )

    assert first_order == second_order
    assert first_order != other_order
    assert set(first_order) == set(sample_ids)
    assert first_arrivals == poisson_arrival_offsets(
        24,
        sessions_per_sec=2.5,
        seed=42,
        trace_key="qmsum-g2-load75-rep1",
    )
    assert first_arrivals[0] == 0.0
    assert all(left < right for left, right in pairwise(first_arrivals))
