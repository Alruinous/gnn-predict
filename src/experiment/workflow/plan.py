from __future__ import annotations

import hashlib
import random
from collections.abc import Iterable, Sequence
from typing import Literal

from experiment.workflow.config import LoadPercent, TrialSpec

LOAD_PERCENTS: tuple[LoadPercent, ...] = (75, 125)
Repetition = Literal[1, 2, 3, 4, 5]
REPETITIONS: tuple[Repetition, ...] = (1, 2, 3, 4, 5)


def build_trial_matrix() -> tuple[TrialSpec, ...]:
    trials: dict[str, TrialSpec] = {}

    def add(spec: TrialSpec) -> None:
        existing = trials.get(spec.trial_id)
        if existing is not None and existing != spec:
            raise ValueError(f"trial id collision: {spec.trial_id}")
        trials[spec.trial_id] = spec

    for repetition in REPETITIONS:
        for strategy in ("lg-batch", "wf-fifo", "wf-cache"):
            add(
                TrialSpec(
                    scenario="qmsum",
                    strategy=strategy,
                    gpu_count=2,
                    workload="burst",
                    repetition=repetition,
                )
            )
    for repetition in REPETITIONS:
        for max_num_seqs, queue_capacity in ((1, 16), (3, 1)):
            add(
                TrialSpec(
                    scenario="qmsum",
                    strategy="wf-cache",
                    gpu_count=2,
                    workload="burst",
                    repetition=repetition,
                    max_num_seqs=max_num_seqs,
                    queue_capacity=queue_capacity,
                )
            )

    for repetition in REPETITIONS:
        for load_percent in LOAD_PERCENTS:
            for strategy in ("lg-batch", "wf-cache"):
                add(
                    TrialSpec(
                        scenario="qmsum",
                        strategy=strategy,
                        gpu_count=2,
                        workload="open-loop",
                        repetition=repetition,
                        load_percent=load_percent,
                    )
                )

    for repetition in REPETITIONS:
        add(
            TrialSpec(
                scenario="mbpp",
                strategy="lg-batch",
                gpu_count=3,
                workload="burst",
                repetition=repetition,
            )
        )
        for gpu_count in (1, 2, 3):
            add(
                TrialSpec(
                    scenario="mbpp",
                    strategy="wf-cache",
                    gpu_count=gpu_count,
                    workload="burst",
                    repetition=repetition,
                )
            )

    for repetition in REPETITIONS:
        for strategy in ("wf-fifo", "wf-history"):
            add(
                TrialSpec(
                    scenario="mbpp",
                    strategy=strategy,
                    gpu_count=2,
                    workload="burst",
                    repetition=repetition,
                )
            )

    ordered = tuple(trials.values())
    if len(ordered) != 75:
        raise RuntimeError(f"formal matrix must contain 75 trials, got {len(ordered)}")
    return ordered


def session_permutation(
    sample_ids: Sequence[str],
    *,
    seed: int,
    repetition: int,
) -> tuple[str, ...]:
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("sample ids must be unique")
    ordered = sorted(sample_ids)
    random.Random(_derived_seed(seed, "permutation", repetition)).shuffle(ordered)
    return tuple(ordered)


def poisson_arrival_offsets(
    count: int,
    *,
    sessions_per_sec: float,
    seed: int,
    trace_key: str,
) -> tuple[float, ...]:
    if count <= 0:
        raise ValueError("arrival count must be positive")
    if sessions_per_sec <= 0:
        raise ValueError("arrival rate must be positive")
    rng = random.Random(_derived_seed(seed, "arrival", trace_key))
    offsets = [0.0]
    for _ in range(count - 1):
        offsets.append(offsets[-1] + rng.expovariate(sessions_per_sec))
    return tuple(offsets)


def calibration_key(scenario: str, gpu_count: int) -> str:
    return f"{scenario}__wf-cache__g{gpu_count}"


def select_trials(
    trials: Iterable[TrialSpec],
    *,
    trial_ids: set[str] | None = None,
    scenario: str | None = None,
) -> tuple[TrialSpec, ...]:
    selected = tuple(
        trial
        for trial in trials
        if (trial_ids is None or trial.trial_id in trial_ids)
        and (scenario is None or trial.scenario == scenario)
    )
    if trial_ids is not None:
        missing = trial_ids - {trial.trial_id for trial in selected}
        if missing:
            raise KeyError(f"unknown trial ids: {', '.join(sorted(missing))}")
    return selected


def _derived_seed(seed: int, *parts: object) -> int:
    payload = "|".join((str(seed), *(str(part) for part in parts)))
    return int.from_bytes(hashlib.sha256(payload.encode()).digest()[:8], "big")
