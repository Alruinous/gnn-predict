from __future__ import annotations

import random
from collections.abc import Callable, Sequence

from dataset.schema import TaskSample


def stratified_by_input_length(
    samples: Sequence[TaskSample],
    *,
    total_count: int,
    seed: int,
    strata_count: int = 3,
) -> list[TaskSample]:
    assert total_count % strata_count == 0, (total_count, strata_count)
    per_stratum = total_count // strata_count
    ordered = sorted(samples, key=lambda sample: len(sample.input_text))
    strata = split_evenly(ordered, strata_count)
    selected: list[TaskSample] = []
    rng = random.Random(seed)
    for stratum in strata:
        assert len(stratum) >= per_stratum, (len(stratum), per_stratum)
        selected.extend(rng.sample(list(stratum), per_stratum))
    return sorted(selected, key=sample_sort_key)


def split_evenly[T](items: Sequence[T], part_count: int) -> list[Sequence[T]]:
    assert part_count > 0, part_count
    size = len(items)
    return [
        items[(size * index) // part_count : (size * (index + 1)) // part_count]
        for index in range(part_count)
    ]


def qmsum_sample_rows(samples: Sequence[TaskSample]) -> list[dict[str, object]]:
    return [
        {
            "sample_id": sample.sample_id,
            "input_char_count": len(sample.input_text),
            "query": sample.metadata["query"],
            "query_type": sample.metadata["query_type"],
            "turn_count": sample.metadata["turn_count"],
        }
        for sample in samples
    ]


def mbpp_sample_rows(samples: Sequence[TaskSample]) -> list[dict[str, object]]:
    return [
        {
            "sample_id": sample.sample_id,
            "task_id": sample.metadata["task_id"],
            "input_char_count": len(sample.input_text),
            "test_count": len(sample.test_list),
        }
        for sample in samples
    ]


def sample_sort_key(sample: TaskSample) -> tuple[int, str]:
    value = sample.metadata.get("task_id")
    if isinstance(value, int):
        return value, sample.sample_id
    return 0, sample.sample_id


def filter_samples(
    samples: Sequence[TaskSample],
    predicate: Callable[[TaskSample], bool],
) -> list[TaskSample]:
    return [sample for sample in samples if predicate(sample)]
