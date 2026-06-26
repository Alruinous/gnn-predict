from __future__ import annotations

from dataset.schema import TaskSample
from scripts.build_workflow_pilot_samples import stratified_sample_by_length


def test_stratified_sample_by_length_selects_each_bucket() -> None:
    samples = [
        TaskSample(
            sample_id=f"sample_{index}",
            source_dataset="test",
            split="test",
            task_type="multi_document_summarization",
            input_text="x" * (index + 1),
            gold_answer="answer",
            quality_metric="summary_quality",
        )
        for index in range(9)
    ]

    rows = stratified_sample_by_length(samples, samples_per_stratum=1, seed=1)

    assert [row["stratum"] for row in rows] == ["short", "medium", "long"]
    assert len({row["sample_id"] for row in rows}) == 3
