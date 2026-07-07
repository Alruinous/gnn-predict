from __future__ import annotations

import json
from pathlib import Path

import pytest

from dataset.schema import TaskSample
from scripts.motivation.analysis import summarize_mbpp, summarize_qmsum
from scripts.motivation.common import write_jsonl
from scripts.motivation.sampling import stratified_by_input_length


def make_sample(index: int, length: int) -> TaskSample:
    return TaskSample(
        sample_id=f"sample_{index}",
        source_dataset="qmsum",
        split="test",
        task_type="query_focused_summarization",
        input_text="x" * length,
        gold_answer="answer",
        quality_metric="summary_quality",
        metadata={"query": "query", "query_type": "general", "turn_count": 1},
    )


def test_stratified_by_input_length_is_stable() -> None:
    samples = [make_sample(index, index + 1) for index in range(30)]

    first = stratified_by_input_length(samples, total_count=9, seed=42)
    second = stratified_by_input_length(samples, total_count=9, seed=42)

    assert [sample.sample_id for sample in first] == [
        sample.sample_id for sample in second
    ]
    assert len(first) == 9


def test_qmsum_summary_computes_pipeline_oracle(tmp_path: Path) -> None:
    trace_path = tmp_path / "qmsum.jsonl"
    rows = [
        qmsum_event("s1", "chunk_0", 0.0, 2.0),
        qmsum_event("s1", "chunk_1", 0.0, 4.0),
        qmsum_event("s1", "chunk_2", 0.0, 3.0),
        qmsum_event("s1", "merge", 4.0, 5.0),
        qmsum_event("s1", "judge", 5.0, 6.0, quality=qmsum_quality()),
        qmsum_event("s2", "chunk_0", 5.0, 7.0),
        qmsum_event("s2", "chunk_1", 5.0, 9.0),
        qmsum_event("s2", "chunk_2", 5.0, 8.0),
        qmsum_event("s2", "merge", 9.0, 10.0),
        qmsum_event("s2", "judge", 10.0, 11.0, quality=qmsum_quality()),
    ]
    write_jsonl(trace_path, rows)

    summary = summarize_qmsum(trace_path, tmp_path / "summary.json")

    assert summary["baseline_total_sec"] == pytest.approx(10.0)
    assert summary["oracle_total_sec"] == pytest.approx(9.0)
    assert summary["pipeline_idle_time"]["total"] == pytest.approx(12.0)
    assert summary["speedup_upper_bound"] == pytest.approx(10.0 / 9.0)


def test_mbpp_summary_computes_resource_gap(tmp_path: Path) -> None:
    trace_path = tmp_path / "mbpp.jsonl"
    rows = [
        mbpp_event("s1", "coder", "llm", 0.0, 2.0, quality=None),
        mbpp_event("s1", "tester", "deterministic", 2.0, 3.0, quality=None),
        mbpp_event("s1", "reviewer", "llm", 3.0, 4.0, quality=None),
        mbpp_event("s1", "repair", "llm", 4.0, 6.0, quality=None),
        mbpp_event("s1", "final_tester", "deterministic", 6.0, 7.0, quality={"passed": True}),
    ]
    write_jsonl(trace_path, rows)

    summary = summarize_mbpp(trace_path, tmp_path / "summary.json")

    assert summary["resident_model_seconds"]["total"] == pytest.approx(21.0)
    assert summary["active_model_seconds"]["total"] == pytest.approx(5.0)
    assert summary["resource_gap"] == pytest.approx(4.2)
    assert summary["pass_at_1"] == pytest.approx(1.0)


def qmsum_event(
    sample_id: str,
    node_name: str,
    started_at: float,
    ended_at: float,
    *,
    quality: dict[str, object] | None = None,
) -> dict[str, object]:
    node_type = "evaluator" if node_name == "judge" else "llm"
    return base_event(sample_id, node_name, node_type, started_at, ended_at, quality)


def qmsum_quality() -> dict[str, object]:
    return {
        "rouge1": 1,
        "rouge2": 1,
        "rouge_l": 1,
        "llm_score": 4,
        "passed": True,
    }


def mbpp_event(
    sample_id: str,
    node_name: str,
    node_type: str,
    started_at: float,
    ended_at: float,
    *,
    quality: dict[str, object] | None,
) -> dict[str, object]:
    return base_event(sample_id, node_name, node_type, started_at, ended_at, quality)


def base_event(
    sample_id: str,
    node_name: str,
    node_type: str,
    started_at: float,
    ended_at: float,
    quality: dict[str, object] | None,
) -> dict[str, object]:
    return {
        "workflow_name": "test",
        "sample_id": sample_id,
        "node_name": node_name,
        "node_type": node_type,
        "started_at": started_at,
        "ended_at": ended_at,
        "duration_sec": ended_at - started_at,
        "ready_at": started_at,
        "release_at": ended_at,
        "pipeline_idle_time": 0.0,
        "model_name": None,
        "model_instance_id": None,
        "resident_started_at": None,
        "resident_ended_at": None,
        "input_token_count": 0,
        "output_token_count": 0,
        "status": "ok",
        "quality_result": quality,
        "output_text": json.dumps(quality or {}),
    }
