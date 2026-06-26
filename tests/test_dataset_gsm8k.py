from __future__ import annotations

from pathlib import Path

import pytest

from dataset.gsm8k import evaluate_gsm8k, load_gsm8k_split


def test_load_gsm8k_split_normalizes_samples(tmp_path: Path) -> None:
    path = tmp_path / "test.jsonl"
    path.write_text(
        (
            '{"question": "How many?", "answer": "Work\\n#### 70,000"}\n'
            '{"question": "How much?", "answer": "Work\\n#### $18.0"}\n'
        ),
        encoding="utf-8",
    )

    samples = load_gsm8k_split(path, split="test")

    assert [sample.sample_id for sample in samples] == ["test_00000", "test_00001"]
    assert samples[0].source_dataset == "gsm8k"
    assert samples[0].task_type == "math_word_problem"
    assert samples[0].gold_answer == "70000"
    assert samples[1].gold_answer == "18"
    assert samples[0].quality_metric == "numeric_exact_match"


def test_load_gsm8k_split_rejects_missing_answer_marker(tmp_path: Path) -> None:
    path = tmp_path / "test.jsonl"
    path.write_text(
        '{"question": "How many?", "answer": "No final marker"}\n',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="missing final answer marker"):
        load_gsm8k_split(path, split="test")


def test_evaluate_gsm8k_matches_normalized_numeric_answer() -> None:
    result = evaluate_gsm8k("Reasoning therefore #### $70,000.0", "70000")

    assert result.exact_match
    assert result.predicted_answer == "70000"
    assert result.gold_answer == "70000"
    assert result.error_type is None


def test_evaluate_gsm8k_extracts_last_number_without_marker() -> None:
    result = evaluate_gsm8k("First 10, final answer is 18.0", "$18")

    assert result.exact_match
    assert result.predicted_answer == "18"


def test_evaluate_gsm8k_handles_large_integral_answer() -> None:
    large_answer = "123456789012345678901234567890"

    result = evaluate_gsm8k(f"Reasoning therefore #### {large_answer}.0", large_answer)

    assert result.exact_match
    assert result.predicted_answer == large_answer


def test_evaluate_gsm8k_reports_missing_numeric_answer() -> None:
    result = evaluate_gsm8k("No final answer here", "18")

    assert not result.exact_match
    assert result.predicted_answer is None
    assert result.error_type == "no_numeric_answer"
