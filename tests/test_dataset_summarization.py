from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from dataset.schema import TaskSample
from dataset.summarization import (
    OpenRouterSummaryJudge,
    evaluate_summary_rouge,
    evaluate_summary_with_judge,
    format_qmsum_transcript,
    load_multi_news_split,
    load_qmsum_split,
    parse_summary_judge_response,
)


class FakeSummaryJudge:
    def evaluate(
        self,
        sample: TaskSample,
        generated_summary: str,
    ) -> dict[str, Any]:
        assert sample.sample_id
        assert generated_summary
        return {
            "score": 4,
            "coverage": 5,
            "relevance": 4,
            "coherence": 4,
            "faithfulness": 5,
            "reason": "covers the key facts",
        }


class FailingSummaryJudge:
    def evaluate(
        self,
        sample: TaskSample,
        generated_summary: str,
    ) -> dict[str, Any]:
        assert sample.sample_id
        assert generated_summary
        raise RuntimeError("network failed")


def test_load_multi_news_split_normalizes_documents(tmp_path: Path) -> None:
    data_dir = tmp_path / "multi_news"
    data_dir.mkdir()
    (data_dir / "test.src.cleaned").write_text(
        "First NEWLINE_CHAR story.|||||Second story.\n",
        encoding="utf-8",
    )
    (data_dir / "test.tgt").write_text("Combined reference summary.\n", encoding="utf-8")

    samples = load_multi_news_split(data_dir, split="test")

    assert len(samples) == 1
    sample = samples[0]
    assert sample.sample_id == "test_00000"
    assert sample.source_dataset == "multi_news"
    assert sample.task_type == "multi_document_summarization"
    assert sample.quality_metric == "summary_quality"
    assert sample.metadata["document_count"] == 2
    assert "Document 1:" in sample.input_text
    assert sample.gold_answer == "Combined reference summary."


def test_load_multi_news_split_skips_empty_sources(tmp_path: Path) -> None:
    data_dir = tmp_path / "multi_news"
    data_dir.mkdir()
    (data_dir / "test.src.cleaned").write_text(
        "\nFirst story.|||||Second story.\n",
        encoding="utf-8",
    )
    (data_dir / "test.tgt").write_text("Missing source.\nReference.\n", encoding="utf-8")

    samples = load_multi_news_split(data_dir, split="test")

    assert len(samples) == 1
    assert samples[0].sample_id == "test_00001"


def test_load_qmsum_split_expands_general_and_specific_queries(tmp_path: Path) -> None:
    split_dir = tmp_path / "ALL" / "test"
    split_dir.mkdir(parents=True)
    payload = {
        "meeting_transcripts": [
            {"speaker": "A", "content": "First turn."},
            {"speaker": "B", "content": "Second turn."},
        ],
        "general_query_list": [
            {"query": "Summarize the whole meeting.", "answer": "Whole meeting."}
        ],
        "specific_query_list": [
            {
                "query": "What happened first?",
                "answer": "The first turn happened.",
                "relevant_text_span": [["0", "0"]],
            }
        ],
    }
    (split_dir / "meeting.json").write_text(json.dumps(payload), encoding="utf-8")

    samples = load_qmsum_split(tmp_path / "ALL", split="test")

    assert [sample.metadata["query_type"] for sample in samples] == [
        "general",
        "specific",
    ]
    assert samples[1].sample_id == "test_meeting_specific_000"
    assert samples[1].task_type == "query_focused_summarization"
    assert samples[1].metadata["query"] == "What happened first?"
    assert "A: First turn." in samples[1].input_text


def test_format_qmsum_transcript_separates_turns_for_chunking() -> None:
    transcript = format_qmsum_transcript(
        [
            {"speaker": "A", "content": "First turn."},
            {"speaker": "B", "content": "Second turn."},
        ]
    )

    assert transcript == "A: First turn.\n\nB: Second turn."


def test_evaluate_summary_rouge_reports_empty_output() -> None:
    result = evaluate_summary_rouge("", "reference summary")

    assert not result.passed
    assert result.error_type == "no_summary"
    assert result.rouge1 == 0.0


def test_evaluate_summary_with_judge_uses_structured_scores(
    tmp_path: Path,
) -> None:
    sample = load_multi_news_split(make_multi_news_fixture(tmp_path), split="test")[0]

    result = evaluate_summary_with_judge(
        sample,
        "Combined reference summary.",
        FakeSummaryJudge(),
    )

    assert result.passed
    assert result.llm_score == 4.0
    assert result.coverage == 5.0
    assert result.judge_reason == "covers the key facts"


def test_evaluate_summary_with_judge_records_judge_failure(
    tmp_path: Path,
) -> None:
    sample = load_multi_news_split(make_multi_news_fixture(tmp_path), split="test")[0]

    result = evaluate_summary_with_judge(
        sample,
        "Combined reference summary.",
        FailingSummaryJudge(),
    )

    assert not result.passed
    assert result.llm_passed is False
    assert result.error_type == "llm_judge_error"
    assert result.error_message == "summary judge failed: RuntimeError"


def test_parse_summary_judge_response_accepts_json_wrapped_in_text() -> None:
    payload = parse_summary_judge_response(
        'Result: {"score": 5, "coverage": 5, "relevance": 5, '
        '"coherence": 4, "faithfulness": 5, "reason": "ok"}'
    )

    assert payload["score"] == 5


def test_parse_summary_judge_response_rejects_invalid_score() -> None:
    with pytest.raises(ValueError, match="between 0 and 5"):
        parse_summary_judge_response(
            '{"score": 6, "coverage": 5, "relevance": 5, '
            '"coherence": 4, "faithfulness": 5, "reason": "bad"}'
        )


def test_openrouter_summary_judge_requires_env(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    for name in ["LLM_BASE_URL", "LLM_API_KEY", "LLM_NAME"]:
        monkeypatch.setenv(name, "")

    with pytest.raises(ValueError, match="LLM_BASE_URL"):
        OpenRouterSummaryJudge.from_env()


def make_multi_news_fixture(tmp_path: Path) -> Path:
    data_dir = tmp_path / "multi_news_eval"
    data_dir.mkdir()
    (data_dir / "test.src.cleaned").write_text(
        "First story.|||||Second story.\n",
        encoding="utf-8",
    )
    (data_dir / "test.tgt").write_text("Combined reference summary.\n", encoding="utf-8")
    return data_dir
