from __future__ import annotations

import json
import os
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol, cast

from dotenv import load_dotenv
from openai import OpenAI
from rouge_score import rouge_scorer

from dataset.schema import SummaryEvaluation, TaskSample

MULTI_NEWS_SEPARATOR = "|||||"
NEWLINE_TOKEN = "NEWLINE_CHAR"
SUMMARY_JUDGE_PASS_SCORE = 4.0
SUMMARY_ROUGE_TYPES = ("rouge1", "rouge2", "rougeL")
SUMMARY_JUDGE_MAX_ATTEMPTS = 3
SUMMARY_JUDGE_RETRY_DELAY_SEC = 2.0


class SummaryJudge(Protocol):
    def evaluate(
        self,
        sample: TaskSample,
        generated_summary: str,
    ) -> dict[str, Any]: ...


class OpenRouterSummaryJudge:
    def __init__(
        self,
        client: OpenAI,
        model_name: str,
        max_attempts: int = SUMMARY_JUDGE_MAX_ATTEMPTS,
        retry_delay_sec: float = SUMMARY_JUDGE_RETRY_DELAY_SEC,
    ) -> None:
        self.client = client
        self.model_name = model_name
        self.max_attempts = max_attempts
        self.retry_delay_sec = retry_delay_sec

    @classmethod
    def from_env(cls) -> OpenRouterSummaryJudge:
        load_dotenv()
        base_url = os.environ.get("LLM_BASE_URL")
        api_key = os.environ.get("LLM_API_KEY")
        model_name = os.environ.get("LLM_NAME")
        missing = [
            name
            for name, value in {
                "LLM_BASE_URL": base_url,
                "LLM_API_KEY": api_key,
                "LLM_NAME": model_name,
            }.items()
            if not value
        ]
        if missing:
            raise ValueError(
                "summary_llm_judge requires environment variables: "
                + ", ".join(missing)
            )
        assert base_url is not None
        assert api_key is not None
        assert model_name is not None
        return cls(OpenAI(base_url=base_url, api_key=api_key), model_name)

    def evaluate(self, sample: TaskSample, generated_summary: str) -> dict[str, Any]:
        last_error: Exception | None = None
        for attempt in range(self.max_attempts):
            try:
                return self.evaluate_once(sample, generated_summary)
            except Exception as exc:
                last_error = exc
                if attempt + 1 >= self.max_attempts:
                    break
                time.sleep(self.retry_delay_sec * (attempt + 1))
        assert last_error is not None
        raise last_error

    def evaluate_once(
        self,
        sample: TaskSample,
        generated_summary: str,
    ) -> dict[str, Any]:
        response = self.client.chat.completions.create(
            model=self.model_name,
            temperature=0,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a strict reference-based summarization evaluator. "
                        "Return only one JSON object."
                    ),
                },
                {
                    "role": "user",
                    "content": build_summary_judge_prompt(sample, generated_summary),
                },
            ],
        )
        content = response.choices[0].message.content
        if not isinstance(content, str) or not content.strip():
            raise ValueError("summary judge returned empty content")
        return parse_summary_judge_response(content)


def load_multi_news_split(path: str | Path, split: str) -> list[TaskSample]:
    base_path = Path(path)
    source_path = base_path / f"{split}.src.cleaned"
    target_path = base_path / f"{split}.tgt"
    source_lines = source_path.read_text(encoding="utf-8").splitlines()
    target_lines = target_path.read_text(encoding="utf-8").splitlines()
    if len(source_lines) != len(target_lines):
        raise ValueError("Multi-News source and target files have different lengths")
    samples = []
    for index, (source, target) in enumerate(
        zip(source_lines, target_lines, strict=True)
    ):
        if not source.strip():
            continue
        samples.append(
            normalize_multi_news_record(source, target, split=split, index=index)
        )
    return samples


def normalize_multi_news_record(
    source: str,
    target: str,
    *,
    split: str,
    index: int,
) -> TaskSample:
    documents = [
        normalize_summary_text(document)
        for document in source.split(MULTI_NEWS_SEPARATOR)
        if normalize_summary_text(document)
    ]
    if not documents:
        raise ValueError(f"Multi-News sample has no documents: {split}_{index:05d}")
    input_text = format_multi_document_input(documents)
    reference = normalize_summary_text(target)
    return TaskSample(
        sample_id=f"{split}_{index:05d}",
        source_dataset="multi_news",
        split=split,
        task_type="multi_document_summarization",
        input_text=input_text,
        gold_answer=reference,
        raw_answer=target,
        quality_metric="summary_quality",
        metadata={
            "document_count": len(documents),
            "source_char_count": len(input_text),
            "reference_char_count": len(reference),
        },
    )


def load_qmsum_split(path: str | Path, split: str) -> list[TaskSample]:
    split_path = Path(path) / split
    if not split_path.is_dir():
        raise ValueError(f"QMSum split directory does not exist: {split_path}")
    samples: list[TaskSample] = []
    for meeting_path in sorted(split_path.glob("*.json")):
        raw_meeting = json.loads(meeting_path.read_text(encoding="utf-8"))
        samples.extend(normalize_qmsum_meeting(raw_meeting, split, meeting_path.stem))
    if not samples:
        raise ValueError(f"QMSum split is empty: {split}")
    return samples


def normalize_qmsum_meeting(
    raw_meeting: Mapping[str, Any],
    split: str,
    meeting_id: str,
) -> list[TaskSample]:
    transcript = format_qmsum_transcript(raw_meeting["meeting_transcripts"])
    samples: list[TaskSample] = []
    query_groups = [
        ("general", raw_meeting.get("general_query_list", [])),
        ("specific", raw_meeting.get("specific_query_list", [])),
    ]
    for query_type, queries in query_groups:
        if not isinstance(queries, list):
            raise ValueError(f"QMSum query list must be an array: {meeting_id}")
        for index, raw_query_payload in enumerate(queries):
            if not isinstance(raw_query_payload, dict):
                raise ValueError(f"QMSum query must be an object: {meeting_id}")
            query_payload = cast(dict[str, Any], raw_query_payload)
            query = normalize_summary_text(query_payload["query"])
            answer = normalize_summary_text(query_payload["answer"])
            samples.append(
                TaskSample(
                    sample_id=f"{split}_{meeting_id}_{query_type}_{index:03d}",
                    source_dataset="qmsum",
                    split=split,
                    task_type="query_focused_summarization",
                    input_text=transcript,
                    gold_answer=answer,
                    raw_answer=query_payload["answer"],
                    quality_metric="summary_quality",
                    metadata={
                        "meeting_id": meeting_id,
                        "query_type": query_type,
                        "query": query,
                        "relevant_text_span": query_payload.get(
                            "relevant_text_span",
                            [],
                        ),
                        "turn_count": len(raw_meeting["meeting_transcripts"]),
                    },
                )
            )
    return samples


def format_qmsum_transcript(turns: Sequence[Mapping[str, Any]]) -> str:
    lines = []
    for turn in turns:
        speaker = normalize_summary_text(str(turn["speaker"]))
        content = normalize_summary_text(str(turn["content"]))
        lines.append(f"{speaker}: {content}")
    return "\n\n".join(lines)


def format_multi_document_input(documents: Sequence[str]) -> str:
    return "\n\n".join(
        f"Document {index + 1}:\n{document}" for index, document in enumerate(documents)
    )


def normalize_summary_text(value: str) -> str:
    return " ".join(value.replace(NEWLINE_TOKEN, "\n").split())


def evaluate_summary_rouge(
    generated_summary: str,
    reference_summary: str,
) -> SummaryEvaluation:
    generated_summary = generated_summary.strip()
    if not generated_summary:
        return SummaryEvaluation(
            passed=False,
            rouge1=0.0,
            rouge2=0.0,
            rouge_l=0.0,
            error_type="no_summary",
            error_message="generated summary is empty",
        )
    scores = rouge_scorer.RougeScorer(
        list(SUMMARY_ROUGE_TYPES),
        use_stemmer=True,
    ).score(reference_summary, generated_summary)
    return SummaryEvaluation(
        passed=True,
        rouge1=scores["rouge1"].fmeasure,
        rouge2=scores["rouge2"].fmeasure,
        rouge_l=scores["rougeL"].fmeasure,
    )


def evaluate_summary_with_judge(
    sample: TaskSample,
    generated_summary: str,
    judge: SummaryJudge,
) -> SummaryEvaluation:
    if sample.gold_answer is None:
        raise ValueError(f"summary sample is missing gold_answer: {sample.sample_id}")
    base_result = evaluate_summary_rouge(generated_summary, sample.gold_answer)
    if not generated_summary.strip():
        return base_result
    try:
        judge_result = judge.evaluate(sample, generated_summary)
    except Exception as exc:
        return base_result.model_copy(
            update={
                "passed": False,
                "llm_score": None,
                "llm_passed": False,
                "error_type": "llm_judge_error",
                "error_message": f"summary judge failed: {type(exc).__name__}",
            }
        )
    llm_score = require_score(judge_result, "score")
    return base_result.model_copy(
        update={
            "passed": llm_score >= SUMMARY_JUDGE_PASS_SCORE,
            "llm_score": llm_score,
            "llm_passed": llm_score >= SUMMARY_JUDGE_PASS_SCORE,
            "coverage": require_score(judge_result, "coverage"),
            "relevance": require_score(judge_result, "relevance"),
            "coherence": require_score(judge_result, "coherence"),
            "faithfulness": require_score(judge_result, "faithfulness"),
            "judge_reason": str(judge_result.get("reason", "")),
        }
    )


def build_summary_judge_prompt(sample: TaskSample, generated_summary: str) -> str:
    query = str(sample.metadata.get("query", "Summarize the documents."))
    reference = sample.gold_answer or ""
    return json.dumps(
        {
            "task": "Evaluate the generated summary against the reference answer.",
            "dataset": sample.source_dataset,
            "query": query,
            "reference_summary": reference,
            "generated_summary": generated_summary,
            "rubric": {
                "score": "Overall quality from 0 to 5.",
                "coverage": "Whether key reference information is covered, 0 to 5.",
                "relevance": "Whether content answers the query, 0 to 5.",
                "coherence": "Whether the summary is clear and well organized, 0 to 5.",
                "faithfulness": (
                    "Whether it avoids contradicting the reference, 0 to 5."
                ),
            },
            "required_json_keys": [
                "score",
                "coverage",
                "relevance",
                "coherence",
                "faithfulness",
                "reason",
            ],
        },
        ensure_ascii=False,
    )


def parse_summary_judge_response(content: str) -> dict[str, Any]:
    payload = json.loads(extract_json_object(content))
    if not isinstance(payload, dict):
        raise ValueError("summary judge response must be a JSON object")
    for key in ["score", "coverage", "relevance", "coherence", "faithfulness"]:
        require_score(payload, key)
    return payload


def extract_json_object(content: str) -> str:
    stripped = content.strip()
    if stripped.startswith("{") and stripped.endswith("}"):
        return stripped
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start < 0 or end < start:
        raise ValueError("summary judge response does not contain a JSON object")
    return stripped[start : end + 1]


def require_score(payload: Mapping[str, Any], key: str) -> float:
    value = payload.get(key)
    if not isinstance(value, int | float) or isinstance(value, bool):
        raise ValueError(f"summary judge score must be numeric: {key}")
    score = float(value)
    if score < 0 or score > 5:
        raise ValueError(f"summary judge score must be between 0 and 5: {key}")
    return score
