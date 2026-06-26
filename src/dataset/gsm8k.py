from __future__ import annotations

import json
import re
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

from dataset.schema import Gsm8kEvaluation, TaskSample

ANSWER_MARKER = "####"
NUMBER_PATTERN = re.compile(r"[-+]?\$?\d[\d,]*(?:\.\d+)?")


def load_gsm8k_split(path: str | Path, split: str) -> list[TaskSample]:
    split_path = Path(path)
    samples = []
    for index, line in enumerate(split_path.read_text(encoding="utf-8").splitlines()):
        if not line.strip():
            continue
        raw_sample = json.loads(line)
        samples.append(normalize_gsm8k_record(raw_sample, split=split, index=index))
    return samples


def normalize_gsm8k_record(
    raw_sample: dict[str, Any],
    *,
    split: str,
    index: int,
) -> TaskSample:
    question = raw_sample["question"]
    answer = raw_sample["answer"]
    gold_answer = extract_gsm8k_gold_answer(answer)
    return TaskSample(
        sample_id=f"{split}_{index:05d}",
        source_dataset="gsm8k",
        split=split,
        task_type="math_word_problem",
        input_text=question,
        gold_answer=gold_answer,
        raw_answer=answer,
        quality_metric="numeric_exact_match",
    )


def extract_gsm8k_gold_answer(answer: str) -> str:
    marker_index = answer.rfind(ANSWER_MARKER)
    if marker_index < 0:
        raise ValueError("GSM8K answer missing final answer marker")
    value = answer[marker_index + len(ANSWER_MARKER) :].strip()
    if not value:
        raise ValueError("GSM8K final answer is empty")
    return normalize_numeric_answer(value)


def evaluate_gsm8k(model_output: str, gold_answer: str) -> Gsm8kEvaluation:
    normalized_gold = normalize_numeric_answer(gold_answer)
    predicted_answer = extract_final_numeric_answer(model_output)
    if predicted_answer is None:
        return Gsm8kEvaluation(
            exact_match=False,
            predicted_answer=None,
            gold_answer=normalized_gold,
            error_type="no_numeric_answer",
        )
    return Gsm8kEvaluation(
        exact_match=predicted_answer == normalized_gold,
        predicted_answer=predicted_answer,
        gold_answer=normalized_gold,
    )


def extract_final_numeric_answer(model_output: str) -> str | None:
    marker_index = model_output.rfind(ANSWER_MARKER)
    search_text = (
        model_output[marker_index + len(ANSWER_MARKER) :]
        if marker_index >= 0
        else model_output
    )
    matches = NUMBER_PATTERN.findall(search_text)
    if not matches:
        return None
    return normalize_numeric_answer(matches[-1])


def normalize_numeric_answer(answer: str) -> str:
    value = answer.strip().replace(",", "").replace("$", "")
    value = re.sub(r"\s+", "", value)
    if not value:
        raise ValueError("numeric answer is empty")
    try:
        decimal_value = Decimal(value)
    except InvalidOperation as exc:
        raise ValueError(f"invalid numeric answer: {answer}") from exc
    if decimal_value == decimal_value.to_integral_value():
        return format(decimal_value, "f").split(".", maxsplit=1)[0]
    return format(decimal_value.normalize(), "f")
