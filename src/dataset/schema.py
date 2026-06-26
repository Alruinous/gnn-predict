from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints

TaskType = Literal[
    "math_word_problem",
    "python_function_generation",
    "multi_document_summarization",
    "query_focused_summarization",
]
QualityMetric = Literal["numeric_exact_match", "pass_at_1", "summary_quality"]
NonEmptyStr = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1),
]


class TaskSample(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sample_id: NonEmptyStr
    source_dataset: NonEmptyStr
    split: NonEmptyStr
    task_type: TaskType
    input_text: NonEmptyStr
    quality_metric: QualityMetric
    gold_answer: str | None = None
    raw_answer: str | None = None
    reference_code: str | None = None
    test_imports: list[str] = Field(default_factory=list)
    test_list: list[str] = Field(default_factory=list)
    metadata: dict[str, object] = Field(default_factory=dict)


class Gsm8kEvaluation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    exact_match: bool
    predicted_answer: str | None
    gold_answer: str
    error_type: str | None = None


class MbppEvaluation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    passed: bool
    passed_tests: int
    total_tests: int
    error_type: str | None = None
    error_message: str | None = None


class SummaryEvaluation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    passed: bool
    rouge1: float
    rouge2: float
    rouge_l: float
    llm_score: float | None = None
    llm_passed: bool | None = None
    coverage: float | None = None
    relevance: float | None = None
    coherence: float | None = None
    faithfulness: float | None = None
    judge_reason: str | None = None
    error_type: str | None = None
    error_message: str | None = None
