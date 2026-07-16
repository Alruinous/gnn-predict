from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from statistics import fmean
from typing import Literal

from pydantic import BaseModel, ConfigDict, NonNegativeInt

from dataset.schema import TaskSample
from experiment.workflow.artifacts import write_json_exclusive, write_jsonl_exclusive
from experiment.workflow.mbpp import MbppWorkflowResult, parse_mbpp_result
from experiment.workflow.qmsum import evaluate_qmsum_output


class QualityModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class QmsumQualityRow(QualityModel):
    position: NonNegativeInt
    sample_id: str
    session_id: str
    rouge1: float
    rouge2: float
    rouge_l: float
    empty_output: bool


class QmsumQualitySummary(QualityModel):
    scenario: Literal["qmsum"] = "qmsum"
    sample_count: NonNegativeInt
    rouge1_mean: float
    rouge2_mean: float
    rouge_l_mean: float
    empty_output_count: NonNegativeInt


class QmsumQualityReport(QualityModel):
    rows: tuple[QmsumQualityRow, ...]
    summary: QmsumQualitySummary


class MbppQualityRow(QualityModel):
    position: NonNegativeInt
    sample_id: str
    session_id: str
    initial_pass: bool
    final_pass: bool
    initial_failure_type: str | None
    final_failure_type: str | None


class MbppQualitySummary(QualityModel):
    scenario: Literal["mbpp"] = "mbpp"
    sample_count: NonNegativeInt
    initial_pass_count: NonNegativeInt
    final_pass_count: NonNegativeInt
    initial_pass_rate: float
    final_pass_rate: float
    repaired_count: NonNegativeInt
    regressed_count: NonNegativeInt
    initial_failure_types: dict[str, NonNegativeInt]
    final_failure_types: dict[str, NonNegativeInt]


class MbppQualityReport(QualityModel):
    rows: tuple[MbppQualityRow, ...]
    summary: MbppQualitySummary


QualityReport = QmsumQualityReport | MbppQualityReport


def resolve_sample_sessions(
    samples: Sequence[TaskSample], sample_sessions: Mapping[str, str]
) -> tuple[tuple[TaskSample, str], ...]:
    sample_ids = tuple(sample.sample_id for sample in samples)
    if len(set(sample_ids)) != len(sample_ids):
        raise ValueError("samples contain duplicate sample IDs")
    expected = set(sample_ids)
    actual = set(sample_sessions)
    if actual != expected:
        raise ValueError(_set_mismatch("sample_sessions", expected, actual))
    session_ids = tuple(sample_sessions[sample_id] for sample_id in sample_ids)
    if any(
        not isinstance(session_id, str) or not session_id for session_id in session_ids
    ):
        raise ValueError("session IDs must be non-empty strings")
    if len(set(session_ids)) != len(session_ids):
        raise ValueError("sample_sessions contain duplicate session IDs")
    return tuple(zip(samples, session_ids, strict=True))


def summarize_qmsum_quality(
    samples: Sequence[TaskSample],
    sample_sessions: Mapping[str, str],
    session_outputs: Mapping[str, str],
) -> QmsumQualityReport:
    ordered = _resolve_outputs(samples, sample_sessions, session_outputs)
    rows = tuple(
        _qmsum_row(position, sample, session_id, output)
        for position, (sample, session_id, output) in enumerate(ordered)
    )
    return QmsumQualityReport(
        rows=rows,
        summary=QmsumQualitySummary(
            sample_count=len(rows),
            rouge1_mean=_mean(row.rouge1 for row in rows),
            rouge2_mean=_mean(row.rouge2 for row in rows),
            rouge_l_mean=_mean(row.rouge_l for row in rows),
            empty_output_count=sum(row.empty_output for row in rows),
        ),
    )


def summarize_mbpp_quality(
    samples: Sequence[TaskSample],
    sample_sessions: Mapping[str, str],
    session_outputs: Mapping[str, str],
) -> MbppQualityReport:
    ordered = _resolve_outputs(samples, sample_sessions, session_outputs)
    rows = tuple(
        _mbpp_row(position, sample, session_id, output)
        for position, (sample, session_id, output) in enumerate(ordered)
    )
    initial_pass_count = sum(row.initial_pass for row in rows)
    final_pass_count = sum(row.final_pass for row in rows)
    return MbppQualityReport(
        rows=rows,
        summary=MbppQualitySummary(
            sample_count=len(rows),
            initial_pass_count=initial_pass_count,
            final_pass_count=final_pass_count,
            initial_pass_rate=_rate(initial_pass_count, len(rows)),
            final_pass_rate=_rate(final_pass_count, len(rows)),
            repaired_count=sum(not row.initial_pass and row.final_pass for row in rows),
            regressed_count=sum(
                row.initial_pass and not row.final_pass for row in rows
            ),
            initial_failure_types=_failure_counts(rows, "initial_failure_type"),
            final_failure_types=_failure_counts(rows, "final_failure_type"),
        ),
    )


def write_quality_report(
    report: QualityReport, *, rows_path: Path, summary_path: Path
) -> None:
    write_jsonl_exclusive(
        rows_path, (row.model_dump(mode="json") for row in report.rows)
    )
    write_json_exclusive(summary_path, report.summary.model_dump(mode="json"))


def _resolve_outputs(
    samples: Sequence[TaskSample],
    sample_sessions: Mapping[str, str],
    session_outputs: Mapping[str, str],
) -> tuple[tuple[TaskSample, str, str], ...]:
    ordered = resolve_sample_sessions(samples, sample_sessions)
    expected = {session_id for _, session_id in ordered}
    actual = set(session_outputs)
    if actual != expected:
        raise ValueError(_set_mismatch("session_outputs", expected, actual))
    resolved: list[tuple[TaskSample, str, str]] = []
    for sample, session_id in ordered:
        output = session_outputs[session_id]
        if not isinstance(output, str):
            raise TypeError(f"session output {session_id!r} must be a string")
        resolved.append((sample, session_id, output))
    return tuple(resolved)


def _qmsum_row(
    position: int, sample: TaskSample, session_id: str, output: str
) -> QmsumQualityRow:
    evaluation = evaluate_qmsum_output(sample, output)
    return QmsumQualityRow(
        position=position,
        sample_id=sample.sample_id,
        session_id=session_id,
        rouge1=evaluation.rouge1,
        rouge2=evaluation.rouge2,
        rouge_l=evaluation.rouge_l,
        empty_output=evaluation.error_type == "no_summary",
    )


def _mbpp_row(
    position: int, sample: TaskSample, session_id: str, output: str
) -> MbppQualityRow:
    result = parse_mbpp_result(output)
    if result.sample_id != sample.sample_id:
        raise ValueError(
            f"session {session_id!r} returned sample {result.sample_id!r}, "
            f"expected {sample.sample_id!r}"
        )
    return MbppQualityRow(
        position=position,
        sample_id=sample.sample_id,
        session_id=session_id,
        initial_pass=result.initial_eval.passed,
        final_pass=result.final_eval.passed,
        initial_failure_type=_failure_type(result, initial=True),
        final_failure_type=_failure_type(result, initial=False),
    )


def _failure_type(result: MbppWorkflowResult, *, initial: bool) -> str | None:
    evaluation = result.initial_eval if initial else result.final_eval
    if evaluation.passed:
        if evaluation.error_type is not None:
            stage = "initial" if initial else "final"
            raise ValueError(f"passed {stage} MBPP evaluation has an error_type")
        return None
    if evaluation.error_type is None:
        stage = "initial" if initial else "final"
        raise ValueError(f"failed {stage} MBPP evaluation has no error_type")
    return evaluation.error_type


def _failure_counts(
    rows: Sequence[MbppQualityRow],
    field: Literal["initial_failure_type", "final_failure_type"],
) -> dict[str, int]:
    counts = Counter(
        value for row in rows if (value := getattr(row, field)) is not None
    )
    return dict(sorted(counts.items()))


def _mean(values: Iterable[float]) -> float:
    resolved = tuple(values)
    return fmean(resolved) if resolved else 0.0


def _rate(count: int, total: int) -> float:
    return count / total if total else 0.0


def _set_mismatch(label: str, expected: set[str], actual: set[str]) -> str:
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    return f"{label} mismatch: missing={missing}, extra={extra}"
