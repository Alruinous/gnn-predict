from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from pathlib import Path

import pytest

from dataset.schema import MbppEvaluation, TaskSample
from experiment.workflow.mbpp import MbppAttempt, MbppWorkflowResult
from experiment.workflow.preflight import (
    PromptLimitExceededError,
    preflight_mbpp,
    preflight_qmsum,
    run_formal_preflight,
    write_preflight_report,
)
from experiment.workflow.quality import (
    summarize_mbpp_quality,
    summarize_qmsum_quality,
    write_quality_report,
)
from workflow.schema import ExecutionConfig
from workflow.tokenizer import PromptTokenizer


def test_qmsum_quality_is_paired_by_fixed_sample_and_session() -> None:
    samples = (qmsum_sample("q-1"), qmsum_sample("q-2"))
    report = summarize_qmsum_quality(
        samples,
        {"q-1": "session-b", "q-2": "session-a"},
        {"session-a": "", "session-b": "The answer is one."},
    )

    assert [row.sample_id for row in report.rows] == ["q-1", "q-2"]
    assert [row.session_id for row in report.rows] == ["session-b", "session-a"]
    assert report.rows[0].rouge_l == pytest.approx(1.0)
    assert report.rows[1].empty_output is True
    assert report.summary.rouge1_mean == pytest.approx(0.5)
    assert report.summary.rouge2_mean == pytest.approx(0.5)
    assert report.summary.rouge_l_mean == pytest.approx(0.5)
    assert report.summary.empty_output_count == 1


def test_quality_rejects_missing_or_duplicated_session_contracts() -> None:
    samples = (qmsum_sample("q-1"), qmsum_sample("q-2"))

    with pytest.raises(ValueError, match="session_outputs mismatch"):
        summarize_qmsum_quality(
            samples,
            {"q-1": "session-1", "q-2": "session-2"},
            {"session-1": "answer"},
        )
    with pytest.raises(ValueError, match="duplicate session IDs"):
        summarize_qmsum_quality(
            samples,
            {"q-1": "session-1", "q-2": "session-1"},
            {"session-1": "answer"},
        )


def test_mbpp_quality_reports_initial_final_pass_and_failure_types() -> None:
    samples = (mbpp_sample("11"), mbpp_sample("12"))
    first = mbpp_result(
        "11",
        initial=evaluation(False, "assertion_error"),
        final=evaluation(True),
    )
    second = mbpp_result(
        "12",
        initial=evaluation(True),
        final=evaluation(False, "timeout"),
    )
    report = summarize_mbpp_quality(
        samples,
        {"11": "session-11", "12": "session-12"},
        {
            "session-11": first.model_dump_json(),
            "session-12": second.model_dump_json(),
        },
    )

    assert report.summary.initial_pass_count == 1
    assert report.summary.final_pass_count == 1
    assert report.summary.initial_pass_rate == pytest.approx(0.5)
    assert report.summary.final_pass_rate == pytest.approx(0.5)
    assert report.summary.repaired_count == 1
    assert report.summary.regressed_count == 1
    assert report.summary.initial_failure_types == {"assertion_error": 1}
    assert report.summary.final_failure_types == {"timeout": 1}


def test_mbpp_quality_rejects_output_for_another_sample() -> None:
    sample = mbpp_sample("11")
    output = mbpp_result(
        "12",
        initial=evaluation(True),
        final=evaluation(True),
    )

    with pytest.raises(ValueError, match="returned sample '12', expected '11'"):
        summarize_mbpp_quality(
            (sample,),
            {"11": "session-11"},
            {"session-11": output.model_dump_json()},
        )


def test_qmsum_preflight_uses_chat_template_for_all_agent_prompts() -> None:
    sample = qmsum_sample("q-1")
    tokenizers = TokenizerHarness()
    outputs = {"q-1": {f"chunk_{index}": f"partial {index}" for index in range(6)}}

    report = preflight_qmsum(
        (sample,),
        {"q-1": "session-q"},
        outputs,
        tokenizer_factory=tokenizers,
    )

    assert report.summary.valid is True
    assert report.summary.prompt_count == 7
    assert [row.node_id for row in report.rows] == [
        "chunk_0",
        "chunk_1",
        "chunk_2",
        "chunk_3",
        "chunk_4",
        "chunk_5",
        "merge",
    ]
    assert len(tokenizers.backends) == 2
    calls = [call for backend in tokenizers.backends.values() for call in backend.calls]
    assert len(calls) == 7
    assert all(call[1]["truncation"] is False for call in calls)
    assert all("<user>" in call[0] for call in calls)
    merge_prompt = tokenizers.backends["/data/Models/Qwen/Qwen3-8B"].calls[0][0]
    assert '"chunk_0": "partial 0"' in merge_prompt
    assert '"chunk_5": "partial 5"' in merge_prompt


def test_mbpp_preflight_replays_tester_and_reviewer_outputs() -> None:
    sample = mbpp_sample("11")
    code = "def add_one(value):\n    return value"
    attempt = MbppAttempt(code=code, initial_eval=evaluation(False, "assertion_error"))
    tokenizers = TokenizerHarness()

    report = preflight_mbpp(
        (sample,),
        {"11": "session-11"},
        {
            "11": {
                "coder": code,
                "tester": attempt.model_dump_json(),
                "reviewer": "The result is off by one.",
            }
        },
        tokenizer_factory=tokenizers,
    )

    assert report.summary.prompt_count == 3
    assert [row.node_id for row in report.rows] == ["coder", "reviewer", "repair"]
    reviewer_prompt = tokenizers.backends["/data/Models/Qwen/Qwen3-4B"].calls[0][0]
    repair_prompt = tokenizers.backends["/data/Models/Qwen/Qwen3-8B"].calls[0][0]
    assert "assertion_error" in reviewer_prompt
    assert "The result is off by one." in repair_prompt


def test_preflight_rejects_missing_dynamic_output_and_token_overflow() -> None:
    sample = qmsum_sample("q-1")
    outputs = {"q-1": {f"chunk_{index}": f"partial {index}" for index in range(5)}}

    with pytest.raises(ValueError, match="chunk_5"):
        preflight_qmsum(
            (sample,),
            {"q-1": "session-q"},
            outputs,
            tokenizer_factory=TokenizerHarness(),
        )

    overflow_tokenizers = TokenizerHarness(token_count=lambda _: 9_000)
    complete_outputs = {
        "q-1": {f"chunk_{index}": f"partial {index}" for index in range(6)}
    }
    with pytest.raises(PromptLimitExceededError) as error:
        preflight_qmsum(
            (sample,),
            {"q-1": "session-q"},
            complete_outputs,
            tokenizer_factory=overflow_tokenizers,
        )
    assert error.value.report.summary.overflow_count == 7
    assert all(not row.fits for row in error.value.report.rows)


def test_quality_and_preflight_reports_are_persistable(tmp_path: Path) -> None:
    sample = qmsum_sample("q-1")
    quality = summarize_qmsum_quality(
        (sample,),
        {"q-1": "session-q"},
        {"session-q": "The answer is one."},
    )
    preflight = preflight_qmsum(
        (sample,),
        {"q-1": "session-q"},
        {"q-1": {f"chunk_{index}": "partial" for index in range(6)}},
        tokenizer_factory=TokenizerHarness(),
    )

    write_quality_report(
        quality,
        rows_path=tmp_path / "quality_rows.jsonl",
        summary_path=tmp_path / "quality_summary.json",
    )
    write_preflight_report(
        preflight,
        rows_path=tmp_path / "preflight_rows.jsonl",
        summary_path=tmp_path / "preflight_summary.json",
    )

    quality_summary = json.loads(
        (tmp_path / "quality_summary.json").read_text(encoding="utf-8")
    )
    preflight_rows = (
        (tmp_path / "preflight_rows.jsonl").read_text(encoding="utf-8").splitlines()
    )
    assert quality_summary["sample_count"] == 1
    assert len(preflight_rows) == 7


def test_formal_preflight_builds_conservative_reports_for_all_samples(
    tmp_path: Path,
) -> None:
    qmsum_samples = tuple(qmsum_sample(f"q-{index:03d}") for index in range(60))
    mbpp_samples = tuple(mbpp_sample(str(index)) for index in range(11, 71))
    tokenizers = TokenizerHarness(model_adjustment=True)

    result = run_formal_preflight(
        qmsum_samples,
        mbpp_samples,
        tmp_path,
        tokenizer_factory=tokenizers,
    )

    assert result.output_dir == tmp_path / "prepared" / "preflight"
    assert result.qmsum.summary.sample_count == 60
    assert result.qmsum.summary.prompt_count == 420
    assert result.mbpp.summary.sample_count == 60
    assert result.mbpp.summary.prompt_count == 180
    construction = result.manifest.construction
    assert construction.bound_claim.endswith("not_mathematical_upper_bound")
    assert construction.uses_model_inference is False
    assert construction.uses_truncation is False
    assert len(construction.outputs) == 68
    repeated = [
        output
        for output in construction.outputs
        if output.construction_method == "repeated_text"
    ]
    for output in repeated:
        assert output.producer_tokens is not None
        assert output.max_new_tokens is not None
        assert output.producer_budget_utilization is not None
        assert output.producer_tokens <= output.max_new_tokens
        assert output.producer_budget_utilization >= 0.9
    qmsum_output = next(output for output in repeated if output.scenario == "qmsum")
    assert (
        qmsum_output.tokenizer_token_counts["/data/Models/Qwen/Qwen3-4B"]
        != (qmsum_output.tokenizer_token_counts["/data/Models/Qwen/Qwen3-8B"])
    )
    assert all(
        call[1]["truncation"] is False
        for backend in tokenizers.all_backends
        for call in backend.calls
    )
    expected_files = {
        "construction_manifest.json",
        "mbpp_rows.jsonl",
        "mbpp_summary.json",
        "qmsum_rows.jsonl",
        "qmsum_summary.json",
    }
    assert {path.name for path in result.output_dir.iterdir()} == expected_files
    qmsum_rows = result.output_dir / "qmsum_rows.jsonl"
    assert (
        result.manifest.artifact_sha256["qmsum_rows.jsonl"]
        == hashlib.sha256(qmsum_rows.read_bytes()).hexdigest()
    )
    assert len(result.manifest.construction_sha256) == 64


def test_formal_preflight_reuses_only_identical_prepared_artifacts(
    tmp_path: Path,
) -> None:
    qmsum_samples = tuple(qmsum_sample(f"q-{index:03d}") for index in range(60))
    mbpp_samples = tuple(mbpp_sample(str(index)) for index in range(11, 71))
    first = run_formal_preflight(
        qmsum_samples,
        mbpp_samples,
        tmp_path,
        tokenizer_factory=TokenizerHarness(),
    )
    second = run_formal_preflight(
        qmsum_samples,
        mbpp_samples,
        tmp_path,
        tokenizer_factory=TokenizerHarness(),
    )
    assert second.manifest == first.manifest

    summary_path = first.output_dir / "qmsum_summary.json"
    summary_path.write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="artifact content mismatch"):
        run_formal_preflight(
            qmsum_samples,
            mbpp_samples,
            tmp_path,
            tokenizer_factory=TokenizerHarness(),
        )


def test_formal_preflight_requires_fixed_sample_count(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="requires 60 samples"):
        run_formal_preflight(
            tuple(qmsum_sample(f"q-{index:03d}") for index in range(59)),
            tuple(mbpp_sample(str(index)) for index in range(11, 71)),
            tmp_path,
            tokenizer_factory=TokenizerHarness(),
        )


class FakeTokenizerBackend:
    def __init__(self, token_count: Callable[[str], int]) -> None:
        self.truncation_side = "right"
        self._token_count = token_count
        self.calls: list[tuple[str, dict[str, object]]] = []

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        **kwargs: object,
    ) -> str:
        body = "".join(
            f"<{message['role']}>{message['content']}" for message in messages
        )
        return f"{body}<assistant>"

    def encode(self, prompt: str, **kwargs: object) -> list[int]:
        self.calls.append((prompt, kwargs))
        return list(range(self._token_count(prompt)))

    def decode(self, token_ids: object, **kwargs: object) -> str:
        return ""


class TokenizerHarness:
    def __init__(
        self,
        token_count: Callable[[str], int] | None = None,
        *,
        model_adjustment: bool = False,
    ) -> None:
        self._token_count = token_count or (lambda prompt: len(prompt.split()))
        self._model_adjustment = model_adjustment
        self.backends: dict[str, FakeTokenizerBackend] = {}
        self.all_backends: list[FakeTokenizerBackend] = []

    def __call__(self, execution: ExecutionConfig) -> PromptTokenizer:
        def token_count(prompt: str) -> int:
            count = self._token_count(prompt)
            if self._model_adjustment and execution.model_path.endswith("Qwen3-8B"):
                return count + count // 20
            return count

        backend = FakeTokenizerBackend(token_count)
        self.backends[execution.model_path] = backend
        self.all_backends.append(backend)
        return PromptTokenizer(
            execution,
            tokenizer_factory=lambda _: backend,
        )


def qmsum_sample(sample_id: str) -> TaskSample:
    turns = [f"Speaker {index}: turn {index}" for index in range(6)]
    return TaskSample(
        sample_id=sample_id,
        source_dataset="qmsum",
        split="test",
        task_type="query_focused_summarization",
        input_text="\n\n".join(turns),
        gold_answer="The answer is one.",
        quality_metric="summary_quality",
        metadata={
            "query": "What is the answer?",
            "query_type": "specific",
            "turn_count": 6,
        },
    )


def mbpp_sample(sample_id: str) -> TaskSample:
    return TaskSample(
        sample_id=sample_id,
        source_dataset="mbpp_sanitized",
        split="test",
        task_type="python_function_generation",
        input_text="Write add_one.",
        test_list=["assert add_one(1) == 2"],
        quality_metric="pass_at_1",
        metadata={"task_id": int(sample_id)},
    )


def mbpp_result(
    sample_id: str,
    *,
    initial: MbppEvaluation,
    final: MbppEvaluation,
) -> MbppWorkflowResult:
    return MbppWorkflowResult(
        sample_id=sample_id,
        initial_code="initial code",
        final_code="final code",
        initial_eval=initial,
        final_eval=final,
    )


def evaluation(passed: bool, error_type: str | None = None) -> MbppEvaluation:
    return MbppEvaluation(
        passed=passed,
        passed_tests=int(passed),
        total_tests=1,
        error_type=error_type,
    )
