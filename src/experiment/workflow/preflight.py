from __future__ import annotations

import hashlib
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Literal

from langchain.agents import AgentState
from langchain_core.messages import AIMessage
from pydantic import BaseModel, ConfigDict, NonNegativeInt, PositiveInt

from dataset.schema import MbppEvaluation, TaskSample
from experiment.workflow.artifacts import (
    canonical_json,
    stable_digest,
    write_json_exclusive,
    write_jsonl_exclusive,
)
from experiment.workflow.mbpp import (
    MbppAttempt,
    build_mbpp_session_inputs,
    build_mbpp_workflow,
)
from experiment.workflow.qmsum import (
    QMSUM_CHUNK_COUNT,
    build_qmsum_session_inputs,
    build_qmsum_workflow,
    split_qmsum_session,
)
from experiment.workflow.quality import resolve_sample_sessions
from workflow.schema import AgentNodeConfig, ExecutionConfig, Workflow
from workflow.tokenizer import PromptTokenizer
from workflow.worker import (
    PromptTokenizerFactory,
    PromptTokenizerProtocol,
    build_prompt_context,
)

Scenario = Literal["qmsum", "mbpp"]
NodeOutputs = Mapping[str, Mapping[str, str]]
FORMAL_SAMPLE_COUNT = 60
FORMAL_PREFLIGHT_DIRECTORY = Path("prepared") / "preflight"
QMSUM_ROWS_FILE = "qmsum_rows.jsonl"
QMSUM_SUMMARY_FILE = "qmsum_summary.json"
MBPP_ROWS_FILE = "mbpp_rows.jsonl"
MBPP_SUMMARY_FILE = "mbpp_summary.json"
CONSTRUCTION_MANIFEST_FILE = "construction_manifest.json"


class PreflightModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class PromptPreflightRow(PreflightModel):
    scenario: Scenario
    position: NonNegativeInt
    sample_id: str
    session_id: str
    node_id: str
    model_name: str
    model_path: str
    prompt_chars: NonNegativeInt
    prompt_sha256: str
    input_tokens: NonNegativeInt
    max_new_tokens: NonNegativeInt
    total_tokens: NonNegativeInt
    max_model_len: NonNegativeInt
    remaining_tokens: int
    fits: bool


class PromptNodeSummary(PreflightModel):
    prompt_count: NonNegativeInt
    max_input_tokens: NonNegativeInt
    max_total_tokens: NonNegativeInt
    min_remaining_tokens: int
    overflow_count: NonNegativeInt


class PromptPreflightSummary(PreflightModel):
    scenario: Scenario
    sample_count: NonNegativeInt
    prompt_count: NonNegativeInt
    max_input_tokens: NonNegativeInt
    max_total_tokens: NonNegativeInt
    min_remaining_tokens: int
    overflow_count: NonNegativeInt
    valid: bool
    nodes: dict[str, PromptNodeSummary]


class PromptPreflightReport(PreflightModel):
    rows: tuple[PromptPreflightRow, ...]
    summary: PromptPreflightSummary


class ConstructedNodeOutput(PreflightModel):
    scenario: Scenario
    node_id: str
    sample_id: str | None = None
    construction_method: Literal["repeated_text", "mbpp_attempt_json"]
    producer_model_path: str | None = None
    max_new_tokens: PositiveInt | None = None
    producer_tokens: NonNegativeInt | None = None
    producer_budget_utilization: float | None = None
    prefix: str | None = None
    repeated_unit: str | None = None
    suffix: str | None = None
    repetitions: NonNegativeInt | None = None
    dependencies: tuple[str, ...] = ()
    text_chars: NonNegativeInt
    text_sha256: str
    tokenizer_token_counts: dict[str, NonNegativeInt]


class FormalPreflightConstruction(PreflightModel):
    mode: Literal["conservative_representative"] = "conservative_representative"
    bound_claim: Literal["conservative_representative_not_mathematical_upper_bound"] = (
        "conservative_representative_not_mathematical_upper_bound"
    )
    uses_model_inference: Literal[False] = False
    uses_truncation: Literal[False] = False
    qmsum_sample_ids: tuple[str, ...]
    mbpp_sample_ids: tuple[str, ...]
    outputs: tuple[ConstructedNodeOutput, ...]


class FormalPreflightManifest(PreflightModel):
    version: PositiveInt = 1
    construction: FormalPreflightConstruction
    construction_sha256: str
    artifact_sha256: dict[str, str]


class FormalPreflightResult(PreflightModel):
    output_dir: Path
    qmsum: PromptPreflightReport
    mbpp: PromptPreflightReport
    manifest: FormalPreflightManifest


class PromptLimitExceededError(ValueError):
    def __init__(self, report: PromptPreflightReport) -> None:
        self.report = report
        first = next(row for row in report.rows if not row.fits)
        super().__init__(
            f"{report.summary.overflow_count} prompts exceed max_model_len; "
            f"first={first.session_id}/{first.node_id} "
            f"({first.total_tokens}>{first.max_model_len})"
        )


def preflight_qmsum(
    samples: Sequence[TaskSample],
    sample_sessions: Mapping[str, str],
    node_outputs: NodeOutputs,
    *,
    tokenizer_factory: PromptTokenizerFactory = PromptTokenizer,
) -> PromptPreflightReport:
    required_nodes = tuple(f"chunk_{index}" for index in range(QMSUM_CHUNK_COUNT))
    outputs = _resolve_node_outputs(samples, node_outputs, required_nodes)
    workflow = build_qmsum_workflow()
    tokenizers: dict[str, PromptTokenizerProtocol] = {}
    rows: list[PromptPreflightRow] = []
    for position, (sample, session_id) in enumerate(
        resolve_sample_sessions(samples, sample_sessions)
    ):
        session_inputs = build_qmsum_session_inputs(sample)
        split_states = split_qmsum_session(session_inputs, {}, {})
        sample_outputs = outputs[sample.sample_id]
        merge_states: dict[str, AgentState] = {}
        for node_name in required_nodes:
            node = _agent_node(workflow, node_name)
            rows.append(
                _prompt_row(
                    "qmsum",
                    position,
                    sample,
                    session_id,
                    node,
                    build_prompt_context(
                        session_inputs,
                        {"split": split_states[node_name]},
                    ),
                    tokenizer_factory,
                    tokenizers,
                )
            )
            merge_states[node_name] = _text_state(sample_outputs[node_name])
        merge = _agent_node(workflow, "merge")
        rows.append(
            _prompt_row(
                "qmsum",
                position,
                sample,
                session_id,
                merge,
                build_prompt_context(session_inputs, merge_states),
                tokenizer_factory,
                tokenizers,
            )
        )
    return _validated_report("qmsum", len(samples), rows)


def preflight_mbpp(
    samples: Sequence[TaskSample],
    sample_sessions: Mapping[str, str],
    node_outputs: NodeOutputs,
    *,
    tokenizer_factory: PromptTokenizerFactory = PromptTokenizer,
) -> PromptPreflightReport:
    required_nodes = ("coder", "tester", "reviewer")
    outputs = _resolve_node_outputs(samples, node_outputs, required_nodes)
    workflow = build_mbpp_workflow()
    tokenizers: dict[str, PromptTokenizerProtocol] = {}
    rows: list[PromptPreflightRow] = []
    for position, (sample, session_id) in enumerate(
        resolve_sample_sessions(samples, sample_sessions)
    ):
        session_inputs = build_mbpp_session_inputs(sample)
        sample_outputs = outputs[sample.sample_id]
        coder_output = sample_outputs["coder"]
        attempt = MbppAttempt.model_validate_json(sample_outputs["tester"])
        if attempt.code != coder_output:
            raise ValueError(
                f"MBPP tester output does not match coder output: {sample.sample_id}"
            )
        tester_state = _text_state(sample_outputs["tester"])
        reviewer_state = _text_state(sample_outputs["reviewer"])
        contexts = {
            "coder": build_prompt_context(session_inputs, {}),
            "reviewer": build_prompt_context(
                session_inputs,
                {"tester": tester_state},
            ),
            "repair": build_prompt_context(
                session_inputs,
                {"tester": tester_state, "reviewer": reviewer_state},
            ),
        }
        rows.extend(
            _prompt_row(
                "mbpp",
                position,
                sample,
                session_id,
                _agent_node(workflow, node_name),
                contexts[node_name],
                tokenizer_factory,
                tokenizers,
            )
            for node_name in ("coder", "reviewer", "repair")
        )
    return _validated_report("mbpp", len(samples), rows)


def run_formal_preflight(
    qmsum_samples: Sequence[TaskSample],
    mbpp_samples: Sequence[TaskSample],
    output_root: Path,
    *,
    tokenizer_factory: PromptTokenizerFactory = PromptTokenizer,
) -> FormalPreflightResult:
    _require_formal_sample_count("QMSum", qmsum_samples)
    _require_formal_sample_count("MBPP", mbpp_samples)
    tokenizer_cache: dict[str, PromptTokenizerProtocol] = {}

    def shared_tokenizer_factory(
        execution: ExecutionConfig,
    ) -> PromptTokenizerProtocol:
        return _tokenizer(execution, tokenizer_factory, tokenizer_cache)

    qmsum_outputs, qmsum_records = _construct_qmsum_outputs(
        qmsum_samples,
        shared_tokenizer_factory,
    )
    mbpp_outputs, mbpp_records = _construct_mbpp_outputs(
        mbpp_samples,
        shared_tokenizer_factory,
    )
    qmsum_report = preflight_qmsum(
        qmsum_samples,
        _formal_sessions("qmsum", qmsum_samples),
        qmsum_outputs,
        tokenizer_factory=shared_tokenizer_factory,
    )
    mbpp_report = preflight_mbpp(
        mbpp_samples,
        _formal_sessions("mbpp", mbpp_samples),
        mbpp_outputs,
        tokenizer_factory=shared_tokenizer_factory,
    )
    construction = FormalPreflightConstruction(
        qmsum_sample_ids=tuple(sample.sample_id for sample in qmsum_samples),
        mbpp_sample_ids=tuple(sample.sample_id for sample in mbpp_samples),
        outputs=(*qmsum_records, *mbpp_records),
    )
    artifacts = _formal_report_artifacts(qmsum_report, mbpp_report)
    manifest = FormalPreflightManifest(
        construction=construction,
        construction_sha256=stable_digest(construction),
        artifact_sha256={
            name: hashlib.sha256(content.encode()).hexdigest()
            for name, content in artifacts.items()
        },
    )
    artifacts[CONSTRUCTION_MANIFEST_FILE] = f"{canonical_json(manifest)}\n"
    output_dir = output_root / FORMAL_PREFLIGHT_DIRECTORY
    _write_or_validate_artifacts(output_dir, artifacts)
    return FormalPreflightResult(
        output_dir=output_dir,
        qmsum=qmsum_report,
        mbpp=mbpp_report,
        manifest=manifest,
    )


def write_preflight_report(
    report: PromptPreflightReport, *, rows_path: Path, summary_path: Path
) -> None:
    write_jsonl_exclusive(
        rows_path, (row.model_dump(mode="json") for row in report.rows)
    )
    write_json_exclusive(summary_path, report.summary.model_dump(mode="json"))


def _construct_qmsum_outputs(
    samples: Sequence[TaskSample],
    tokenizer_factory: PromptTokenizerFactory,
) -> tuple[dict[str, dict[str, str]], tuple[ConstructedNodeOutput, ...]]:
    workflow = build_qmsum_workflow()
    chunk = _agent_node(workflow, "chunk_0")
    merge = _agent_node(workflow, "merge")
    text, repetitions = _maximize_repeated_text(
        prefix="",
        repeated_unit="detail ",
        suffix="",
        budget=chunk.execution.max_new_tokens,
        execution=chunk.execution,
        tokenizer_factory=tokenizer_factory,
    )
    counts = _raw_token_counts(
        text,
        (chunk.execution, merge.execution),
        tokenizer_factory,
    )
    _validate_producer_count(chunk, counts)
    node_names = tuple(f"chunk_{index}" for index in range(QMSUM_CHUNK_COUNT))
    outputs = {
        sample.sample_id: {node_name: text for node_name in node_names}
        for sample in samples
    }
    records = tuple(
        _repeated_output_record(
            "qmsum",
            node_name,
            chunk,
            text,
            repetitions,
            counts,
            prefix="",
            repeated_unit="detail ",
            suffix="",
        )
        for node_name in node_names
    )
    return outputs, records


def _construct_mbpp_outputs(
    samples: Sequence[TaskSample],
    tokenizer_factory: PromptTokenizerFactory,
) -> tuple[dict[str, dict[str, str]], tuple[ConstructedNodeOutput, ...]]:
    workflow = build_mbpp_workflow()
    coder = _agent_node(workflow, "coder")
    reviewer = _agent_node(workflow, "reviewer")
    repair = _agent_node(workflow, "repair")
    code, code_repetitions = _maximize_repeated_text(
        prefix="```python\n",
        repeated_unit="pass\n",
        suffix="```",
        budget=coder.execution.max_new_tokens,
        execution=coder.execution,
        tokenizer_factory=tokenizer_factory,
    )
    code_counts = _raw_token_counts(
        code,
        (coder.execution, reviewer.execution, repair.execution),
        tokenizer_factory,
    )
    _validate_producer_count(coder, code_counts)
    review, review_repetitions = _maximize_repeated_text(
        prefix="",
        repeated_unit="issue ",
        suffix="",
        budget=reviewer.execution.max_new_tokens,
        execution=reviewer.execution,
        tokenizer_factory=tokenizer_factory,
    )
    review_counts = _raw_token_counts(
        review,
        (reviewer.execution, repair.execution),
        tokenizer_factory,
    )
    _validate_producer_count(reviewer, review_counts)
    records = [
        _repeated_output_record(
            "mbpp",
            "coder",
            coder,
            code,
            code_repetitions,
            code_counts,
            prefix="```python\n",
            repeated_unit="pass\n",
            suffix="```",
        ),
        _repeated_output_record(
            "mbpp",
            "reviewer",
            reviewer,
            review,
            review_repetitions,
            review_counts,
            prefix="",
            repeated_unit="issue ",
            suffix="",
        ),
    ]
    outputs: dict[str, dict[str, str]] = {}
    for sample in samples:
        attempt = MbppAttempt(
            code=code,
            initial_eval=MbppEvaluation(
                passed=False,
                passed_tests=0,
                total_tests=len(sample.test_list),
                error_type="assertion_error",
                error_message="representative assertion failure",
            ),
        )
        tester = canonical_json(attempt)
        tester_counts = _raw_token_counts(
            tester,
            (reviewer.execution, repair.execution),
            tokenizer_factory,
        )
        outputs[sample.sample_id] = {
            "coder": code,
            "tester": tester,
            "reviewer": review,
        }
        records.append(
            ConstructedNodeOutput(
                scenario="mbpp",
                node_id="tester",
                sample_id=sample.sample_id,
                construction_method="mbpp_attempt_json",
                dependencies=("coder",),
                text_chars=len(tester),
                text_sha256=hashlib.sha256(tester.encode()).hexdigest(),
                tokenizer_token_counts=tester_counts,
            )
        )
    return outputs, tuple(records)


def _maximize_repeated_text(
    *,
    prefix: str,
    repeated_unit: str,
    suffix: str,
    budget: int,
    execution: ExecutionConfig,
    tokenizer_factory: PromptTokenizerFactory,
) -> tuple[str, int]:
    if not repeated_unit:
        raise ValueError("repeated_unit must not be empty")

    def candidate(repetitions: int) -> str:
        return f"{prefix}{repeated_unit * repetitions}{suffix}"

    low = 0
    high = 1
    while _raw_token_count(candidate(high), execution, tokenizer_factory) <= budget:
        low = high
        high *= 2
        if high > 1_048_576:
            raise RuntimeError("token count did not exceed the construction budget")
    while low + 1 < high:
        middle = (low + high) // 2
        if _raw_token_count(candidate(middle), execution, tokenizer_factory) <= budget:
            low = middle
        else:
            high = middle
    text = candidate(low)
    token_count = _raw_token_count(text, execution, tokenizer_factory)
    if token_count > budget:
        raise RuntimeError(
            f"constructed output exceeds its producer budget: {token_count}>{budget}"
        )
    if token_count == 0 or token_count * 10 < budget * 9:
        raise RuntimeError(
            f"constructed output uses only {token_count}/{budget} producer tokens"
        )
    return text, low


def _raw_token_counts(
    text: str,
    executions: Sequence[ExecutionConfig],
    tokenizer_factory: PromptTokenizerFactory,
) -> dict[str, int]:
    counts: dict[str, int] = {}
    for execution in executions:
        if execution.model_path not in counts:
            counts[execution.model_path] = _raw_token_count(
                text,
                execution,
                tokenizer_factory,
            )
    return counts


def _raw_token_count(
    text: str,
    execution: ExecutionConfig,
    tokenizer_factory: PromptTokenizerFactory,
) -> int:
    raw_execution = execution.model_copy(
        update={"use_chat_template": False, "enable_thinking": False}
    )
    return tokenizer_factory(raw_execution).build_prompt(text).input_tokens


def _validate_producer_count(
    producer: AgentNodeConfig,
    counts: Mapping[str, int],
) -> None:
    token_count = counts[producer.execution.model_path]
    budget = producer.execution.max_new_tokens
    if token_count > budget:
        raise ValueError(
            f"constructed {producer.name} output exceeds its token budget: "
            f"{token_count}>{budget}"
        )


def _repeated_output_record(
    scenario: Scenario,
    node_id: str,
    producer: AgentNodeConfig,
    text: str,
    repetitions: int,
    counts: Mapping[str, int],
    *,
    prefix: str,
    repeated_unit: str,
    suffix: str,
) -> ConstructedNodeOutput:
    producer_tokens = counts[producer.execution.model_path]
    budget = producer.execution.max_new_tokens
    return ConstructedNodeOutput(
        scenario=scenario,
        node_id=node_id,
        construction_method="repeated_text",
        producer_model_path=producer.execution.model_path,
        max_new_tokens=budget,
        producer_tokens=producer_tokens,
        producer_budget_utilization=producer_tokens / budget,
        prefix=prefix,
        repeated_unit=repeated_unit,
        suffix=suffix,
        repetitions=repetitions,
        text_chars=len(text),
        text_sha256=hashlib.sha256(text.encode()).hexdigest(),
        tokenizer_token_counts=dict(counts),
    )


def _formal_sessions(
    scenario: Scenario,
    samples: Sequence[TaskSample],
) -> dict[str, str]:
    return {
        sample.sample_id: f"preflight-{scenario}-{position:03d}"
        for position, sample in enumerate(samples)
    }


def _require_formal_sample_count(
    label: str,
    samples: Sequence[TaskSample],
) -> None:
    if len(samples) != FORMAL_SAMPLE_COUNT:
        raise ValueError(
            f"formal {label} preflight requires {FORMAL_SAMPLE_COUNT} samples, "
            f"got {len(samples)}"
        )
    sample_ids = [sample.sample_id for sample in samples]
    if len(set(sample_ids)) != len(sample_ids):
        raise ValueError(f"formal {label} samples contain duplicate sample IDs")


def _formal_report_artifacts(
    qmsum: PromptPreflightReport,
    mbpp: PromptPreflightReport,
) -> dict[str, str]:
    return {
        QMSUM_ROWS_FILE: _jsonl_content(qmsum.rows),
        QMSUM_SUMMARY_FILE: f"{canonical_json(qmsum.summary)}\n",
        MBPP_ROWS_FILE: _jsonl_content(mbpp.rows),
        MBPP_SUMMARY_FILE: f"{canonical_json(mbpp.summary)}\n",
    }


def _jsonl_content(rows: Sequence[BaseModel]) -> str:
    return "".join(f"{canonical_json(row)}\n" for row in rows)


def _write_or_validate_artifacts(
    output_dir: Path,
    artifacts: Mapping[str, str],
) -> None:
    expected_names = set(artifacts)
    if output_dir.exists():
        if not output_dir.is_dir():
            raise ValueError(f"preflight output is not a directory: {output_dir}")
        actual_names = {path.name for path in output_dir.iterdir()}
        if actual_names != expected_names:
            raise ValueError(
                _set_mismatch("preflight artifacts", expected_names, actual_names)
            )
        for name, content in artifacts.items():
            path = output_dir / name
            if not path.is_file() or path.read_text(encoding="utf-8") != content:
                raise ValueError(f"preflight artifact content mismatch: {path}")
        return

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir()
    written: list[Path] = []
    try:
        for name, content in artifacts.items():
            path = output_dir / name
            with path.open("x", encoding="utf-8") as file:
                written.append(path)
                file.write(content)
                file.flush()
    except OSError:
        for path in written:
            path.unlink(missing_ok=True)
        output_dir.rmdir()
        raise


def _prompt_row(
    scenario: Scenario,
    position: int,
    sample: TaskSample,
    session_id: str,
    node: AgentNodeConfig,
    context: Mapping[str, object],
    tokenizer_factory: PromptTokenizerFactory,
    tokenizers: dict[str, PromptTokenizerProtocol],
) -> PromptPreflightRow:
    prompt = node.prompt_template.format_map(context)
    tokenizer = _tokenizer(node.execution, tokenizer_factory, tokenizers)
    encoding = tokenizer.build_prompt(prompt, system_prompt=node.system_prompt)
    max_new_tokens = node.execution.max_new_tokens
    max_model_len = node.execution.serving.max_model_len
    total_tokens = encoding.input_tokens + max_new_tokens
    return PromptPreflightRow(
        scenario=scenario,
        position=position,
        sample_id=sample.sample_id,
        session_id=session_id,
        node_id=node.name,
        model_name=node.model.name,
        model_path=node.execution.model_path,
        prompt_chars=len(encoding.text),
        prompt_sha256=hashlib.sha256(encoding.text.encode()).hexdigest(),
        input_tokens=encoding.input_tokens,
        max_new_tokens=max_new_tokens,
        total_tokens=total_tokens,
        max_model_len=max_model_len,
        remaining_tokens=max_model_len - total_tokens,
        fits=total_tokens <= max_model_len,
    )


def _tokenizer(
    execution: ExecutionConfig,
    tokenizer_factory: PromptTokenizerFactory,
    tokenizers: dict[str, PromptTokenizerProtocol],
) -> PromptTokenizerProtocol:
    key = execution.model_dump_json()
    tokenizer = tokenizers.get(key)
    if tokenizer is None:
        tokenizer = tokenizer_factory(execution)
        tokenizers[key] = tokenizer
    return tokenizer


def _agent_node(workflow: Workflow, node_name: str) -> AgentNodeConfig:
    node = workflow.node_map()[node_name]
    if not isinstance(node, AgentNodeConfig):
        raise TypeError(f"workflow node is not an agent: {node_name}")
    return node


def _resolve_node_outputs(
    samples: Sequence[TaskSample],
    node_outputs: NodeOutputs,
    required_nodes: Sequence[str],
) -> dict[str, dict[str, str]]:
    sample_ids = {sample.sample_id for sample in samples}
    if set(node_outputs) != sample_ids:
        raise ValueError(_set_mismatch("node_outputs", sample_ids, set(node_outputs)))
    required = set(required_nodes)
    resolved: dict[str, dict[str, str]] = {}
    for sample in samples:
        outputs = node_outputs[sample.sample_id]
        if set(outputs) != required:
            raise ValueError(
                _set_mismatch(
                    f"node_outputs[{sample.sample_id!r}]",
                    required,
                    set(outputs),
                )
            )
        if any(not isinstance(output, str) for output in outputs.values()):
            raise TypeError(f"node outputs must be strings: {sample.sample_id}")
        resolved[sample.sample_id] = dict(outputs)
    return resolved


def _validated_report(
    scenario: Scenario,
    sample_count: int,
    rows: Sequence[PromptPreflightRow],
) -> PromptPreflightReport:
    report = PromptPreflightReport(
        rows=tuple(rows),
        summary=_summarize(scenario, sample_count, rows),
    )
    if not report.summary.valid:
        raise PromptLimitExceededError(report)
    return report


def _summarize(
    scenario: Scenario,
    sample_count: int,
    rows: Sequence[PromptPreflightRow],
) -> PromptPreflightSummary:
    grouped: defaultdict[str, list[PromptPreflightRow]] = defaultdict(list)
    for row in rows:
        grouped[row.node_id].append(row)
    overflow_count = sum(not row.fits for row in rows)
    return PromptPreflightSummary(
        scenario=scenario,
        sample_count=sample_count,
        prompt_count=len(rows),
        max_input_tokens=max((row.input_tokens for row in rows), default=0),
        max_total_tokens=max((row.total_tokens for row in rows), default=0),
        min_remaining_tokens=min((row.remaining_tokens for row in rows), default=0),
        overflow_count=overflow_count,
        valid=overflow_count == 0,
        nodes={name: _summarize_node(node_rows) for name, node_rows in grouped.items()},
    )


def _summarize_node(rows: Sequence[PromptPreflightRow]) -> PromptNodeSummary:
    return PromptNodeSummary(
        prompt_count=len(rows),
        max_input_tokens=max(row.input_tokens for row in rows),
        max_total_tokens=max(row.total_tokens for row in rows),
        min_remaining_tokens=min(row.remaining_tokens for row in rows),
        overflow_count=sum(not row.fits for row in rows),
    )


def _text_state(output: str) -> AgentState:
    return AgentState(messages=[AIMessage(content=output)])


def _set_mismatch(label: str, expected: set[str], actual: set[str]) -> str:
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    return f"{label} mismatch: missing={missing}, extra={extra}"
