from __future__ import annotations

import json
from collections.abc import Sequence
from io import StringIO
from pathlib import Path
from typing import Any

import yaml
import pytest

from dataset.schema import TaskSample
from workflow.execution import (
    GenerationResult,
    build_input_chunks,
    infer_workflow_chunk_count,
    load_resume_sample_records,
    load_sample_ids,
    main,
    run_workflow_evaluation,
    select_samples_by_id,
    select_samples,
)
from workflow.schema import Workflow, WorkflowNodeConfig


QWEN_PARAMETERS = {
    "hidden_size": 1536,
    "intermediate_size": 5120,
    "num_hidden_layers": 28,
    "num_attention_heads": 24,
    "num_key_value_heads": 4,
    "vocab_size": 151936,
    "max_position_embeddings": 40960,
}


class FakeBackend:
    def __init__(self, outputs: dict[str, list[str]]) -> None:
        self.outputs = {name: list(values) for name, values in outputs.items()}
        self.preloaded: list[str] = []
        self.prompts: list[tuple[str, str]] = []

    def preload(self, nodes: Sequence[WorkflowNodeConfig]) -> None:
        self.preloaded = [node.name for node in nodes]

    def generate(self, node: WorkflowNodeConfig, prompt: str) -> GenerationResult:
        self.prompts.append((node.name, prompt))
        values = self.outputs[node.name]
        if not values:
            raise AssertionError(f"no fake output left for {node.name}")
        output = values.pop(0)
        return GenerationResult(
            text=output,
            input_token_count=len(prompt.split()),
            output_token_count=len(output.split()),
            total_token_count=len(prompt.split()) + len(output.split()),
        )


def build_model_node(name: str, *, node_type: str = "tool") -> dict[str, Any]:
    return {
        "name": name,
        "type": node_type,
        "task": "text_generation",
        "prompt_template": "Question: {input_text}\nPrevious: {previous_output}",
        "model": {"name": "qwen3", "parameters": dict(QWEN_PARAMETERS)},
        "runtime": {
            "batch_size": 1,
            "sequence_length": 128,
            "decode_max_output_length": 32,
            "phase": "decode",
        },
        "execution": {
            "model_path": f"/models/{name}",
            "devices": ["cuda:0"],
        },
    }


def build_repair_workflow() -> Workflow:
    repair_node = build_model_node("repair")
    repair_node["prompt_template"] = (
        "Question: {input_text}\nPrevious: {previous_output}\n"
        "Evaluator: {evaluator_results_json}"
    )
    return Workflow.model_validate(
        {
            "nodes": [
                {"name": "input", "type": "input"},
                build_model_node("solver"),
                {
                    "name": "judge",
                    "type": "evaluator",
                    "task": "gsm8k_numeric_exact_match",
                },
                repair_node,
                {
                    "name": "judge_after_repair",
                    "type": "evaluator",
                    "task": "gsm8k_numeric_exact_match",
                },
                {"name": "output", "type": "output"},
            ],
            "edges": [
                {"source": "input", "target": "solver", "attributes": {}},
                {"source": "solver", "target": "judge", "attributes": {}},
                {
                    "source": "judge",
                    "target": "output",
                    "attributes": {"condition": "passed"},
                },
                {
                    "source": "judge",
                    "target": "repair",
                    "attributes": {"condition": "failed"},
                },
                {"source": "repair", "target": "judge_after_repair", "attributes": {}},
                {
                    "source": "judge_after_repair",
                    "target": "output",
                    "attributes": {},
                },
            ],
        }
    )


def build_direct_workflow() -> Workflow:
    return Workflow.model_validate(
        {
            "nodes": [
                {"name": "input", "type": "input"},
                build_model_node("solver"),
                {
                    "name": "judge",
                    "type": "evaluator",
                    "task": "gsm8k_numeric_exact_match",
                },
                {"name": "output", "type": "output"},
            ],
            "edges": [
                {"source": "input", "target": "solver", "attributes": {}},
                {"source": "solver", "target": "judge", "attributes": {}},
                {"source": "judge", "target": "output", "attributes": {}},
            ],
        }
    )


def build_react_workflow() -> Workflow:
    controller = build_model_node("controller", node_type="agent")
    controller["prompt_template"] = "Handle sample {sample_id}: {input_text}"
    return Workflow.model_validate(
        {
            "nodes": [
                {"name": "input", "type": "input"},
                controller,
                build_model_node("solver"),
                {
                    "name": "judge",
                    "type": "evaluator",
                    "task": "gsm8k_numeric_exact_match",
                },
                {"name": "output", "type": "output"},
            ],
            "edges": [
                {"source": "input", "target": "controller", "attributes": {}},
                {"source": "controller", "target": "output", "attributes": {}},
            ],
        }
    )


def build_gsm8k_sample(sample_id: str = "test_00000") -> TaskSample:
    return TaskSample(
        sample_id=sample_id,
        source_dataset="gsm8k",
        split="test",
        task_type="math_word_problem",
        input_text="What is 1 + 1?",
        quality_metric="numeric_exact_match",
        gold_answer="2",
    )


def build_summary_sample(sample_id: str = "test_00000") -> TaskSample:
    return TaskSample(
        sample_id=sample_id,
        source_dataset="multi_news",
        split="test",
        task_type="multi_document_summarization",
        input_text=(
            "Document 1:\nAlpha article.\n\n"
            "Document 2:\nBeta article.\n\n"
            "Document 3:\nGamma article."
        ),
        quality_metric="summary_quality",
        gold_answer="alpha beta gamma",
    )


def build_six_document_summary_sample(sample_id: str = "test_00000") -> TaskSample:
    return TaskSample(
        sample_id=sample_id,
        source_dataset="multi_news",
        split="test",
        task_type="multi_document_summarization",
        input_text="\n\n".join(
            f"Document {index}:\nArticle {index}." for index in range(1, 7)
        ),
        quality_metric="summary_quality",
        gold_answer=" ".join(f"article {index}" for index in range(1, 7)),
    )


def build_summary_model_node(name: str, prompt_template: str) -> dict[str, Any]:
    node = build_model_node(name)
    node["prompt_template"] = prompt_template
    return node


def build_summary_direct_workflow(evaluator_task: str = "summary_rouge") -> Workflow:
    return Workflow.model_validate(
        {
            "nodes": [
                {"name": "input", "type": "input"},
                build_summary_model_node(
                    "summarizer",
                    "Summarize:\n{input_text}\nQuery: {query}",
                ),
                {"name": "judge", "type": "evaluator", "task": evaluator_task},
                {"name": "output", "type": "output"},
            ],
            "edges": [
                {"source": "input", "target": "summarizer", "attributes": {}},
                {"source": "summarizer", "target": "judge", "attributes": {}},
                {"source": "judge", "target": "output", "attributes": {}},
            ],
        }
    )


def build_parallel_summary_workflow() -> Workflow:
    return Workflow.model_validate(
        {
            "nodes": [
                {"name": "input", "type": "input"},
                build_summary_model_node("chunk_0", "Summarize:\n{chunk_0_text}"),
                build_summary_model_node("chunk_1", "Summarize:\n{chunk_1_text}"),
                build_summary_model_node("chunk_2", "Summarize:\n{chunk_2_text}"),
                build_summary_model_node(
                    "merge",
                    "Merge partial summaries:\n{node_outputs_json}",
                ),
                {"name": "judge", "type": "evaluator", "task": "summary_rouge"},
                {"name": "output", "type": "output"},
            ],
            "edges": [
                {"source": "input", "target": "chunk_0", "attributes": {}},
                {"source": "input", "target": "chunk_1", "attributes": {}},
                {"source": "input", "target": "chunk_2", "attributes": {}},
                {"source": "chunk_0", "target": "merge", "attributes": {}},
                {"source": "chunk_1", "target": "merge", "attributes": {}},
                {"source": "chunk_2", "target": "merge", "attributes": {}},
                {"source": "merge", "target": "judge", "attributes": {}},
                {"source": "judge", "target": "output", "attributes": {}},
            ],
        }
    )


def build_parallel_six_way_summary_workflow() -> Workflow:
    return Workflow.model_validate(
        {
            "nodes": [
                {"name": "input", "type": "input"},
                *[
                    build_summary_model_node(
                        f"chunk_{index}",
                        f"Summarize:\n{{chunk_{index}_text}}",
                    )
                    for index in range(6)
                ],
                build_summary_model_node(
                    "merge",
                    "Merge partial summaries:\n{node_outputs_json}",
                ),
                {"name": "judge", "type": "evaluator", "task": "summary_rouge"},
                {"name": "output", "type": "output"},
            ],
            "edges": [
                *[
                    {"source": "input", "target": f"chunk_{index}", "attributes": {}}
                    for index in range(6)
                ],
                *[
                    {"source": f"chunk_{index}", "target": "merge", "attributes": {}}
                    for index in range(6)
                ],
                {"source": "merge", "target": "judge", "attributes": {}},
                {"source": "judge", "target": "output", "attributes": {}},
            ],
        }
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
            "coverage": 4,
            "relevance": 5,
            "coherence": 4,
            "faithfulness": 5,
            "reason": "good enough",
        }


def parse_jsonl(value: str) -> list[dict[str, Any]]:
    return [json.loads(line) for line in value.splitlines()]


def test_select_samples_uses_seeded_random_subset() -> None:
    samples = [build_gsm8k_sample(f"test_{index:05d}") for index in range(5)]

    first = select_samples(samples, limit=3, seed=42)
    second = select_samples(samples, limit=3, seed=42)

    assert [sample.sample_id for sample in first] == [
        sample.sample_id for sample in second
    ]
    assert len(first) == 3


def test_select_samples_by_id_preserves_file_order(tmp_path: Path) -> None:
    samples = [build_gsm8k_sample(f"test_{index:05d}") for index in range(3)]
    sample_ids_path = tmp_path / "sample_ids.jsonl"
    sample_ids_path.write_text(
        "\n".join(
            [
                json.dumps({"sample_id": "test_00002"}),
                json.dumps("test_00000"),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    selected = select_samples_by_id(samples, sample_ids_path)

    assert [sample.sample_id for sample in selected] == [
        "test_00002",
        "test_00000",
    ]


def test_select_samples_by_id_rejects_missing_id(tmp_path: Path) -> None:
    samples = [build_gsm8k_sample("test_00000")]
    sample_ids_path = tmp_path / "sample_ids.jsonl"
    sample_ids_path.write_text(
        json.dumps({"sample_id": "missing"}) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="sample ids do not exist: missing"):
        select_samples_by_id(samples, sample_ids_path)


def test_build_input_chunks_preserves_contiguous_order() -> None:
    input_text = "\n\n".join(f"part {index}" for index in range(6))

    chunks = build_input_chunks(input_text, chunk_count=3)

    assert chunks == [
        "part 0\n\npart 1",
        "part 2\n\npart 3",
        "part 4\n\npart 5",
    ]


def test_infer_workflow_chunk_count_reads_prompt_variables() -> None:
    workflow = build_parallel_six_way_summary_workflow()

    assert infer_workflow_chunk_count(workflow) == 6


def test_dag_mode_skips_repair_when_evaluator_passes() -> None:
    stdout = StringIO()
    backend = FakeBackend({"solver": ["#### 2"], "repair": ["#### 2"]})

    run_workflow_evaluation(
        workflow=build_repair_workflow(),
        dataset="gsm8k",
        samples=[build_gsm8k_sample()],
        mode="dag",
        backend=backend,
        stdout=stdout,
    )

    rows = parse_jsonl(stdout.getvalue())
    assert rows[0]["passed"]
    assert "repair" not in rows[0]["node_outputs"]
    assert rows[1]["record_type"] == "summary"
    assert rows[1]["accuracy"] == 1.0
    assert rows[1]["total_output_tokens"] > 0


def test_dag_mode_runs_repair_when_evaluator_fails() -> None:
    stdout = StringIO()
    backend = FakeBackend({"solver": ["#### 1"], "repair": ["#### 2"]})

    run_workflow_evaluation(
        workflow=build_repair_workflow(),
        dataset="gsm8k",
        samples=[build_gsm8k_sample()],
        mode="dag",
        backend=backend,
        stdout=stdout,
    )

    rows = parse_jsonl(stdout.getvalue())
    assert rows[0]["passed"]
    assert rows[0]["node_outputs"]["repair"] == "#### 2"
    assert rows[0]["evaluator_results"]["judge"]["exact_match"] is False
    assert rows[0]["evaluator_results"]["judge_after_repair"]["exact_match"] is True
    assert '"judge"' in backend.prompts[-1][1]


def test_react_mode_executes_json_actions() -> None:
    stdout = StringIO()
    backend = FakeBackend(
        {
            "controller": [
                '{"action": "solver", "input": "solve"}',
                '{"action": "judge", "input": "#### 2"}',
                '{"final": "#### 2"}',
            ],
            "solver": ["#### 2"],
        }
    )

    run_workflow_evaluation(
        workflow=build_react_workflow(),
        dataset="gsm8k",
        samples=[build_gsm8k_sample()],
        mode="react",
        backend=backend,
        stdout=stdout,
    )

    rows = parse_jsonl(stdout.getvalue())
    assert rows[0]["passed"]
    assert rows[0]["final_output"] == "#### 2"
    assert rows[0]["node_outputs"]["solver"] == "#### 2"


def test_parallel_mode_runs_fan_out_and_merge_for_summary() -> None:
    stdout = StringIO()
    backend = FakeBackend(
        {
            "chunk_0": ["alpha"],
            "chunk_1": ["beta"],
            "chunk_2": ["gamma"],
            "merge": ["alpha beta gamma"],
        }
    )

    run_workflow_evaluation(
        workflow=build_parallel_summary_workflow(),
        dataset="multi_news",
        samples=[build_summary_sample()],
        mode="parallel",
        backend=backend,
        stdout=stdout,
    )

    rows = parse_jsonl(stdout.getvalue())
    prompts = dict(backend.prompts)
    assert rows[0]["passed"]
    assert rows[0]["mode"] == "parallel"
    assert rows[0]["node_outputs"]["merge"] == "alpha beta gamma"
    assert "Document 1:" in prompts["chunk_0"]
    assert "Document 2:" in prompts["chunk_1"]
    assert "Document 3:" in prompts["chunk_2"]
    assert '"chunk_0": "alpha"' in prompts["merge"]
    assert rows[1]["model_call_count"] == 4
    assert rows[1]["avg_rouge1"] > 0.0


def test_parallel_mode_renders_six_chunk_prompts() -> None:
    stdout = StringIO()
    backend = FakeBackend(
        {
            **{f"chunk_{index}": [f"article {index + 1}"] for index in range(6)},
            "merge": ["article 1 article 2 article 3 article 4 article 5 article 6"],
        }
    )

    run_workflow_evaluation(
        workflow=build_parallel_six_way_summary_workflow(),
        dataset="multi_news",
        samples=[build_six_document_summary_sample()],
        mode="parallel",
        backend=backend,
        stdout=stdout,
    )

    rows = parse_jsonl(stdout.getvalue())
    prompts = dict(backend.prompts)
    assert rows[0]["passed"]
    assert "Document 6:" in prompts["chunk_5"]
    assert rows[1]["model_call_count"] == 7


def test_summary_llm_judge_uses_injected_judge() -> None:
    stdout = StringIO()
    backend = FakeBackend({"summarizer": ["alpha beta gamma"]})

    run_workflow_evaluation(
        workflow=build_summary_direct_workflow("summary_llm_judge"),
        dataset="multi_news",
        samples=[build_summary_sample()],
        mode="dag",
        backend=backend,
        stdout=stdout,
        summary_judge=FakeSummaryJudge(),
    )

    rows = parse_jsonl(stdout.getvalue())
    assert rows[0]["passed"]
    assert rows[0]["evaluator_results"]["judge"]["llm_score"] == 4.0
    assert rows[1]["avg_llm_score"] == 4.0
    assert rows[1]["llm_pass_rate"] == 1.0


def test_main_prints_jsonl_for_single_workflow(tmp_path: Path) -> None:
    workflow_path = tmp_path / "workflow.yaml"
    workflow_path.write_text(
        yaml.safe_dump(build_direct_workflow().model_dump(mode="json")),
        encoding="utf-8",
    )
    stdout = StringIO()
    backend = FakeBackend({"solver": ["#### 0"]})

    exit_code = main(
        [
            "--workflow",
            str(workflow_path),
            "--dataset",
            "gsm8k",
            "--limit",
            "1",
            "--seed",
            "42",
        ],
        backend=backend,
        stdout=stdout,
    )

    rows = parse_jsonl(stdout.getvalue())
    assert exit_code == 0
    assert rows[0]["record_type"] == "sample"
    assert rows[1]["record_type"] == "summary"
    assert backend.preloaded == ["solver"]


def test_main_runs_sample_id_file_subset(tmp_path: Path) -> None:
    workflow_path = tmp_path / "workflow.yaml"
    workflow_path.write_text(
        yaml.safe_dump(build_direct_workflow().model_dump(mode="json")),
        encoding="utf-8",
    )
    sample_ids_path = tmp_path / "sample_ids.jsonl"
    sample_ids_path.write_text(
        json.dumps({"sample_id": "test_00001"}) + "\n",
        encoding="utf-8",
    )
    stdout = StringIO()
    backend = FakeBackend({"solver": ["#### 18"]})

    exit_code = main(
        [
            "--workflow",
            str(workflow_path),
            "--dataset",
            "gsm8k",
            "--sample-ids",
            str(sample_ids_path),
        ],
        backend=backend,
        stdout=stdout,
    )

    rows = parse_jsonl(stdout.getvalue())
    assert exit_code == 0
    assert rows[0]["sample_id"] == "test_00001"
    assert rows[1]["sample_count"] == 1


def test_main_resumes_from_existing_jsonl(tmp_path: Path) -> None:
    workflow_path = tmp_path / "workflow.yaml"
    workflow_path.write_text(
        yaml.safe_dump(build_direct_workflow().model_dump(mode="json")),
        encoding="utf-8",
    )
    resume_path = tmp_path / "partial.jsonl"
    resume_path.write_text(
        json.dumps(
            {
                "record_type": "sample",
                "dataset": "gsm8k",
                "mode": "dag",
                "sample_id": "test_01309",
                "split": "test",
                "passed": True,
                "duration_sec": 1.0,
                "final_output": "#### 0",
                "node_outputs": {},
                "evaluator_results": {},
                "node_records": [
                    {
                        "node_name": "judge",
                        "type": "evaluator",
                        "duration_sec": 0.0,
                        "passed": True,
                        "result": {"exact_match": True, "error_type": None},
                    }
                ],
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    stdout = StringIO()
    backend = FakeBackend({"solver": ["#### 0"]})

    exit_code = main(
        [
            "--workflow",
            str(workflow_path),
            "--dataset",
            "gsm8k",
            "--limit",
            "2",
            "--seed",
            "42",
            "--resume-from",
            str(resume_path),
        ],
        backend=backend,
        stdout=stdout,
    )

    rows = parse_jsonl(stdout.getvalue())
    assert exit_code == 0
    assert [row["record_type"] for row in rows] == ["sample", "summary"]
    assert rows[0]["sample_id"] == "test_00228"
    assert rows[1]["sample_count"] == 2
    assert rows[1]["passed_count"] >= 1


def test_load_resume_sample_records_ignores_summary(tmp_path: Path) -> None:
    resume_path = tmp_path / "partial.jsonl"
    resume_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "record_type": "sample",
                        "sample_id": "test_00000",
                        "passed": True,
                    },
                    sort_keys=True,
                ),
                json.dumps({"record_type": "summary", "sample_count": 1}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    records = load_resume_sample_records(resume_path)

    assert [record["sample_id"] for record in records] == ["test_00000"]
