from __future__ import annotations

import hashlib
from collections import Counter
from pathlib import Path

import pytest
from langchain.agents import AgentState
from langchain_core.messages import AIMessage

from dataset.schema import TaskSample
from experiment.workflow.mbpp import (
    MBPP_MANIFEST_PATH,
    MBPP_REPAIR_PROMPT,
    build_mbpp_session_inputs,
    build_mbpp_workflow,
    evaluate_mbpp_final,
    evaluate_mbpp_initial,
    parse_mbpp_result,
)
from experiment.workflow.qmsum import (
    QMSUM_MANIFEST_PATH,
    build_qmsum_session_inputs,
    build_qmsum_workflow,
    evaluate_qmsum_output,
    resolve_qmsum_samples,
    split_qmsum_session,
)
from experiment.workflow.sample import (
    SampleManifestEntry,
    load_sample_manifest,
    resolve_manifest_samples,
    stable_sample_digest,
)
from workflow.schema import AgentNodeConfig, FunctionNodeConfig
from workflow.worker import build_prompt_context


def test_frozen_sample_manifests() -> None:
    expected = {
        QMSUM_MANIFEST_PATH: (
            "c29706a27493dc2c7b0003b2b78d132d1e2f7ce207f108f7a358767828645441",
            "test_Bed003_specific_005",
            "test_education_9_specific_011",
        ),
        MBPP_MANIFEST_PATH: (
            "7e43782d5cbe77a382566fa9f6bdce31af25c28f49aa3b0eeaa5a2e4379ed49e",
            "11",
            "478",
        ),
    }
    for path, (digest, first_id, last_id) in expected.items():
        entries = load_sample_manifest(path)
        assert len(entries) == 60
        assert entries[0].sample_id == first_id
        assert entries[-1].sample_id == last_id
        assert Counter(entry.stratum for entry in entries) == {
            "short": 20,
            "medium": 20,
            "long": 20,
        }
        assert hashlib.sha256(path.read_bytes()).hexdigest() == digest


def test_manifest_resolver_rejects_source_drift() -> None:
    sample = qmsum_sample()
    entry = SampleManifestEntry(
        position=0,
        sample_id=sample.sample_id,
        stratum="short",
        metadata={},
        digest=stable_sample_digest(sample),
    )
    changed = sample.model_copy(update={"input_text": sample.input_text + " changed"})

    with pytest.raises(ValueError, match="sample digest mismatch"):
        resolve_manifest_samples((entry,), (changed,))


def test_manifest_loader_requires_ordered_positions(tmp_path: Path) -> None:
    sample = qmsum_sample()
    entry = SampleManifestEntry(
        position=1,
        sample_id=sample.sample_id,
        stratum="short",
        metadata={},
        digest=stable_sample_digest(sample),
    )
    path = tmp_path / "samples.jsonl"
    path.write_text(entry.model_dump_json() + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="positions must be contiguous"):
        load_sample_manifest(path)


def test_qmsum_resolver_validates_manifest_metadata(tmp_path: Path) -> None:
    sample = qmsum_sample()
    entry = SampleManifestEntry(
        position=0,
        sample_id=sample.sample_id,
        stratum="short",
        metadata={"input_char_count": 1},
        digest=stable_sample_digest(sample),
    )
    path = tmp_path / "qmsum.jsonl"
    path.write_text(entry.model_dump_json() + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="sample metadata mismatch"):
        resolve_qmsum_samples((sample,), path)


def test_qmsum_split_and_quality_are_offline() -> None:
    sample = qmsum_sample()
    inputs = build_qmsum_session_inputs(sample)

    assert "gold_answer" not in inputs
    split = split_qmsum_session(inputs, {}, {})
    assert list(split) == [f"chunk_{index}" for index in range(6)]
    chunks = [message_text(split[f"chunk_{index}"]) for index in range(6)]
    assert "\n\n".join(chunks) == sample.input_text
    result = evaluate_qmsum_output(sample, "The answer is one.")
    assert result.rouge1 == pytest.approx(1.0)
    assert result.rouge2 == pytest.approx(1.0)
    assert result.rouge_l == pytest.approx(1.0)


def test_qmsum_workflow_has_one_shared_chunk_deployment() -> None:
    workflow = build_qmsum_workflow(max_num_seqs=3, queue_capacity=4)
    nodes = workflow.node_map()

    assert workflow.graph.entry_node == "split"
    assert workflow.graph.terminal_node == "merge"
    split = nodes["split"]
    assert isinstance(split, FunctionNodeConfig)
    assert split.routing == "targeted"
    chunks = [nodes[f"chunk_{index}"] for index in range(6)]
    assert all(isinstance(node, AgentNodeConfig) for node in chunks)
    chunk_deployments = {
        (node.model.model_dump_json(), node.execution.model_dump_json())
        for node in chunks
        if isinstance(node, AgentNodeConfig)
    }
    assert len(chunk_deployments) == 1
    chunk = chunks[0]
    assert isinstance(chunk, AgentNodeConfig)
    assert chunk.execution.max_new_tokens == 384
    assert chunk.execution.serving.max_model_len == 8192
    assert chunk.execution.serving.max_num_batched_tokens == 24576
    merge = nodes["merge"]
    assert isinstance(merge, AgentNodeConfig)
    assert merge.execution.max_new_tokens == 384
    assert workflow.graph.dependencies["merge"] == tuple(
        f"chunk_{index}" for index in range(6)
    )


def test_mbpp_functions_preserve_initial_and_final_evaluations() -> None:
    sample = mbpp_sample()
    inputs = build_mbpp_session_inputs(sample)
    assert "reference_code" not in inputs
    coder = AgentState(
        messages=[AIMessage(content="def add_one(value):\n    return value")]
    )
    tester = evaluate_mbpp_initial(inputs, {"coder": coder}, {})
    reviewer = AgentState(
        messages=[AIMessage(content="The return value is unchanged.")]
    )
    repair = AgentState(
        messages=[AIMessage(content="def add_one(value):\n    return value + 1")]
    )

    assert "initial_eval" in message_text(tester)
    context = build_prompt_context(inputs, {"tester": tester, "reviewer": reviewer})
    repair_prompt = MBPP_REPAIR_PROMPT.format_map(context)
    assert "def add_one" in repair_prompt
    assert "The return value is unchanged." in repair_prompt
    final = evaluate_mbpp_final(
        inputs,
        {"tester": tester, "repair": repair},
        {},
    )
    result = parse_mbpp_result(message_text(final))
    assert not result.initial_eval.passed
    assert result.initial_eval.error_type == "assertion_error"
    assert result.final_eval.passed
    assert result.final_code.endswith("value + 1")


def test_mbpp_workflow_keeps_chain_critical_path() -> None:
    workflow = build_mbpp_workflow(max_num_seqs=3, queue_capacity=16)
    nodes = workflow.node_map()

    assert workflow.graph.entry_node == "coder"
    assert workflow.graph.terminal_node == "final_tester"
    assert workflow.graph.dependencies["reviewer"] == ("tester",)
    assert workflow.graph.dependencies["repair"] == ("tester", "reviewer")
    assert workflow.graph.dependencies["final_tester"] == ("tester", "repair")
    tester = nodes["tester"]
    final_tester = nodes["final_tester"]
    assert isinstance(tester, FunctionNodeConfig)
    assert isinstance(final_tester, FunctionNodeConfig)
    assert tester.max_concurrency == 8
    assert final_tester.max_concurrency == 8
    expected = {
        "coder": ("Qwen3-14B", 1024, 3072),
        "reviewer": ("Qwen3-4B", 2048, 6144),
        "repair": ("Qwen3-8B", 3072, 9216),
    }
    for name, (model_name, max_model_len, max_batched_tokens) in expected.items():
        node = nodes[name]
        assert isinstance(node, AgentNodeConfig)
        assert node.model.name == model_name
        assert node.execution.max_new_tokens == 512
        assert node.execution.serving.max_model_len == max_model_len
        assert node.execution.serving.max_num_batched_tokens == max_batched_tokens


def qmsum_sample() -> TaskSample:
    turns = [f"Speaker {index}: turn {index}" for index in range(6)]
    return TaskSample(
        sample_id="test_fixture_specific_000",
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


def mbpp_sample() -> TaskSample:
    return TaskSample(
        sample_id="11",
        source_dataset="mbpp_sanitized",
        split="test",
        task_type="python_function_generation",
        input_text="Write add_one.",
        reference_code="def add_one(value):\n    return value + 1",
        test_list=["assert add_one(1) == 2", "assert add_one(5) == 6"],
        quality_metric="pass_at_1",
        metadata={"task_id": 11},
    )


def message_text(state: AgentState) -> str:
    content = state["messages"][-1].content
    assert isinstance(content, str), type(content)
    return content
