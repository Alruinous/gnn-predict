from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

from langchain.agents import AgentState
from langchain_core.messages import AIMessage
from pydantic import JsonValue

from dataset.schema import SummaryEvaluation, TaskSample
from dataset.summarization import evaluate_summary_rouge, load_qmsum_split
from experiment.workflow.sample import (
    load_sample_manifest,
    resolve_manifest_samples,
)
from workflow.schema import Workflow

QMSUM_MANIFEST_PATH = Path(__file__).with_name("assets") / "qmsum_test_60_seed42.jsonl"
QMSUM_CHUNK_COUNT = 6
QWEN3_4B_PATH = "/data/Models/Qwen/Qwen3-4B"
QWEN3_8B_PATH = "/data/Models/Qwen/Qwen3-8B"

QMSUM_CHUNK_PROMPT = (
    "Summarize transcript details relevant to the query.\n"
    "Preserve decisions, reasons, participants, and concrete outcomes.\n"
    "Return only the partial answer summary.\n\n"
    "Query:\n{query}\n\nTranscript chunk:\n{previous_output}"
)
QMSUM_MERGE_PROMPT = (
    "Merge the partial answers into one final answer to the query.\n"
    "Remove duplication, preserve query-relevant facts, and keep the answer "
    "grounded.\n"
    "Return only the final answer.\n\n"
    "Query:\n{query}\n\nPartial answers:\n{node_outputs_json}"
)


def load_qmsum_experiment_samples(
    dataset_path: str | Path,
    manifest_path: str | Path = QMSUM_MANIFEST_PATH,
) -> tuple[TaskSample, ...]:
    return resolve_qmsum_samples(
        load_qmsum_split(dataset_path, "test"),
        manifest_path,
    )


def resolve_qmsum_samples(
    samples: Sequence[TaskSample],
    manifest_path: str | Path = QMSUM_MANIFEST_PATH,
) -> tuple[TaskSample, ...]:
    entries = load_sample_manifest(manifest_path)
    resolved = resolve_manifest_samples(entries, samples)
    for entry, sample in zip(entries, resolved, strict=True):
        if sample.source_dataset != "qmsum" or sample.split != "test":
            raise ValueError(f"manifest sample is not QMSum test: {sample.sample_id}")
        if entry.metadata != _qmsum_metadata(sample):
            raise ValueError(f"sample metadata mismatch: {sample.sample_id}")
    return resolved


def build_qmsum_session_inputs(sample: TaskSample) -> dict[str, JsonValue]:
    if sample.source_dataset != "qmsum" or sample.split != "test":
        raise ValueError(f"sample is not QMSum test: {sample.sample_id}")
    query = sample.metadata.get("query")
    if not isinstance(query, str):
        raise TypeError(f"QMSum query must be text: {sample.sample_id}")
    return {
        "sample_id": sample.sample_id,
        "query": query,
        "transcript": sample.input_text,
    }


def split_qmsum_session(
    session_inputs: Mapping[str, object],
    states: Mapping[str, AgentState],
    parameters: Mapping[str, object],
) -> dict[str, AgentState]:
    if states or parameters:
        raise ValueError("QMSum split expects no input states or parameters")
    transcript = session_inputs.get("transcript")
    if not isinstance(transcript, str):
        raise TypeError("QMSum transcript must be text")
    turns = transcript.split("\n\n")
    chunks = _split_evenly(turns, QMSUM_CHUNK_COUNT)
    return {
        f"chunk_{index}": AgentState(messages=[AIMessage(content="\n\n".join(chunk))])
        for index, chunk in enumerate(chunks)
    }


def evaluate_qmsum_output(sample: TaskSample, output: str) -> SummaryEvaluation:
    if sample.gold_answer is None:
        raise ValueError(f"QMSum sample is missing gold answer: {sample.sample_id}")
    return evaluate_summary_rouge(output, sample.gold_answer)


def qmsum_functions() -> dict[str, Callable[..., object]]:
    return {"split_qmsum": split_qmsum_session}


def build_qmsum_workflow(
    *,
    max_num_seqs: int = 3,
    queue_capacity: int = 16,
    qwen3_4b_path: str = QWEN3_4B_PATH,
    qwen3_8b_path: str = QWEN3_8B_PATH,
) -> Workflow:
    chunk_execution = _execution(
        qwen3_4b_path,
        max_model_len=8192,
        max_new_tokens=384,
        max_num_seqs=max_num_seqs,
    )
    nodes: list[dict[str, object]] = [
        {
            "name": "split",
            "type": "function",
            "function": "split_qmsum",
            "routing": "targeted",
            "queue_capacity": queue_capacity,
        }
    ]
    nodes.extend(
        {
            "name": f"chunk_{index}",
            "type": "agent",
            "task": "text_generation",
            "queue_capacity": queue_capacity,
            "model": {"name": "Qwen3-4B"},
            "execution": chunk_execution,
            "prompt_template": QMSUM_CHUNK_PROMPT,
        }
        for index in range(QMSUM_CHUNK_COUNT)
    )
    nodes.append(
        {
            "name": "merge",
            "type": "agent",
            "task": "text_generation",
            "queue_capacity": queue_capacity,
            "model": {"name": "Qwen3-8B"},
            "execution": _execution(
                qwen3_8b_path,
                max_model_len=4096,
                max_new_tokens=384,
                max_num_seqs=max_num_seqs,
            ),
            "prompt_template": QMSUM_MERGE_PROMPT,
        }
    )
    edges = [
        {"source": "split", "target": f"chunk_{index}"}
        for index in range(QMSUM_CHUNK_COUNT)
    ]
    edges.extend(
        {"source": f"chunk_{index}", "target": "merge"}
        for index in range(QMSUM_CHUNK_COUNT)
    )
    return Workflow.model_validate({"nodes": nodes, "edges": edges})


def _qmsum_metadata(sample: TaskSample) -> dict[str, JsonValue]:
    query = sample.metadata.get("query")
    query_type = sample.metadata.get("query_type")
    turn_count = sample.metadata.get("turn_count")
    if not isinstance(query, str) or not isinstance(query_type, str):
        raise TypeError(f"QMSum metadata contains invalid text: {sample.sample_id}")
    if not isinstance(turn_count, int) or isinstance(turn_count, bool):
        raise TypeError(f"QMSum turn count must be an integer: {sample.sample_id}")
    return {
        "input_char_count": len(sample.input_text),
        "query": query,
        "query_type": query_type,
        "turn_count": turn_count,
    }


def _split_evenly[T](items: Sequence[T], part_count: int) -> tuple[Sequence[T], ...]:
    if len(items) < part_count:
        raise ValueError("QMSum transcript has fewer turns than chunks")
    return tuple(
        items[
            (len(items) * index) // part_count : (len(items) * (index + 1))
            // part_count
        ]
        for index in range(part_count)
    )


def _execution(
    model_path: str,
    *,
    max_model_len: int,
    max_new_tokens: int,
    max_num_seqs: int,
) -> dict[str, object]:
    return {
        "model_path": model_path,
        "dtype": "float16",
        "max_new_tokens": max_new_tokens,
        "do_sample": False,
        "use_chat_template": True,
        "enable_thinking": False,
        "serving": {
            "max_model_len": max_model_len,
            "max_num_seqs": max_num_seqs,
            "max_num_batched_tokens": max_model_len * max_num_seqs,
            "gpu_memory_utilization": 0.98,
        },
    }
