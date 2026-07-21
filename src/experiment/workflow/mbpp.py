from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

from langchain.agents import AgentState
from langchain_core.messages import AIMessage
from pydantic import BaseModel, ConfigDict, JsonValue

from common.validate import NonEmptyStr
from dataset.mbpp import evaluate_mbpp, load_mbpp_samples
from dataset.schema import MbppEvaluation, TaskSample
from experiment.workflow.sample import load_sample_manifest, resolve_manifest_samples
from workflow.schema import Workflow

MBPP_MANIFEST_PATH = (
    Path(__file__).with_name("assets") / "mbpp_sanitized_test_60_seed42.jsonl"
)
QWEN3_4B_PATH = "/data/Models/Qwen/Qwen3-4B"
QWEN3_8B_PATH = "/data/Models/Qwen/Qwen3-8B"
QWEN3_14B_PATH = "/data/Models/Qwen/Qwen3-14B"

MBPP_CODER_PROMPT = (
    "Write a correct Python solution for the programming task.\n"
    "Return only Python code in one fenced code block.\n\n"
    "Task:\n{task}\n\nTests that must pass:\n{tests}"
)
MBPP_REVIEWER_PROMPT = (
    "Review the Python solution for correctness.\n"
    "Focus on logic errors, edge cases, and API misuse.\n"
    "Return only the review analysis, no code.\n\n"
    "Task:\n{task}\n\nCurrent solution and test result:\n{previous_output}\n\n"
    "Tests:\n{tests}"
)
MBPP_REPAIR_PROMPT = (
    "Repair the Python solution so that it passes the tests.\n"
    "Return only Python code in one fenced code block.\n"
    "The execution context JSON contains the tester's code and initial result "
    "and the reviewer's analysis.\n\n"
    "Task:\n{task}\n\nExecution context:\n{node_outputs_json}\n\n"
    "Tests:\n{tests}"
)


class MbppAttempt(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    code: str
    initial_eval: MbppEvaluation


class MbppWorkflowResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    sample_id: NonEmptyStr
    initial_code: str
    final_code: str
    initial_eval: MbppEvaluation
    final_eval: MbppEvaluation


def load_mbpp_experiment_samples(
    dataset_path: str | Path,
    manifest_path: str | Path = MBPP_MANIFEST_PATH,
) -> tuple[TaskSample, ...]:
    samples = [
        sample for sample in load_mbpp_samples(dataset_path) if sample.split == "test"
    ]
    return resolve_mbpp_samples(samples, manifest_path)


def resolve_mbpp_samples(
    samples: Sequence[TaskSample],
    manifest_path: str | Path = MBPP_MANIFEST_PATH,
) -> tuple[TaskSample, ...]:
    entries = load_sample_manifest(manifest_path)
    resolved = resolve_manifest_samples(entries, samples)
    for entry, sample in zip(entries, resolved, strict=True):
        if sample.source_dataset != "mbpp_sanitized" or sample.split != "test":
            raise ValueError(f"manifest sample is not MBPP test: {sample.sample_id}")
        if entry.metadata != _mbpp_metadata(sample):
            raise ValueError(f"sample metadata mismatch: {sample.sample_id}")
    return resolved


def build_mbpp_session_inputs(sample: TaskSample) -> dict[str, JsonValue]:
    if sample.source_dataset != "mbpp_sanitized" or sample.split != "test":
        raise ValueError(f"sample is not MBPP test: {sample.sample_id}")
    return {
        "sample_id": sample.sample_id,
        "task": sample.input_text,
        "tests": "\n".join(sample.test_imports + sample.test_list),
        "test_imports_json": _json_payload(sample.test_imports),
        "test_list_json": _json_payload(sample.test_list),
    }


def evaluate_mbpp_initial(
    session_inputs: Mapping[str, object],
    states: Mapping[str, AgentState],
    parameters: Mapping[str, object],
) -> AgentState:
    if parameters or set(states) != {"coder"}:
        raise ValueError("MBPP tester expects only the coder state")
    code = _state_text(states, "coder")
    evaluation = evaluate_mbpp(
        code,
        _string_list_input(session_inputs, "test_imports_json"),
        _string_list_input(session_inputs, "test_list_json"),
    )
    attempt = MbppAttempt(code=code, initial_eval=evaluation)
    return AgentState(messages=[AIMessage(content=_json_payload(attempt))])


def evaluate_mbpp_final(
    session_inputs: Mapping[str, object],
    states: Mapping[str, AgentState],
    parameters: Mapping[str, object],
) -> AgentState:
    if parameters or set(states) != {"tester", "repair"}:
        raise ValueError("MBPP final tester expects tester and repair states")
    attempt = MbppAttempt.model_validate_json(_state_text(states, "tester"))
    final_code = _state_text(states, "repair")
    final_eval = evaluate_mbpp(
        final_code,
        _string_list_input(session_inputs, "test_imports_json"),
        _string_list_input(session_inputs, "test_list_json"),
    )
    sample_id = session_inputs.get("sample_id")
    if not isinstance(sample_id, str):
        raise TypeError("MBPP sample id must be text")
    result = MbppWorkflowResult(
        sample_id=sample_id,
        initial_code=attempt.code,
        final_code=final_code,
        initial_eval=attempt.initial_eval,
        final_eval=final_eval,
    )
    return AgentState(messages=[AIMessage(content=_json_payload(result))])


def parse_mbpp_result(output: str) -> MbppWorkflowResult:
    return MbppWorkflowResult.model_validate_json(output)


def mbpp_functions() -> dict[str, Callable[..., object]]:
    return {
        "evaluate_mbpp_initial": evaluate_mbpp_initial,
        "evaluate_mbpp_final": evaluate_mbpp_final,
    }


def build_mbpp_workflow(
    *,
    workflow_name: str = "mbpp",
    max_num_seqs: int = 3,
    queue_capacity: int = 16,
    qwen3_4b_path: str = QWEN3_4B_PATH,
    qwen3_8b_path: str = QWEN3_8B_PATH,
    qwen3_14b_path: str = QWEN3_14B_PATH,
) -> Workflow:
    nodes = [
        _agent_node(
            "coder",
            "Qwen3-14B",
            qwen3_14b_path,
            MBPP_CODER_PROMPT,
            max_model_len=1024,
            max_num_seqs=max_num_seqs,
            queue_capacity=queue_capacity,
        ),
        {
            "name": "tester",
            "type": "function",
            "function": "evaluate_mbpp_initial",
            "max_concurrency": 8,
            "queue_capacity": queue_capacity,
        },
        _agent_node(
            "reviewer",
            "Qwen3-4B",
            qwen3_4b_path,
            MBPP_REVIEWER_PROMPT,
            max_model_len=2048,
            max_num_seqs=max_num_seqs,
            queue_capacity=queue_capacity,
        ),
        _agent_node(
            "repair",
            "Qwen3-8B",
            qwen3_8b_path,
            MBPP_REPAIR_PROMPT,
            max_model_len=3072,
            max_num_seqs=max_num_seqs,
            queue_capacity=queue_capacity,
        ),
        {
            "name": "final_tester",
            "type": "function",
            "function": "evaluate_mbpp_final",
            "max_concurrency": 8,
            "queue_capacity": queue_capacity,
        },
    ]
    edges = [
        {"source": "coder", "target": "tester"},
        {"source": "tester", "target": "reviewer"},
        {"source": "tester", "target": "repair"},
        {"source": "reviewer", "target": "repair"},
        {"source": "tester", "target": "final_tester"},
        {"source": "repair", "target": "final_tester"},
    ]
    return Workflow.model_validate(
        {"workflow_name": workflow_name, "nodes": nodes, "edges": edges}
    )


def _agent_node(
    name: str,
    model_name: str,
    model_path: str,
    prompt_template: str,
    *,
    max_model_len: int,
    max_num_seqs: int,
    queue_capacity: int,
) -> dict[str, object]:
    return {
        "name": name,
        "type": "agent",
        "task": "text_generation",
        "queue_capacity": queue_capacity,
        "model": {"name": model_name},
        "execution": {
            "model_path": model_path,
            "dtype": "float16",
            "max_new_tokens": 512,
            "do_sample": False,
            "use_chat_template": True,
            "enable_thinking": False,
            "serving": {
                "max_model_len": max_model_len,
                "max_num_seqs": max_num_seqs,
                "max_num_batched_tokens": max_model_len * max_num_seqs,
                "gpu_memory_utilization": 0.98,
            },
        },
        "prompt_template": prompt_template,
    }


def _mbpp_metadata(sample: TaskSample) -> dict[str, JsonValue]:
    task_id = sample.metadata.get("task_id")
    if not isinstance(task_id, int) or isinstance(task_id, bool):
        raise TypeError(f"MBPP task id must be an integer: {sample.sample_id}")
    return {
        "input_char_count": len(sample.input_text),
        "task_id": task_id,
        "test_count": len(sample.test_list),
    }


def _state_text(states: Mapping[str, AgentState], node_name: str) -> str:
    messages = states[node_name].get("messages")
    if not messages:
        raise ValueError(f"MBPP state has no messages: {node_name}")
    content = messages[-1].content
    if not isinstance(content, str):
        raise TypeError(f"MBPP state message must be text: {node_name}")
    return content


def _string_list_input(
    session_inputs: Mapping[str, object],
    key: str,
) -> list[str]:
    value = session_inputs.get(key)
    if not isinstance(value, str):
        raise TypeError(f"MBPP session input must be JSON text: {key}")
    decoded = json.loads(value)
    if not isinstance(decoded, list) or any(
        not isinstance(item, str) for item in decoded
    ):
        raise TypeError(f"MBPP session input must contain a string list: {key}")
    return decoded


def _json_payload(value: object) -> str:
    if isinstance(value, BaseModel):
        value = value.model_dump(mode="json")
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
