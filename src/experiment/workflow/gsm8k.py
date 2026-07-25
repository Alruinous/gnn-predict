from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path

from langchain.agents import AgentState
from langchain_core.messages import AIMessage
from pydantic import JsonValue

from dataset.gsm8k import extract_final_numeric_answer, load_gsm8k_split
from dataset.schema import TaskSample
from experiment.workflow.sample import load_sample_manifest, resolve_manifest_samples
from workflow.schema import Workflow

GSM8K_MANIFEST_PATH = Path(__file__).with_name("assets") / "gsm8k_test_60_seed42.jsonl"
GSM8K_SOLVER_COUNT = 5
QWEN3_4B_PATH = "/data/Models/Qwen/Qwen3-4B"
QWEN3_14B_PATH = "/data/Models/Qwen/Qwen3-14B"

GSM8K_SOLVE_PROMPT = (
    "Solve the math word problem step by step.\n"
    "End with a line '#### ' followed by the final numeric answer.\n\n"
    "Problem:\n{question}"
)
GSM8K_REFINE_PROMPT = (
    "Decide the final answer to the math word problem using the candidate "
    "solutions and their majority vote.\n"
    "End with a line '#### ' followed by the final numeric answer.\n\n"
    "Problem:\n{question}\n\nCandidate solutions:\n{previous_output}"
)


def load_gsm8k_experiment_samples(
    dataset_path: str | Path,
    manifest_path: str | Path = GSM8K_MANIFEST_PATH,
) -> tuple[TaskSample, ...]:
    return resolve_gsm8k_samples(load_gsm8k_split(dataset_path, "test"), manifest_path)


def resolve_gsm8k_samples(
    samples: Sequence[TaskSample],
    manifest_path: str | Path = GSM8K_MANIFEST_PATH,
) -> tuple[TaskSample, ...]:
    entries = load_sample_manifest(manifest_path)
    resolved = resolve_manifest_samples(entries, samples)
    for entry, sample in zip(entries, resolved, strict=True):
        if sample.source_dataset != "gsm8k" or sample.split != "test":
            raise ValueError(f"manifest sample is not GSM8K test: {sample.sample_id}")
        if entry.metadata != _gsm8k_metadata(sample):
            raise ValueError(f"sample metadata mismatch: {sample.sample_id}")
    return resolved


def build_gsm8k_session_inputs(sample: TaskSample) -> dict[str, JsonValue]:
    if sample.source_dataset != "gsm8k" or sample.split != "test":
        raise ValueError(f"sample is not GSM8K test: {sample.sample_id}")
    return {"sample_id": sample.sample_id, "question": sample.input_text}


def fanout_gsm8k_session(
    session_inputs: Mapping[str, object],
    states: Mapping[str, AgentState],
    parameters: Mapping[str, object],
) -> dict[str, AgentState]:
    if states or parameters:
        raise ValueError("GSM8K fanout expects no input states or parameters")
    question = session_inputs.get("question")
    if not isinstance(question, str):
        raise TypeError("GSM8K question must be text")
    return {
        f"solve_{index}": AgentState(messages=[AIMessage(content=question)])
        for index in range(GSM8K_SOLVER_COUNT)
    }


def aggregate_gsm8k_votes(
    session_inputs: Mapping[str, object],
    states: Mapping[str, AgentState],
    parameters: Mapping[str, object],
) -> AgentState:
    expected = {f"solve_{index}" for index in range(GSM8K_SOLVER_COUNT)}
    if parameters or set(states) != expected:
        raise ValueError("GSM8K aggregate expects only the solver states")
    reasonings = {name: _state_text(states, name) for name in sorted(states)}
    answers = {
        name: extract_final_numeric_answer(text) for name, text in reasonings.items()
    }
    majority = _majority_answer(answers.values())
    representative = _representative_reasoning(reasonings, answers, majority)
    content = (
        f"Majority answer: {majority if majority is not None else 'none'}\n\n"
        f"Representative reasoning:\n{representative}"
    )
    return AgentState(messages=[AIMessage(content=content)])


def gsm8k_functions() -> dict[str, Callable[..., object]]:
    return {
        "fanout_gsm8k": fanout_gsm8k_session,
        "aggregate_gsm8k_votes": aggregate_gsm8k_votes,
    }


def build_gsm8k_workflow(
    *,
    workflow_name: str = "gsm8k",
    max_num_seqs: int = 3,
    queue_capacity: int = 16,
    qwen3_4b_path: str = QWEN3_4B_PATH,
    qwen3_14b_path: str = QWEN3_14B_PATH,
) -> Workflow:
    nodes: list[dict[str, object]] = [
        {
            "name": "fanout",
            "type": "function",
            "function": "fanout_gsm8k",
            "routing": "targeted",
            "queue_capacity": queue_capacity,
        }
    ]
    nodes.extend(
        {
            "name": f"solve_{index}",
            "type": "agent",
            "task": "text_generation",
            "queue_capacity": queue_capacity,
            "model": {"name": "Qwen3-4B"},
            "execution": _execution(
                qwen3_4b_path,
                max_model_len=1024,
                max_new_tokens=512,
                max_num_seqs=max_num_seqs,
            ),
            "prompt_template": GSM8K_SOLVE_PROMPT,
        }
        for index in range(GSM8K_SOLVER_COUNT)
    )
    nodes.append(
        {
            "name": "aggregate",
            "type": "function",
            "function": "aggregate_gsm8k_votes",
            "max_concurrency": 8,
            "queue_capacity": queue_capacity,
        }
    )
    nodes.append(
        {
            "name": "refine",
            "type": "agent",
            "task": "text_generation",
            "queue_capacity": queue_capacity,
            "model": {"name": "Qwen3-14B"},
            "execution": _execution(
                qwen3_14b_path,
                max_model_len=2048,
                max_new_tokens=512,
                max_num_seqs=max_num_seqs,
            ),
            "prompt_template": GSM8K_REFINE_PROMPT,
        }
    )
    edges = [
        {"source": "fanout", "target": f"solve_{index}"}
        for index in range(GSM8K_SOLVER_COUNT)
    ]
    edges.extend(
        {"source": f"solve_{index}", "target": "aggregate"}
        for index in range(GSM8K_SOLVER_COUNT)
    )
    edges.append({"source": "aggregate", "target": "refine"})
    return Workflow.model_validate(
        {"workflow_name": workflow_name, "nodes": nodes, "edges": edges}
    )


def _gsm8k_metadata(sample: TaskSample) -> dict[str, JsonValue]:
    return {"input_char_count": len(sample.input_text)}


def _majority_answer(answers: Iterable[str | None]) -> str | None:
    counts = Counter(answer for answer in answers if answer is not None)
    if not counts:
        return None
    return counts.most_common(1)[0][0]


def _representative_reasoning(
    reasonings: Mapping[str, str],
    answers: Mapping[str, str | None],
    majority: str | None,
) -> str:
    for name in sorted(reasonings):
        if answers[name] == majority:
            return reasonings[name]
    return reasonings[min(reasonings)]


def _state_text(states: Mapping[str, AgentState], node_name: str) -> str:
    messages = states[node_name].get("messages")
    if not messages:
        raise ValueError(f"GSM8K state has no messages: {node_name}")
    content = messages[-1].content
    if not isinstance(content, str):
        raise TypeError(f"GSM8K state message must be text: {node_name}")
    return content


def _execution(
    model_path: str,
    *,
    max_model_len: int,
    max_new_tokens: int,
    max_num_seqs: int,
    do_sample: bool = False,
    temperature: float | None = None,
) -> dict[str, object]:
    return {
        "model_path": model_path,
        "dtype": "float16",
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "temperature": temperature,
        "use_chat_template": True,
        "enable_thinking": False,
        "serving": {
            "max_model_len": max_model_len,
            "max_num_seqs": max_num_seqs,
            "max_num_batched_tokens": max_model_len * max_num_seqs,
            "gpu_memory_utilization": 0.98,
        },
    }
