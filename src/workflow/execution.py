from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Any, Literal, Protocol, TextIO, cast

from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

from dataset.gsm8k import evaluate_gsm8k, load_gsm8k_split
from dataset.mbpp import evaluate_mbpp, load_mbpp_samples
from dataset.schema import TaskSample
from dataset.summarization import (
    OpenRouterSummaryJudge,
    SummaryJudge,
    evaluate_summary_rouge,
    evaluate_summary_with_judge,
    load_multi_news_split,
    load_qmsum_split,
)
from workflow.loader import load_workflow
from workflow.schema import Workflow, WorkflowExecutionConfig, WorkflowNodeConfig
from workflow.validation import validate_workflow

WorkflowDataset = Literal["gsm8k", "mbpp", "multi_news", "qmsum"]
WorkflowEvalMode = Literal["dag", "react", "parallel"]
MODEL_NODE_TYPES = {"agent", "tool"}
DATASET_EVALUATOR_TASKS = {
    "gsm8k": {"gsm8k_numeric_exact_match"},
    "mbpp": {"mbpp_pass_at_1"},
    "multi_news": {"summary_rouge", "summary_llm_judge"},
    "qmsum": {"summary_rouge", "summary_llm_judge"},
}
DEFAULT_MAX_NEW_TOKENS = 256
DEFAULT_REACT_MAX_STEPS = 8
DEFAULT_INPUT_CHUNK_COUNT = 3
CHUNK_PROMPT_PATTERN = re.compile(r"\{chunk_(\d+)_text\}")


class WorkflowExecutionError(RuntimeError):
    pass


class TextGenerationBackend(Protocol):
    def preload(self, nodes: Sequence[WorkflowNodeConfig]) -> None: ...

    def generate(self, node: WorkflowNodeConfig, prompt: str) -> GenerationResult: ...


class WorkflowRunState(TypedDict, total=False):
    sample: TaskSample
    dataset: WorkflowDataset
    node_outputs: dict[str, str]
    evaluator_results: dict[str, dict[str, Any]]
    node_records: list[dict[str, Any]]
    input_chunks: list[str]
    previous_output: str
    final_output: str
    last_evaluator_passed: bool
    observations: list[dict[str, Any]]
    react_steps: int
    react_done: bool
    pending_action: dict[str, str]


@dataclass
class LoadedCausalLm:
    tokenizer: Any
    model: Any
    input_device: Any


@dataclass(frozen=True)
class GenerationResult:
    text: str
    input_token_count: int | None = None
    output_token_count: int | None = None
    total_token_count: int | None = None


class LocalQwenTextGenerationBackend:
    def __init__(self) -> None:
        self.loaded: dict[tuple[str, tuple[str, ...], str], LoadedCausalLm] = {}
        self.generation_locks: dict[tuple[str, tuple[str, ...], str], Lock] = {}
        self.devices: tuple[str, ...] = ()

    def preload(self, nodes: Sequence[WorkflowNodeConfig]) -> None:
        self.devices = tuple(collect_execution_devices(nodes))
        reset_cuda_peak_memory(self.devices)
        for node in nodes:
            if node.type not in MODEL_NODE_TYPES:
                continue
            execution = require_execution(node)
            key = execution_key(execution)
            if key not in self.loaded:
                self.loaded[key] = self.load_model(execution)
                self.generation_locks[key] = Lock()

    def generate(self, node: WorkflowNodeConfig, prompt: str) -> GenerationResult:
        execution = require_execution(node)
        key = execution_key(execution)
        if key not in self.loaded:
            raise WorkflowExecutionError(f"model is not preloaded: {node.name}")

        lock = self.generation_locks.setdefault(key, Lock())
        with lock:
            return self.generate_locked(node, execution, key, prompt)

    def generate_locked(
        self,
        node: WorkflowNodeConfig,
        execution: WorkflowExecutionConfig,
        key: tuple[str, tuple[str, ...], str],
        prompt: str,
    ) -> GenerationResult:
        import torch

        loaded = self.loaded[key]
        prompt_text = build_model_prompt(loaded.tokenizer, execution, prompt)
        tokenization_kwargs: dict[str, Any] = {"return_tensors": "pt"}
        if node.runtime is not None and node.runtime.sequence_length is not None:
            tokenization_kwargs["truncation"] = True
            tokenization_kwargs["max_length"] = node.runtime.sequence_length
        inputs = loaded.tokenizer(prompt_text, **tokenization_kwargs)
        inputs = {name: value.to(loaded.input_device) for name, value in inputs.items()}
        input_length = inputs["input_ids"].shape[-1]
        generation_kwargs = build_generation_kwargs(node)
        eos_token_id = getattr(loaded.tokenizer, "eos_token_id", None)
        if eos_token_id is not None and "pad_token_id" not in generation_kwargs:
            generation_kwargs["pad_token_id"] = eos_token_id
        with torch.inference_mode():
            output_ids = loaded.model.generate(**inputs, **generation_kwargs)
        generated_ids = output_ids[0][input_length:]
        output = loaded.tokenizer.decode(
            generated_ids,
            skip_special_tokens=True,
        ).strip()
        return GenerationResult(
            text=output,
            input_token_count=int(input_length),
            output_token_count=int(generated_ids.shape[-1]),
            total_token_count=int(output_ids.shape[-1]),
        )

    def runtime_metrics(self) -> dict[str, Any]:
        return {
            "loaded_model_count": len(self.loaded),
            "gpu_memory_mb": snapshot_cuda_memory(self.devices),
        }

    def load_model(self, execution: WorkflowExecutionConfig) -> LoadedCausalLm:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        model_path = Path(execution.model_path)
        if not model_path.is_dir():
            raise FileNotFoundError(f"model path does not exist: {model_path}")

        dtype = resolve_torch_dtype(execution.dtype)
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            local_files_only=True,
            trust_remote_code=True,
        )
        if len(execution.devices) == 1:
            device = execution.devices[0]
            model = cast(
                Any,
                AutoModelForCausalLM.from_pretrained(
                    model_path,
                    dtype=dtype,
                    local_files_only=True,
                    trust_remote_code=True,
                ),
            )
            model.to(device)
            model.eval()
            return LoadedCausalLm(tokenizer=tokenizer, model=model, input_device=device)

        model = cast(
            Any,
            AutoModelForCausalLM.from_pretrained(
                model_path,
                dtype=dtype,
                local_files_only=True,
                trust_remote_code=True,
                device_map="auto",
                max_memory=build_max_memory(execution.devices),
            ),
        )
        model.eval()
        input_device = next(model.parameters()).device
        return LoadedCausalLm(
            tokenizer=tokenizer,
            model=model,
            input_device=input_device,
        )


def resolve_torch_dtype(dtype_name: str) -> Any:
    import torch

    dtype_by_name = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    if dtype_name not in dtype_by_name:
        raise ValueError(f"unsupported execution dtype: {dtype_name}")
    return dtype_by_name[dtype_name]


def build_max_memory(devices: Sequence[str]) -> dict[int | str, str]:
    import torch

    selected_indices = {cuda_device_index(device) for device in devices}
    max_memory: dict[int | str, str] = {}
    for device_index in range(torch.cuda.device_count()):
        if device_index not in selected_indices:
            max_memory[device_index] = "0GiB"
            continue
        free_bytes, _total_bytes = torch.cuda.mem_get_info(device_index)
        free_gib = max(1, int(free_bytes // (1024**3)))
        max_memory[device_index] = f"{free_gib}GiB"
    max_memory["cpu"] = "0GiB"
    return max_memory


def execution_key(
    execution: WorkflowExecutionConfig,
) -> tuple[str, tuple[str, ...], str]:
    return (execution.model_path, tuple(execution.devices), execution.dtype)


def build_generation_kwargs(node: WorkflowNodeConfig) -> dict[str, Any]:
    execution = require_execution(node)
    max_new_tokens = execution.max_new_tokens
    if max_new_tokens is None and node.runtime is not None:
        max_new_tokens = node.runtime.decode_max_output_length
    generation_kwargs: dict[str, Any] = {
        "max_new_tokens": max_new_tokens or DEFAULT_MAX_NEW_TOKENS,
        "do_sample": execution.do_sample,
    }
    if execution.do_sample and execution.temperature is not None:
        generation_kwargs["temperature"] = execution.temperature
    return generation_kwargs


def build_model_prompt(
    tokenizer: Any,
    execution: WorkflowExecutionConfig,
    prompt: str,
) -> str:
    if not execution.use_chat_template or not hasattr(tokenizer, "apply_chat_template"):
        return prompt
    messages = [{"role": "user", "content": prompt}]
    try:
        return str(
            tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=execution.enable_thinking,
            )
        )
    except TypeError:
        return str(
            tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        )


def collect_execution_devices(nodes: Sequence[WorkflowNodeConfig]) -> list[str]:
    devices: list[str] = []
    seen_devices: set[str] = set()
    for node in nodes:
        if node.type not in MODEL_NODE_TYPES:
            continue
        execution = require_execution(node)
        for device in execution.devices:
            if device in seen_devices:
                continue
            seen_devices.add(device)
            devices.append(device)
    return devices


def reset_cuda_peak_memory(devices: Sequence[str]) -> None:
    if not devices:
        return
    import torch

    for device in devices:
        device_index = cuda_device_index(device)
        torch.cuda.set_device(device_index)
        tensor = torch.empty((), device=device)
        del tensor
        torch.cuda.reset_peak_memory_stats(device_index)


def snapshot_cuda_memory(devices: Sequence[str]) -> dict[str, dict[str, float]]:
    if not devices:
        return {}
    import torch

    memory_by_device: dict[str, dict[str, float]] = {}
    for device in devices:
        device_index = cuda_device_index(device)
        free_bytes, total_bytes = torch.cuda.mem_get_info(device_index)
        memory_by_device[device] = {
            "current_used_mb": bytes_to_mb(total_bytes - free_bytes),
            "peak_allocated_mb": bytes_to_mb(
                torch.cuda.max_memory_allocated(device_index)
            ),
            "peak_reserved_mb": bytes_to_mb(
                torch.cuda.max_memory_reserved(device_index)
            ),
        }
    return memory_by_device


def cuda_device_index(device: str) -> int:
    return int(device.split(":", maxsplit=1)[1])


def bytes_to_mb(value: int) -> float:
    return value / (1024**2)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a Workflow YAML on supported datasets and print JSONL results."
        ),
    )
    parser.add_argument("--workflow", required=True, help="Path to one Workflow YAML.")
    parser.add_argument(
        "--dataset",
        required=True,
        choices=("gsm8k", "mbpp", "multi_news", "qmsum"),
    )
    parser.add_argument("--split", default="test")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mode", choices=("dag", "react", "parallel"), default="dag")
    parser.add_argument(
        "--resume-from",
        type=Path,
        default=None,
        help="Existing workflow JSONL output used to skip completed samples.",
    )
    parser.add_argument(
        "--sample-ids",
        type=Path,
        default=None,
        help="JSONL file containing sample_id values to run in listed order.",
    )
    return parser.parse_args(argv)


def main(
    argv: list[str] | None = None,
    *,
    backend: TextGenerationBackend | None = None,
    stdout: TextIO | None = None,
) -> int:
    args = parse_args(argv)
    workflow = load_workflow(args.workflow)
    loaded_samples = load_task_samples(args.dataset, args.split)
    samples = select_samples_by_id(loaded_samples, args.sample_ids)
    samples = select_samples(samples, limit=args.limit, seed=args.seed)
    existing_records = load_resume_sample_records(args.resume_from)
    selected_sample_ids = {sample.sample_id for sample in samples}
    existing_records = [
        record
        for record in existing_records
        if record["sample_id"] in selected_sample_ids
    ]
    run_workflow_evaluation(
        workflow=workflow,
        dataset=args.dataset,
        samples=samples,
        mode=args.mode,
        backend=backend or LocalQwenTextGenerationBackend(),
        stdout=stdout or sys.stdout,
        existing_records=existing_records,
    )
    return 0


def load_task_samples(dataset: WorkflowDataset, split: str) -> list[TaskSample]:
    if dataset == "gsm8k":
        if split not in {"train", "test"}:
            raise ValueError(f"unsupported GSM8K split: {split}")
        return load_gsm8k_split(Path(f"dataset/gsm8k/data/{split}.jsonl"), split=split)

    if dataset == "mbpp":
        samples = load_mbpp_samples(Path("dataset/mbpp/sanitized-mbpp.json"))
        filtered = [sample for sample in samples if sample.split == split]
        if not filtered:
            raise ValueError(f"unsupported or empty MBPP split: {split}")
        return filtered

    if split not in {"train", "val", "test"}:
        raise ValueError(f"unsupported summarization split: {split}")
    if dataset == "multi_news":
        return load_multi_news_split(Path("dataset/multi_news/data"), split=split)
    return load_qmsum_split(Path("dataset/QMSum/data/ALL"), split=split)


def select_samples(
    samples: Sequence[TaskSample],
    *,
    limit: int | None,
    seed: int,
) -> list[TaskSample]:
    selected_samples = list(samples)
    if limit is None:
        return selected_samples
    if limit <= 0:
        raise ValueError("limit must be positive")
    if limit > len(selected_samples):
        raise ValueError("limit exceeds available samples")
    return random.Random(seed).sample(selected_samples, limit)


def select_samples_by_id(
    samples: Sequence[TaskSample],
    sample_ids_path: Path | None,
) -> list[TaskSample]:
    if sample_ids_path is None:
        return list(samples)
    sample_ids = load_sample_ids(sample_ids_path)
    samples_by_id = {sample.sample_id: sample for sample in samples}
    missing = [sample_id for sample_id in sample_ids if sample_id not in samples_by_id]
    if missing:
        raise ValueError(f"sample ids do not exist: {', '.join(missing)}")
    return [samples_by_id[sample_id] for sample_id in sample_ids]


def load_sample_ids(path: Path) -> list[str]:
    sample_ids: list[str] = []
    lines = path.read_text(encoding="utf-8").splitlines()
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        payload = json.loads(line)
        if isinstance(payload, str):
            sample_id = payload
        elif isinstance(payload, dict) and isinstance(payload.get("sample_id"), str):
            sample_id = payload["sample_id"]
        else:
            raise ValueError(f"sample id file has invalid row at line {line_number}")
        sample_ids.append(sample_id)
    if not sample_ids:
        raise ValueError("sample id file is empty")
    return sample_ids


def load_resume_sample_records(path: Path | None) -> list[dict[str, Any]]:
    if path is None or not path.exists():
        return []
    records_by_sample_id: dict[str, dict[str, Any]] = {}
    lines = path.read_text(encoding="utf-8").splitlines()
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        record = json.loads(line)
        record_type = record.get("record_type")
        if record_type == "summary":
            continue
        if record_type != "sample":
            raise ValueError(
                f"resume JSONL has non-workflow record at line {line_number}"
            )
        sample_id = record.get("sample_id")
        if not isinstance(sample_id, str):
            raise ValueError(
                f"resume JSONL sample is missing sample_id at line {line_number}"
            )
        records_by_sample_id[sample_id] = record
    return list(records_by_sample_id.values())


def run_workflow_evaluation(
    *,
    workflow: Workflow,
    dataset: WorkflowDataset,
    samples: Sequence[TaskSample],
    mode: WorkflowEvalMode,
    backend: TextGenerationBackend,
    stdout: TextIO,
    summary_judge: SummaryJudge | None = None,
    existing_records: Sequence[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    validate_executable_workflow(workflow, dataset, mode)
    if summary_judge is None and workflow_uses_summary_judge(workflow):
        summary_judge = OpenRouterSummaryJudge.from_env()
    preload_started_at = time.perf_counter()
    backend.preload(model_nodes(workflow))
    preload_duration_sec = time.perf_counter() - preload_started_at
    graph = None
    if mode == "dag":
        graph = build_dag_graph(workflow, dataset, backend, summary_judge)
    elif mode == "react":
        graph = build_react_graph(workflow, dataset, backend, summary_judge)

    records = list(existing_records or [])
    completed_sample_ids = {
        str(record["sample_id"])
        for record in records
        if record.get("record_type") == "sample"
    }
    for sample in samples:
        if sample.sample_id in completed_sample_ids:
            continue
        if mode == "parallel":
            record = run_parallel_sample(
                workflow,
                sample,
                dataset,
                backend,
                summary_judge,
            )
        else:
            assert graph is not None
            record = run_sample(graph, sample, dataset, mode)
        records.append(record)
        stdout.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        stdout.flush()

    summary = build_summary_record(
        dataset,
        mode,
        records,
        preload_duration_sec=preload_duration_sec,
        backend_metrics=backend_runtime_metrics(backend),
    )
    stdout.write(json.dumps(summary, ensure_ascii=False, sort_keys=True) + "\n")
    stdout.flush()
    return summary


def validate_executable_workflow(
    workflow: Workflow,
    dataset: WorkflowDataset,
    mode: WorkflowEvalMode,
) -> None:
    validate_workflow(workflow)
    expected_evaluator_tasks = DATASET_EVALUATOR_TASKS[dataset]
    evaluator_count = 0
    for node in workflow.nodes:
        if node.type in MODEL_NODE_TYPES:
            if node.execution is None:
                raise ValueError(f"model node must define execution: {node.name}")
            if node.prompt_template is None:
                raise ValueError(f"model node must define prompt_template: {node.name}")
        if node.type == "evaluator":
            evaluator_count += int(node.task in expected_evaluator_tasks)
            if node.task not in expected_evaluator_tasks:
                raise ValueError(f"evaluator task does not match dataset: {node.name}")
    if evaluator_count == 0:
        raise ValueError("workflow must define a matching evaluator node")
    if mode == "react" and find_react_controller(workflow) is None:
        raise ValueError("react mode requires an agent node with execution")
    if mode == "dag":
        validate_serial_dag(workflow)
    if mode == "parallel":
        validate_parallel_workflow(workflow)


def validate_serial_dag(workflow: Workflow) -> None:
    outgoing = build_outgoing_edges(workflow)
    node_map = workflow.node_map()
    for node in workflow.nodes:
        if node.type == "output":
            continue
        edges = outgoing.get(node.name, [])
        if node.type == "evaluator" and any(edge.attributes for edge in edges):
            continue
        if len(edges) > 1:
            raise ValueError(f"dag mode supports one outgoing edge: {node.name}")
        for edge in edges:
            if edge.attributes:
                raise ValueError(
                    f"conditional edge requires evaluator source: {node.name}"
                )
            if node_map[edge.target].type == "input":
                raise ValueError("input nodes cannot be edge targets")


def validate_parallel_workflow(workflow: Workflow) -> None:
    outgoing = build_outgoing_edges(workflow)
    if not outgoing.get("input"):
        raise ValueError("parallel mode requires at least one input edge")
    for edge in workflow.edges:
        if edge.attributes:
            raise ValueError("parallel mode does not support conditional edges")


def model_nodes(workflow: Workflow) -> list[WorkflowNodeConfig]:
    return [node for node in workflow.nodes if node.type in MODEL_NODE_TYPES]


def build_dag_graph(
    workflow: Workflow,
    dataset: WorkflowDataset,
    backend: TextGenerationBackend,
    summary_judge: SummaryJudge | None = None,
) -> Any:
    outgoing = build_outgoing_edges(workflow)
    node_map = workflow.node_map()
    builder = StateGraph(cast(Any, WorkflowRunState))
    for node in workflow.nodes:
        if node.type in {"input", "output"}:
            continue
        builder.add_node(
            node.name,
            build_dag_node(node, dataset, backend, summary_judge),
        )

    input_edges = outgoing.get("input", [])
    if len(input_edges) != 1:
        raise ValueError("dag mode requires exactly one input edge")
    builder.add_edge(START, graph_target(input_edges[0].target, node_map))

    for node in workflow.nodes:
        if node.type in {"input", "output"}:
            continue
        edges = outgoing.get(node.name, [])
        if not edges:
            builder.add_edge(node.name, END)
            continue
        if node.type == "evaluator" and any(edge.attributes for edge in edges):
            builder.add_conditional_edges(
                node.name,
                build_evaluator_router(edges, node_map),
            )
            continue
        builder.add_edge(node.name, graph_target(edges[0].target, node_map))
    return builder.compile()


def build_dag_node(
    node: WorkflowNodeConfig,
    dataset: WorkflowDataset,
    backend: TextGenerationBackend,
    summary_judge: SummaryJudge | None,
) -> Any:
    def run_node(state: WorkflowRunState) -> dict[str, Any]:
        if node.type in MODEL_NODE_TYPES:
            return run_model_node(node, state, backend)
        if node.type == "evaluator":
            return run_evaluator_node(node, state, dataset, summary_judge)
        raise WorkflowExecutionError(f"unsupported DAG node type: {node.type}")

    return run_node


def build_evaluator_router(
    edges: Sequence[Any],
    node_map: dict[str, WorkflowNodeConfig],
) -> Any:
    route_by_condition = {
        str(edge.attributes.get("condition", "default")): graph_target(
            edge.target,
            node_map,
        )
        for edge in edges
    }

    def route(state: WorkflowRunState) -> str:
        condition = "passed" if state.get("last_evaluator_passed") else "failed"
        return route_by_condition.get(condition) or route_by_condition.get(
            "default",
            END,
        )

    return route


def graph_target(target_name: str, node_map: dict[str, WorkflowNodeConfig]) -> str:
    if node_map[target_name].type == "output":
        return END
    return target_name


def run_model_node(
    node: WorkflowNodeConfig,
    state: WorkflowRunState,
    backend: TextGenerationBackend,
) -> dict[str, Any]:
    started_at = time.perf_counter()
    prompt = render_prompt(node, state)
    generation = backend.generate(node, prompt)
    output = generation.text
    duration_sec = time.perf_counter() - started_at
    node_outputs = dict(state.get("node_outputs", {}))
    node_outputs[node.name] = output
    node_records = list(state.get("node_records", []))
    node_record: dict[str, Any] = {
        "node_name": node.name,
        "type": node.type,
        "duration_sec": duration_sec,
        "output": output,
        "input_token_count": generation.input_token_count,
        "output_token_count": generation.output_token_count,
        "total_token_count": generation.total_token_count,
    }
    if generation.output_token_count is not None and duration_sec > 0:
        node_record["output_tokens_per_sec"] = (
            generation.output_token_count / duration_sec
        )
    node_records.append(node_record)
    return {
        "node_outputs": node_outputs,
        "node_records": node_records,
        "previous_output": output,
        "final_output": output,
    }


def run_evaluator_node(
    node: WorkflowNodeConfig,
    state: WorkflowRunState,
    dataset: WorkflowDataset,
    summary_judge: SummaryJudge | None = None,
) -> dict[str, Any]:
    started_at = time.perf_counter()
    output = state.get("previous_output", "")
    result, passed = evaluate_output(
        node,
        state["sample"],
        output,
        dataset,
        summary_judge,
    )
    duration_sec = time.perf_counter() - started_at
    evaluator_results = dict(state.get("evaluator_results", {}))
    evaluator_results[node.name] = result
    node_records = list(state.get("node_records", []))
    node_records.append(
        {
            "node_name": node.name,
            "type": node.type,
            "duration_sec": duration_sec,
            "result": result,
            "passed": passed,
        }
    )
    return {
        "evaluator_results": evaluator_results,
        "node_records": node_records,
        "last_evaluator_passed": passed,
    }


def evaluate_output(
    node: WorkflowNodeConfig,
    sample: TaskSample,
    output: str,
    dataset: WorkflowDataset,
    summary_judge: SummaryJudge | None = None,
) -> tuple[dict[str, Any], bool]:
    expected_tasks = DATASET_EVALUATOR_TASKS[dataset]
    if node.task not in expected_tasks:
        raise WorkflowExecutionError(
            f"evaluator task does not match dataset: {node.name}"
        )
    if dataset == "gsm8k":
        if sample.gold_answer is None:
            raise WorkflowExecutionError(
                f"GSM8K sample is missing gold_answer: {sample.sample_id}"
            )
        result = evaluate_gsm8k(output, sample.gold_answer)
        return result.model_dump(), result.exact_match

    if dataset == "mbpp":
        result = evaluate_mbpp(output, sample.test_imports, sample.test_list)
        return result.model_dump(), result.passed

    if sample.gold_answer is None:
        raise WorkflowExecutionError(
            f"summary sample is missing gold_answer: {sample.sample_id}"
        )
    if node.task == "summary_rouge":
        result = evaluate_summary_rouge(output, sample.gold_answer)
        return result.model_dump(), result.passed
    if summary_judge is None:
        raise WorkflowExecutionError("summary_llm_judge requires a summary judge")
    result = evaluate_summary_with_judge(sample, output, summary_judge)
    return result.model_dump(), result.passed


def render_prompt(node: WorkflowNodeConfig, state: WorkflowRunState) -> str:
    if node.prompt_template is None:
        raise WorkflowExecutionError(f"node is missing prompt_template: {node.name}")
    sample = state["sample"]
    input_chunks = state.get("input_chunks") or build_input_chunks(sample.input_text)
    values = {
        "dataset": state["dataset"],
        "sample_id": sample.sample_id,
        "split": sample.split,
        "input_text": sample.input_text,
        "query": str(sample.metadata.get("query", "")),
        "gold_answer": sample.gold_answer or "",
        "raw_answer": sample.raw_answer or "",
        "metadata_json": json.dumps(
            sample.metadata,
            ensure_ascii=False,
            sort_keys=True,
        ),
        "previous_output": state.get("previous_output", ""),
        "node_outputs_json": json.dumps(
            state.get("node_outputs", {}),
            ensure_ascii=False,
            sort_keys=True,
        ),
        "evaluator_results_json": json.dumps(
            state.get("evaluator_results", {}),
            ensure_ascii=False,
            sort_keys=True,
        ),
        "test_imports": "\n".join(sample.test_imports),
        "test_imports_json": json.dumps(sample.test_imports, ensure_ascii=False),
        "test_list": "\n".join(sample.test_list),
        "test_list_json": json.dumps(sample.test_list, ensure_ascii=False),
    }
    values.update(
        {
            f"chunk_{index}_text": chunk
            for index, chunk in enumerate(input_chunks)
        }
    )
    try:
        return node.prompt_template.format(**values)
    except KeyError as exc:
        raise WorkflowExecutionError(f"prompt variable is missing: {exc}") from exc


def build_input_chunks(
    input_text: str,
    chunk_count: int = DEFAULT_INPUT_CHUNK_COUNT,
) -> list[str]:
    if chunk_count <= 0:
        raise ValueError("chunk_count must be positive")
    paragraphs = [part.strip() for part in input_text.split("\n\n") if part.strip()]
    if len(paragraphs) >= chunk_count:
        base_size, remainder = divmod(len(paragraphs), chunk_count)
        groups = []
        start = 0
        for index in range(chunk_count):
            end = start + base_size + int(index < remainder)
            groups.append(paragraphs[start:end])
            start = end
        return ["\n\n".join(group) for group in groups]

    chunk_size = max(1, (len(input_text) + chunk_count - 1) // chunk_count)
    chunks = [
        input_text[index * chunk_size : (index + 1) * chunk_size].strip()
        for index in range(chunk_count)
    ]
    return [chunk or input_text.strip() for chunk in chunks]


def build_react_graph(
    workflow: Workflow,
    dataset: WorkflowDataset,
    backend: TextGenerationBackend,
    summary_judge: SummaryJudge | None = None,
) -> Any:
    controller = find_react_controller(workflow)
    if controller is None:
        raise ValueError("react mode requires an agent node with execution")
    node_map = {
        node.name: node
        for node in workflow.nodes
        if node.type in MODEL_NODE_TYPES or node.type == "evaluator"
    }
    builder = StateGraph(cast(Any, WorkflowRunState))
    builder.add_node(
        "controller",
        build_react_controller(controller, workflow, backend),
    )
    builder.add_node(
        "execute_action",
        build_react_action_executor(node_map, dataset, backend, summary_judge),
    )
    builder.add_edge(START, "controller")
    builder.add_conditional_edges(
        "controller",
        route_react_controller,
        {"done": END, "action": "execute_action"},
    )
    builder.add_edge("execute_action", "controller")
    return builder.compile()


def find_react_controller(workflow: Workflow) -> WorkflowNodeConfig | None:
    for node in workflow.nodes:
        if node.type == "agent" and node.execution is not None:
            return node
    return None


def build_react_controller(
    controller: WorkflowNodeConfig,
    workflow: Workflow,
    backend: TextGenerationBackend,
) -> Any:
    def run_controller(state: WorkflowRunState) -> dict[str, Any]:
        steps = int(state.get("react_steps", 0))
        if steps >= DEFAULT_REACT_MAX_STEPS:
            raise WorkflowExecutionError("react max steps exceeded")
        prompt = build_react_prompt(controller, workflow, state)
        raw_output = backend.generate(controller, prompt).text
        action = parse_react_action(raw_output)
        if "final" in action:
            final_output = action["final"]
            return {
                "react_done": True,
                "final_output": final_output,
                "previous_output": final_output,
                "react_steps": steps + 1,
            }
        return {
            "pending_action": action,
            "react_done": False,
            "react_steps": steps + 1,
        }

    return run_controller


def build_react_prompt(
    controller: WorkflowNodeConfig,
    workflow: Workflow,
    state: WorkflowRunState,
) -> str:
    base_prompt = render_prompt(controller, state)
    tools = [
        {
            "name": node.name,
            "type": node.type,
            "task": node.task,
            "description": node.description,
        }
        for node in workflow.nodes
        if node.name != controller.name
        and (node.type in MODEL_NODE_TYPES or node.type == "evaluator")
    ]
    protocol = {
        "action": {"action": "node_name", "input": "text for that node"},
        "final": {"final": "final answer text"},
    }
    return "\n\n".join(
        [
            base_prompt,
            f"Available actions: {json.dumps(tools, ensure_ascii=False)}",
            (
                "Previous observations: "
                f"{json.dumps(state.get('observations', []), ensure_ascii=False)}"
            ),
            (
                "Return exactly one JSON object. "
                f"Protocol: {json.dumps(protocol, ensure_ascii=False)}"
            ),
        ]
    )


def parse_react_action(raw_output: str) -> dict[str, str]:
    payload = json.loads(raw_output.strip())
    if not isinstance(payload, dict):
        raise WorkflowExecutionError("react output must be a JSON object")
    final = payload.get("final")
    if isinstance(final, str):
        return {"final": final}
    action = payload.get("action")
    if not isinstance(action, str) or not action.strip():
        raise WorkflowExecutionError("react output must contain action or final")
    action_input = payload.get("input", "")
    if not isinstance(action_input, str):
        raise WorkflowExecutionError("react action input must be a string")
    return {"action": action, "input": action_input}


def route_react_controller(state: WorkflowRunState) -> str:
    return "done" if state.get("react_done") else "action"


def build_react_action_executor(
    node_map: dict[str, WorkflowNodeConfig],
    dataset: WorkflowDataset,
    backend: TextGenerationBackend,
    summary_judge: SummaryJudge | None = None,
) -> Any:
    def execute_action(state: WorkflowRunState) -> dict[str, Any]:
        action = state.get("pending_action", {})
        node_name = action.get("action")
        if node_name not in node_map:
            raise WorkflowExecutionError(f"react action node is undefined: {node_name}")
        node = node_map[node_name]
        node_input = action.get("input", "")
        node_state = cast(WorkflowRunState, dict(state))
        if node_input:
            node_state["previous_output"] = node_input
        if node.type in MODEL_NODE_TYPES:
            update = run_model_node(node, node_state, backend)
            observation_output = update["previous_output"]
        else:
            update = run_evaluator_node(node, node_state, dataset, summary_judge)
            observation_output = json.dumps(
                update["evaluator_results"][node.name],
                ensure_ascii=False,
            )
        observations = list(state.get("observations", []))
        observations.append(
            {
                "node": node.name,
                "type": node.type,
                "output": observation_output,
            }
        )
        update["observations"] = observations
        return update

    return execute_action


def initial_run_state(
    sample: TaskSample,
    dataset: WorkflowDataset,
    *,
    chunk_count: int = DEFAULT_INPUT_CHUNK_COUNT,
) -> WorkflowRunState:
    return {
        "sample": sample,
        "dataset": dataset,
        "node_outputs": {},
        "evaluator_results": {},
        "node_records": [],
        "input_chunks": build_input_chunks(sample.input_text, chunk_count=chunk_count),
        "previous_output": "",
        "final_output": "",
        "observations": [],
        "react_steps": 0,
        "react_done": False,
    }


def run_parallel_sample(
    workflow: Workflow,
    sample: TaskSample,
    dataset: WorkflowDataset,
    backend: TextGenerationBackend,
    summary_judge: SummaryJudge | None = None,
) -> dict[str, Any]:
    started_at = time.perf_counter()
    node_map = workflow.node_map()
    incoming = build_incoming_edges(workflow)
    state = initial_run_state(
        sample,
        dataset,
        chunk_count=infer_workflow_chunk_count(workflow),
    )
    completed = {"input"}
    pending = {node.name for node in workflow.nodes if node.type != "input"}
    workflow_order = {node.name: index for index, node in enumerate(workflow.nodes)}

    while pending:
        ready_outputs = [
            name
            for name in pending
            if node_map[name].type == "output"
            and all(source in completed for source in incoming.get(name, []))
        ]
        if ready_outputs:
            completed.update(ready_outputs)
            pending.difference_update(ready_outputs)
            continue

        ready_nodes = [
            node_map[name]
            for name in pending
            if node_map[name].type != "output"
            and all(source in completed for source in incoming.get(name, []))
        ]
        ready_nodes.sort(key=lambda node: workflow_order[node.name])
        if not ready_nodes:
            raise WorkflowExecutionError("parallel workflow has no runnable nodes")

        model_ready = [node for node in ready_nodes if node.type in MODEL_NODE_TYPES]
        evaluator_ready = [node for node in ready_nodes if node.type == "evaluator"]
        unsupported = [
            node.name
            for node in ready_nodes
            if node.type not in MODEL_NODE_TYPES and node.type != "evaluator"
        ]
        if unsupported:
            raise WorkflowExecutionError(
                f"parallel workflow contains unsupported nodes: {unsupported}"
            )

        for update in run_parallel_model_nodes(model_ready, state, backend):
            merge_state_update(state, update)
        for node in evaluator_ready:
            node_state = isolated_record_state(state)
            update = run_evaluator_node(node, node_state, dataset, summary_judge)
            merge_state_update(state, update)

        ready_names = {node.name for node in ready_nodes}
        completed.update(ready_names)
        pending.difference_update(ready_names)

    duration_sec = time.perf_counter() - started_at
    evaluator_record = final_evaluator_record(state.get("node_records", []))
    passed = bool(evaluator_record.get("passed")) if evaluator_record else False
    return {
        "record_type": "sample",
        "dataset": dataset,
        "mode": "parallel",
        "sample_id": sample.sample_id,
        "split": sample.split,
        "passed": passed,
        "duration_sec": duration_sec,
        "final_output": state.get("final_output", ""),
        "node_outputs": state.get("node_outputs", {}),
        "evaluator_results": state.get("evaluator_results", {}),
        "node_records": state.get("node_records", []),
    }


def run_parallel_model_nodes(
    nodes: Sequence[WorkflowNodeConfig],
    state: WorkflowRunState,
    backend: TextGenerationBackend,
) -> list[dict[str, Any]]:
    if not nodes:
        return []
    if len(nodes) == 1:
        return [run_model_node(nodes[0], isolated_record_state(state), backend)]
    with ThreadPoolExecutor(max_workers=len(nodes)) as executor:
        futures = [
            executor.submit(
                run_model_node,
                node,
                isolated_record_state(state),
                backend,
            )
            for node in nodes
        ]
        return [future.result() for future in futures]


def isolated_record_state(state: WorkflowRunState) -> WorkflowRunState:
    node_state = cast(WorkflowRunState, dict(state))
    node_state["node_records"] = []
    return node_state


def merge_state_update(state: WorkflowRunState, update: dict[str, Any]) -> None:
    if "node_outputs" in update:
        node_outputs = dict(state.get("node_outputs", {}))
        node_outputs.update(update["node_outputs"])
        state["node_outputs"] = node_outputs
    if "evaluator_results" in update:
        evaluator_results = dict(state.get("evaluator_results", {}))
        evaluator_results.update(update["evaluator_results"])
        state["evaluator_results"] = evaluator_results
    if "node_records" in update:
        state["node_records"] = list(state.get("node_records", [])) + list(
            update["node_records"]
        )
    if "previous_output" in update:
        state["previous_output"] = update["previous_output"]
    if "final_output" in update:
        state["final_output"] = update["final_output"]
    if "last_evaluator_passed" in update:
        state["last_evaluator_passed"] = update["last_evaluator_passed"]
    if "observations" in update:
        state["observations"] = update["observations"]
    if "react_steps" in update:
        state["react_steps"] = update["react_steps"]
    if "react_done" in update:
        state["react_done"] = update["react_done"]
    if "pending_action" in update:
        state["pending_action"] = update["pending_action"]


def run_sample(
    graph: Any,
    sample: TaskSample,
    dataset: WorkflowDataset,
    mode: WorkflowEvalMode,
) -> dict[str, Any]:
    started_at = time.perf_counter()
    initial_state = initial_run_state(sample, dataset)
    result = graph.invoke(initial_state)
    duration_sec = time.perf_counter() - started_at
    evaluator_record = final_evaluator_record(result.get("node_records", []))
    passed = bool(evaluator_record.get("passed")) if evaluator_record else False
    return {
        "record_type": "sample",
        "dataset": dataset,
        "mode": mode,
        "sample_id": sample.sample_id,
        "split": sample.split,
        "passed": passed,
        "duration_sec": duration_sec,
        "final_output": result.get("final_output", ""),
        "node_outputs": result.get("node_outputs", {}),
        "evaluator_results": result.get("evaluator_results", {}),
        "node_records": result.get("node_records", []),
    }


def final_evaluator_record(
    node_records: Sequence[dict[str, Any]],
) -> dict[str, Any] | None:
    for record in reversed(node_records):
        if record.get("type") == "evaluator":
            return record
    return None


def build_summary_record(
    dataset: WorkflowDataset,
    mode: WorkflowEvalMode,
    records: Sequence[dict[str, Any]],
    *,
    preload_duration_sec: float = 0.0,
    backend_metrics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    sample_count = len(records)
    passed_count = sum(1 for record in records if record["passed"])
    durations = [float(record["duration_sec"]) for record in records]
    error_counts: dict[str, int] = {}
    for record in records:
        error_type = final_error_type(record)
        if error_type:
            error_counts[error_type] = error_counts.get(error_type, 0) + 1
    model_records = [
        item
        for record in records
        for item in record["node_records"]
        if item.get("type") in MODEL_NODE_TYPES
    ]
    evaluator_records = [
        item
        for record in records
        for item in record["node_records"]
        if item.get("type") == "evaluator"
    ]
    model_durations = [float(item["duration_sec"]) for item in model_records]
    total_model_duration_sec = sum(model_durations)
    output_token_count = sum(
        int(item["output_token_count"])
        for item in model_records
        if item.get("output_token_count") is not None
    )
    summary = {
        "record_type": "summary",
        "dataset": dataset,
        "mode": mode,
        "sample_count": sample_count,
        "passed_count": passed_count,
        "accuracy": passed_count / sample_count if sample_count else 0.0,
        "total_duration_sec": sum(durations),
        "avg_duration_sec": sum(durations) / sample_count if sample_count else 0.0,
        "duration_stats_sec": build_numeric_stats(durations),
        "preload_duration_sec": preload_duration_sec,
        "model_call_count": len(model_records),
        "evaluator_call_count": len(evaluator_records),
        "total_model_duration_sec": total_model_duration_sec,
        "avg_model_duration_sec": (
            total_model_duration_sec / len(model_records) if model_records else 0.0
        ),
        "total_output_tokens": output_token_count,
        "output_tokens_per_sec": (
            output_token_count / total_model_duration_sec
            if total_model_duration_sec > 0
            else 0.0
        ),
        "node_stats": build_node_stats(model_records + evaluator_records),
        "error_counts": error_counts,
    }
    summary.update(build_quality_metric_summary(evaluator_records))
    if backend_metrics:
        summary["backend_metrics"] = backend_metrics
    return summary


def build_quality_metric_summary(
    evaluator_records: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    results = [
        record.get("result", {})
        for record in evaluator_records
        if isinstance(record.get("result"), dict)
    ]
    metric_values = {
        "avg_rouge1": collect_float_values(results, "rouge1"),
        "avg_rouge2": collect_float_values(results, "rouge2"),
        "avg_rougeL": collect_float_values(results, "rouge_l"),
        "avg_llm_score": collect_float_values(results, "llm_score"),
        "avg_coverage": collect_float_values(results, "coverage"),
        "avg_relevance": collect_float_values(results, "relevance"),
        "avg_coherence": collect_float_values(results, "coherence"),
        "avg_faithfulness": collect_float_values(results, "faithfulness"),
    }
    summary = {
        key: sum(values) / len(values)
        for key, values in metric_values.items()
        if values
    }
    llm_passed_values = [
        result["llm_passed"]
        for result in results
        if isinstance(result.get("llm_passed"), bool)
    ]
    if llm_passed_values:
        llm_passed_count = sum(bool(value) for value in llm_passed_values)
        summary["llm_pass_rate"] = llm_passed_count / len(llm_passed_values)
    return summary


def collect_float_values(
    results: Sequence[dict[str, Any]],
    key: str,
) -> list[float]:
    values: list[float] = []
    for result in results:
        value = result.get(key)
        if isinstance(value, int | float) and not isinstance(value, bool):
            values.append(float(value))
    return values


def build_numeric_stats(values: Sequence[float]) -> dict[str, float]:
    if not values:
        return {
            "min": 0.0,
            "p50": 0.0,
            "p90": 0.0,
            "max": 0.0,
        }
    sorted_values = sorted(values)
    return {
        "min": sorted_values[0],
        "p50": percentile(sorted_values, 0.5),
        "p90": percentile(sorted_values, 0.9),
        "max": sorted_values[-1],
    }


def percentile(sorted_values: Sequence[float], fraction: float) -> float:
    if len(sorted_values) == 1:
        return sorted_values[0]
    index = round((len(sorted_values) - 1) * fraction)
    return sorted_values[index]


def build_node_stats(
    node_records: Sequence[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    stats: dict[str, dict[str, Any]] = {}
    for record in node_records:
        node_name = str(record["node_name"])
        node_stats = stats.setdefault(
            node_name,
            {
                "type": record["type"],
                "call_count": 0,
                "total_duration_sec": 0.0,
                "total_output_tokens": 0,
            },
        )
        node_stats["call_count"] += 1
        node_stats["total_duration_sec"] += float(record["duration_sec"])
        if record.get("output_token_count") is not None:
            node_stats["total_output_tokens"] += int(record["output_token_count"])
    for node_stats in stats.values():
        call_count = int(node_stats["call_count"])
        total_duration_sec = float(node_stats["total_duration_sec"])
        node_stats["avg_duration_sec"] = (
            total_duration_sec / call_count if call_count else 0.0
        )
        node_stats["output_tokens_per_sec"] = (
            int(node_stats["total_output_tokens"]) / total_duration_sec
            if total_duration_sec > 0
            else 0.0
        )
    return stats


def backend_runtime_metrics(backend: TextGenerationBackend) -> dict[str, Any]:
    runtime_metrics = getattr(backend, "runtime_metrics", None)
    if runtime_metrics is None:
        return {}
    metrics = runtime_metrics()
    return metrics if isinstance(metrics, dict) else {}


def final_error_type(record: dict[str, Any]) -> str | None:
    for item in reversed(record["node_records"]):
        if item.get("type") != "evaluator":
            continue
        result = item.get("result", {})
        if isinstance(result, dict):
            error_type = result.get("error_type")
            if isinstance(error_type, str):
                return error_type
        return None if item.get("passed") else "incorrect"
    return "missing_evaluator"


def build_outgoing_edges(workflow: Workflow) -> dict[str, list[Any]]:
    outgoing: dict[str, list[Any]] = {}
    for edge in workflow.edges:
        outgoing.setdefault(edge.source, []).append(edge)
    return outgoing


def infer_workflow_chunk_count(workflow: Workflow) -> int:
    chunk_indices = []
    for node in workflow.nodes:
        if node.prompt_template is None:
            continue
        chunk_indices.extend(
            int(match.group(1)) for match in CHUNK_PROMPT_PATTERN.finditer(
                node.prompt_template
            )
        )
    if not chunk_indices:
        return DEFAULT_INPUT_CHUNK_COUNT
    return max(chunk_indices) + 1


def build_incoming_edges(workflow: Workflow) -> dict[str, list[str]]:
    incoming: dict[str, list[str]] = {}
    for edge in workflow.edges:
        incoming.setdefault(edge.target, []).append(edge.source)
    return incoming


def workflow_uses_summary_judge(workflow: Workflow) -> bool:
    return any(
        node.type == "evaluator" and node.task == "summary_llm_judge"
        for node in workflow.nodes
    )


def require_execution(node: WorkflowNodeConfig) -> WorkflowExecutionConfig:
    if node.execution is None:
        raise WorkflowExecutionError(f"node is missing execution: {node.name}")
    return node.execution


if __name__ == "__main__":
    raise SystemExit(main())
