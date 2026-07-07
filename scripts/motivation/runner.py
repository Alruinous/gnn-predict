from __future__ import annotations

import gc
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import torch
from dotenv import load_dotenv

from dataset.mbpp import evaluate_mbpp
from dataset.schema import SummaryEvaluation, TaskSample
from dataset.summarization import (
    OpenRouterSummaryJudge,
    evaluate_summary_rouge,
    evaluate_summary_with_judge,
)
from scripts.motivation.common import TraceEvent, append_jsonl, event_dict
from scripts.motivation.llm import GenerationResult, LocalQwenGenerator
from scripts.motivation.sampling import split_evenly

QWEN3_4B = "/data/Models/Qwen/Qwen3-4B"
QWEN3_8B = "/data/Models/Qwen/Qwen3-8B"
QWEN3_14B = "/data/Models/Qwen/Qwen3-14B"


def run_qmsum_3way(
    samples: list[TaskSample],
    *,
    output_dir: Path,
    judge_mode: str,
    max_input_tokens: int,
    max_new_tokens: int,
) -> None:
    trace_path = output_dir / "qmsum_3way_trace.jsonl"
    result_path = output_dir / "qmsum_3way_results.jsonl"
    trace_path.unlink(missing_ok=True)
    result_path.unlink(missing_ok=True)
    judge = build_summary_judge(judge_mode)
    generators = {
        "chunk_0": LocalQwenGenerator(
            model_name="qwen3-4b",
            model_path=QWEN3_4B,
            device="cuda:0",
            max_input_tokens=max_input_tokens,
            max_new_tokens=max_new_tokens,
        ),
        "chunk_1": LocalQwenGenerator(
            model_name="qwen3-4b",
            model_path=QWEN3_4B,
            device="cuda:1",
            max_input_tokens=max_input_tokens,
            max_new_tokens=max_new_tokens,
        ),
        "chunk_2": LocalQwenGenerator(
            model_name="qwen3-4b",
            model_path=QWEN3_4B,
            device="cuda:2",
            max_input_tokens=max_input_tokens,
            max_new_tokens=max_new_tokens,
        ),
        "merge": LocalQwenGenerator(
            model_name="qwen3-8b",
            model_path=QWEN3_8B,
            device="cuda:3",
            max_input_tokens=4096,
            max_new_tokens=max_new_tokens,
        ),
    }
    try:
        for sample in samples:
            events, result = run_qmsum_sample(sample, generators, judge, judge_mode)
            append_jsonl(trace_path, [event_dict(event) for event in events])
            append_jsonl(result_path, [result])
    finally:
        del generators
        cleanup_cuda()


def run_qmsum_sample(
    sample: TaskSample,
    generators: dict[str, LocalQwenGenerator],
    judge: OpenRouterSummaryJudge | None,
    judge_mode: str,
) -> tuple[list[TraceEvent], dict[str, Any]]:
    query = str(sample.metadata["query"])
    chunks = split_qmsum_chunks(sample.input_text, 3)
    chunk_results: dict[str, GenerationResult] = {}
    chunk_events: dict[str, TraceEvent] = {}
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = {
            node_name: executor.submit(
                generators[node_name].generate,
                qmsum_chunk_prompt(query, chunk_text),
            )
            for node_name, chunk_text in zip(
                ["chunk_0", "chunk_1", "chunk_2"],
                chunks,
                strict=True,
            )
        }
        for node_name, future in futures.items():
            result = future.result()
            chunk_results[node_name] = result
    merge_result = generators["merge"].generate(
        qmsum_merge_prompt(query, chunk_results)
    )
    quality_result, judge_event = evaluate_qmsum(
        sample,
        merge_result.text,
        judge,
        judge_mode,
        merge_result.ended_at,
    )
    for node_name, result in chunk_results.items():
        chunk_events[node_name] = model_event(
            workflow_name="qmsum_3way",
            sample_id=sample.sample_id,
            node_name=node_name,
            result=result,
            generator=generators[node_name],
            release_at=merge_result.ended_at,
            pipeline_idle_time=max(0.0, merge_result.ended_at - result.ended_at),
        )
    merge_event = model_event(
        workflow_name="qmsum_3way",
        sample_id=sample.sample_id,
        node_name="merge",
        result=merge_result,
        generator=generators["merge"],
        release_at=merge_result.ended_at,
        pipeline_idle_time=0.0,
    )
    events = [chunk_events[name] for name in sorted(chunk_events)] + [merge_event]
    if judge_event is not None:
        events.append(judge_event)
    return events, {
        "workflow_name": "qmsum_3way",
        "sample_id": sample.sample_id,
        "final_output": merge_result.text,
        "quality_result": quality_result,
        "node_outputs": {name: result.text for name, result in chunk_results.items()},
    }


def run_mbpp_chain(
    samples: list[TaskSample],
    *,
    output_dir: Path,
    max_input_tokens: int,
    max_new_tokens: int,
) -> None:
    trace_path = output_dir / "mbpp_chain_trace.jsonl"
    result_path = output_dir / "mbpp_chain_results.jsonl"
    trace_path.unlink(missing_ok=True)
    result_path.unlink(missing_ok=True)
    generators = {
        "coder": LocalQwenGenerator(
            model_name="qwen3-14b",
            model_path=QWEN3_14B,
            device="cuda:0",
            max_input_tokens=max_input_tokens,
            max_new_tokens=max_new_tokens,
        ),
        "reviewer": LocalQwenGenerator(
            model_name="qwen3-8b",
            model_path=QWEN3_8B,
            device="cuda:1",
            max_input_tokens=max_input_tokens,
            max_new_tokens=max_new_tokens,
        ),
        "repair": LocalQwenGenerator(
            model_name="qwen3-14b",
            model_path=QWEN3_14B,
            device="cuda:2",
            max_input_tokens=max_input_tokens,
            max_new_tokens=max_new_tokens,
        ),
    }
    try:
        for sample in samples:
            events, result = run_mbpp_sample(sample, generators)
            append_jsonl(trace_path, [event_dict(event) for event in events])
            append_jsonl(result_path, [result])
    finally:
        del generators
        cleanup_cuda()


def run_mbpp_sample(
    sample: TaskSample,
    generators: dict[str, LocalQwenGenerator],
) -> tuple[list[TraceEvent], dict[str, Any]]:
    coder_result = generators["coder"].generate(mbpp_coder_prompt(sample))
    tester_started_at = time.perf_counter()
    first_eval = evaluate_mbpp(
        coder_result.text,
        sample.test_imports,
        sample.test_list,
    )
    tester_ended_at = time.perf_counter()
    reviewer_result = generators["reviewer"].generate(
        mbpp_reviewer_prompt(sample, coder_result.text, first_eval)
    )
    repair_result = generators["repair"].generate(
        mbpp_repair_prompt(sample, coder_result.text, first_eval, reviewer_result.text)
    )
    final_started_at = time.perf_counter()
    final_eval = evaluate_mbpp(
        repair_result.text,
        sample.test_imports,
        sample.test_list,
    )
    final_ended_at = time.perf_counter()
    session_start = coder_result.started_at
    session_end = final_ended_at
    events = [
        model_event(
            workflow_name="mbpp_chain",
            sample_id=sample.sample_id,
            node_name="coder",
            result=coder_result,
            generator=generators["coder"],
            release_at=session_end,
            pipeline_idle_time=max(0.0, session_end - coder_result.ended_at),
            resident_started_at=session_start,
            resident_ended_at=session_end,
            model_instance_id="coder:qwen3-14b",
        ),
        deterministic_event(
            workflow_name="mbpp_chain",
            sample_id=sample.sample_id,
            node_name="tester",
            started_at=tester_started_at,
            ended_at=tester_ended_at,
            output_text=first_eval.model_dump_json(),
            quality_result=first_eval.model_dump(),
        ),
        model_event(
            workflow_name="mbpp_chain",
            sample_id=sample.sample_id,
            node_name="reviewer",
            result=reviewer_result,
            generator=generators["reviewer"],
            release_at=session_end,
            pipeline_idle_time=max(0.0, session_end - reviewer_result.ended_at),
            resident_started_at=session_start,
            resident_ended_at=session_end,
            model_instance_id="reviewer:qwen3-8b",
        ),
        model_event(
            workflow_name="mbpp_chain",
            sample_id=sample.sample_id,
            node_name="repair",
            result=repair_result,
            generator=generators["repair"],
            release_at=session_end,
            pipeline_idle_time=max(0.0, session_end - repair_result.ended_at),
            resident_started_at=session_start,
            resident_ended_at=session_end,
            model_instance_id="repair:qwen3-14b",
        ),
        deterministic_event(
            workflow_name="mbpp_chain",
            sample_id=sample.sample_id,
            node_name="final_tester",
            started_at=final_started_at,
            ended_at=final_ended_at,
            output_text=final_eval.model_dump_json(),
            quality_result=final_eval.model_dump(),
        ),
    ]
    return events, {
        "workflow_name": "mbpp_chain",
        "sample_id": sample.sample_id,
        "coder_output": coder_result.text,
        "reviewer_output": reviewer_result.text,
        "repair_output": repair_result.text,
        "first_eval": first_eval.model_dump(),
        "final_eval": final_eval.model_dump(),
    }


def build_summary_judge(judge_mode: str) -> OpenRouterSummaryJudge | None:
    if judge_mode == "rouge":
        return None
    assert judge_mode == "llm", judge_mode
    load_dotenv()
    return OpenRouterSummaryJudge.from_env()


def evaluate_qmsum(
    sample: TaskSample,
    output: str,
    judge: OpenRouterSummaryJudge | None,
    judge_mode: str,
    ready_at: float,
) -> tuple[dict[str, Any], TraceEvent | None]:
    started_at = time.perf_counter()
    if judge_mode == "llm":
        assert judge is not None
        evaluation = evaluate_summary_with_judge(sample, output, judge)
    else:
        assert sample.gold_answer is not None
        evaluation = evaluate_summary_rouge(output, sample.gold_answer)
    ended_at = time.perf_counter()
    quality_result = evaluation.model_dump()
    return quality_result, judge_trace_event(
        sample,
        started_at,
        ended_at,
        ready_at,
        evaluation,
    )


def split_qmsum_chunks(input_text: str, chunk_count: int) -> list[str]:
    turns = input_text.split("\n\n")
    groups = split_evenly(turns, chunk_count)
    return ["\n\n".join(group) for group in groups]


def qmsum_chunk_prompt(query: str, chunk_text: str) -> str:
    return (
        "Summarize transcript details relevant to the query.\n"
        "Preserve decisions, reasons, participants, and concrete outcomes.\n"
        "Return only the partial answer summary.\n\n"
        f"Query:\n{query}\n\nTranscript chunk:\n{chunk_text}"
    )


def qmsum_merge_prompt(
    query: str,
    chunk_results: dict[str, GenerationResult],
) -> str:
    partials = {
        node_name: result.text for node_name, result in sorted(chunk_results.items())
    }
    return (
        "Merge the partial answers into one final answer to the query.\n"
        "Remove duplication, preserve query-relevant facts, and keep the answer "
        "grounded.\n"
        "Return only the final answer.\n\n"
        f"Query:\n{query}\n\nPartial answers:\n"
        f"{json.dumps(partials, ensure_ascii=False, sort_keys=True)}"
    )


def mbpp_coder_prompt(sample: TaskSample) -> str:
    tests = "\n".join(sample.test_imports + sample.test_list)
    return (
        "Write a correct Python solution for the programming task.\n"
        "Return only Python code in one fenced code block.\n\n"
        f"Task:\n{sample.input_text}\n\nTests that must pass:\n{tests}"
    )


def mbpp_reviewer_prompt(sample: TaskSample, code: str, evaluation: Any) -> str:
    tests = "\n".join(sample.test_imports + sample.test_list)
    return (
        "Review the Python solution for correctness.\n"
        "Focus on logic errors, edge cases, and API misuse.\n"
        "Return only the review analysis, no code.\n\n"
        f"Task:\n{sample.input_text}\n\nCurrent solution:\n{code}\n\n"
        f"Test result:\n{evaluation.model_dump_json()}\n\nTests:\n{tests}"
    )


def mbpp_repair_prompt(
    sample: TaskSample,
    code: str,
    evaluation: Any,
    review: str,
) -> str:
    tests = "\n".join(sample.test_imports + sample.test_list)
    return (
        "Repair the Python solution so that it passes the tests.\n"
        "Return only Python code in one fenced code block.\n\n"
        f"Task:\n{sample.input_text}\n\nCurrent solution:\n{code}\n\n"
        f"Code review:\n{review}\n\n"
        f"Test result:\n{evaluation.model_dump_json()}\n\nTests:\n{tests}"
    )


def model_event(
    *,
    workflow_name: str,
    sample_id: str,
    node_name: str,
    result: GenerationResult,
    generator: LocalQwenGenerator,
    release_at: float,
    pipeline_idle_time: float,
    resident_started_at: float | None = None,
    resident_ended_at: float | None = None,
    model_instance_id: str | None = None,
) -> TraceEvent:
    return TraceEvent(
        workflow_name=workflow_name,
        sample_id=sample_id,
        node_name=node_name,
        node_type="llm",
        started_at=result.started_at,
        ended_at=result.ended_at,
        duration_sec=result.duration_sec,
        ready_at=result.started_at,
        release_at=release_at,
        pipeline_idle_time=pipeline_idle_time,
        model_name=generator.model_name,
        model_instance_id=model_instance_id or f"{node_name}:{generator.model_name}",
        resident_started_at=resident_started_at or result.started_at,
        resident_ended_at=resident_ended_at or release_at,
        input_token_count=result.input_token_count,
        output_token_count=result.output_token_count,
        status="ok" if result.text else "empty_output",
        output_text=result.text,
    )


def deterministic_event(
    *,
    workflow_name: str,
    sample_id: str,
    node_name: str,
    started_at: float,
    ended_at: float,
    output_text: str,
    quality_result: dict[str, Any] | None = None,
) -> TraceEvent:
    return TraceEvent(
        workflow_name=workflow_name,
        sample_id=sample_id,
        node_name=node_name,
        node_type="deterministic",
        started_at=started_at,
        ended_at=ended_at,
        duration_sec=ended_at - started_at,
        ready_at=started_at,
        release_at=ended_at,
        pipeline_idle_time=0.0,
        model_name=None,
        model_instance_id=None,
        resident_started_at=None,
        resident_ended_at=None,
        input_token_count=0,
        output_token_count=0,
        status="ok",
        quality_result=quality_result,
        output_text=output_text,
    )


def judge_trace_event(
    sample: TaskSample,
    started_at: float,
    ended_at: float,
    ready_at: float,
    evaluation: SummaryEvaluation,
) -> TraceEvent:
    return TraceEvent(
        workflow_name="qmsum_3way",
        sample_id=sample.sample_id,
        node_name="judge",
        node_type="evaluator",
        started_at=started_at,
        ended_at=ended_at,
        duration_sec=ended_at - started_at,
        ready_at=ready_at,
        release_at=ended_at,
        pipeline_idle_time=0.0,
        model_name="external-judge",
        model_instance_id=None,
        resident_started_at=None,
        resident_ended_at=None,
        input_token_count=0,
        output_token_count=0,
        status="ok" if evaluation.error_type is None else "eval_fail",
        quality_result=evaluation.model_dump(),
    )


def cleanup_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
