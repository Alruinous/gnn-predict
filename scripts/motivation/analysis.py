from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from scripts.motivation.common import read_jsonl, stats, write_jsonl


def summarize_qmsum(trace_path: Path, output_path: Path) -> dict[str, Any]:
    events = read_jsonl(trace_path)
    by_sample = group_by_sample(events)
    sample_metrics = [
        qmsum_sample_metrics(sample_id, sample_events)
        for sample_id, sample_events in by_sample.items()
    ]
    busy_total = sum(metric["chunk_busy_time"] for metric in sample_metrics)
    idle_total = sum(metric["pipeline_idle_time"] for metric in sample_metrics)
    baseline_total = sum(metric["generation_duration_sec"] for metric in sample_metrics)
    oracle_total = qmsum_pipeline_oracle_total(sample_metrics)
    summary = {
        "workflow_name": "qmsum_3way",
        "sample_count": len(sample_metrics),
        "generation_duration_sec": stats(
            [metric["generation_duration_sec"] for metric in sample_metrics]
        ),
        "chunk_busy_time": stats(
            [metric["chunk_busy_time"] for metric in sample_metrics]
        ),
        "merge_duration_sec": stats(
            [metric["merge_duration_sec"] for metric in sample_metrics]
        ),
        "pipeline_idle_time": stats(
            [metric["pipeline_idle_time"] for metric in sample_metrics]
        ),
        "pipeline_bubble_ratio": stats(
            [metric["pipeline_bubble_ratio"] for metric in sample_metrics]
        ),
        "stage_utilization": busy_total / (busy_total + idle_total),
        "baseline_total_sec": baseline_total,
        "oracle_total_sec": oracle_total,
        "speedup_upper_bound": baseline_total / oracle_total,
        "quality": summarize_quality(events),
        "samples": sample_metrics,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), "utf-8")
    return summary


def qmsum_sample_metrics(
    sample_id: str,
    events: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    chunks = sorted(
        [event for event in events if event["node_name"].startswith("chunk_")],
        key=lambda event: event["node_name"],
    )
    merge = single_event(events, "merge")
    assert chunks, sample_id
    first_start = min(float(event["started_at"]) for event in chunks)
    generation_end = float(merge["ended_at"])
    chunk_busy = sum(float(event["duration_sec"]) for event in chunks)
    idle = sum(max(0.0, generation_end - float(event["ended_at"])) for event in chunks)
    return {
        "sample_id": sample_id,
        "chunk_durations": [float(event["duration_sec"]) for event in chunks],
        "merge_duration_sec": float(merge["duration_sec"]),
        "generation_duration_sec": generation_end - first_start,
        "chunk_busy_time": chunk_busy,
        "pipeline_idle_time": idle,
        "pipeline_bubble_ratio": idle / (chunk_busy + idle),
        "chunk_finish_spread_sec": max(float(event["ended_at"]) for event in chunks)
        - min(float(event["ended_at"]) for event in chunks),
    }


def qmsum_pipeline_oracle_total(sample_metrics: Sequence[dict[str, Any]]) -> float:
    chunk_count = len(sample_metrics[0]["chunk_durations"])
    chunk_available = [0.0] * chunk_count
    merge_available = 0.0
    for metric in sample_metrics:
        done_times = []
        for index, duration in enumerate(metric["chunk_durations"]):
            start = chunk_available[index]
            end = start + float(duration)
            chunk_available[index] = end
            done_times.append(end)
        merge_start = max(max(done_times), merge_available)
        merge_available = merge_start + float(metric["merge_duration_sec"])
    return merge_available


def summarize_mbpp(trace_path: Path, output_path: Path) -> dict[str, Any]:
    events = read_jsonl(trace_path)
    by_sample = group_by_sample(events)
    sample_metrics = [
        mbpp_sample_metrics(sample_id, sample_events)
        for sample_id, sample_events in by_sample.items()
    ]
    resident_seconds = sum(
        metric["resident_model_seconds"] for metric in sample_metrics
    )
    active_seconds = sum(metric["active_model_seconds"] for metric in sample_metrics)
    summary = {
        "workflow_name": "mbpp_chain",
        "sample_count": len(sample_metrics),
        "end_to_end_duration_sec": stats(
            [metric["end_to_end_duration_sec"] for metric in sample_metrics]
        ),
        "resident_model_seconds": stats(
            [metric["resident_model_seconds"] for metric in sample_metrics]
        ),
        "active_model_seconds": stats(
            [metric["active_model_seconds"] for metric in sample_metrics]
        ),
        "node_idle_time": stats(
            [metric["node_idle_time"] for metric in sample_metrics]
        ),
        "resource_gap": resident_seconds / active_seconds,
        "frontier_gap": 3.0,
        "resident_model_count_peak": 3,
        "active_model_count_peak": 1,
        "pass_at_1": pass_rate(events),
        "samples": sample_metrics,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), "utf-8")
    return summary


def mbpp_sample_metrics(
    sample_id: str,
    events: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    coder = single_event(events, "coder")
    reviewer = single_event(events, "reviewer")
    repair = single_event(events, "repair")
    final_tester = single_event(events, "final_tester")
    start = float(coder["started_at"])
    end = float(final_tester["ended_at"])
    active = (
        float(coder["duration_sec"])
        + float(reviewer["duration_sec"])
        + float(repair["duration_sec"])
    )
    resident = 3.0 * (end - start)
    return {
        "sample_id": sample_id,
        "end_to_end_duration_sec": end - start,
        "active_model_seconds": active,
        "resident_model_seconds": resident,
        "node_idle_time": resident - active,
        "resource_gap": resident / active,
    }


def summarize_quality(events: Sequence[dict[str, Any]]) -> dict[str, float]:
    results = [
        event["quality_result"]
        for event in events
        if event["node_name"] == "judge" and event.get("quality_result")
    ]
    keys = ["rouge1", "rouge2", "rouge_l", "llm_score"]
    quality = {}
    for key in keys:
        values = [
            float(result[key]) for result in results if result.get(key) is not None
        ]
        quality[key] = sum(values) / max(1, len(values))
    quality["pass_rate"] = sum(1 for result in results if result.get("passed")) / max(
        1, len(results)
    )
    return quality


def pass_rate(events: Sequence[dict[str, Any]]) -> float:
    results = [
        event["quality_result"]
        for event in events
        if event["node_name"] == "final_tester" and event.get("quality_result")
    ]
    return sum(1 for result in results if result["passed"]) / max(1, len(results))


def group_by_sample(
    events: Sequence[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    by_sample: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for event in events:
        if event["node_type"] == "input":
            continue
        by_sample[event["sample_id"]].append(event)
    return dict(by_sample)


def single_event(
    events: Sequence[dict[str, Any]],
    node_name: str,
) -> dict[str, Any]:
    matched = [event for event in events if event["node_name"] == node_name]
    assert len(matched) == 1, (node_name, len(matched))
    return matched[0]


def write_combined_summary(
    output_path: Path,
    qmsum_summary: dict[str, Any],
    mbpp_summary: dict[str, Any],
) -> None:
    write_jsonl(
        output_path,
        [
            {"record_type": "summary", **qmsum_summary},
            {"record_type": "summary", **mbpp_summary},
        ],
    )
