from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import pytest

from experiment.workflow.analysis import (
    confidence_interval_95,
    integrate_energy_joules,
    integrate_idle_subtracted_energy_joules,
    integrate_idle_subtracted_memory_gib_seconds,
    integrate_memory_gib_seconds,
    interval_union_duration,
    steady_state_completion_slope,
    summarize_gpu_telemetry,
    summarize_queue_telemetry,
    summarize_trace,
)


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(f"{json.dumps(row)}\n" for row in rows),
        encoding="utf-8",
    )


def event(
    event_type: str,
    ts: float,
    *,
    replica_id: str | None = None,
    node_id: str | None = None,
    session_id: str | None = None,
    item_id: str | None = None,
    source_node: str | None = None,
    target_node: str | None = None,
    acquire_id: str | None = None,
    payload: dict[str, object] | None = None,
) -> dict[str, object]:
    return {
        "event_type": event_type,
        "ts": ts,
        "replica_id": replica_id,
        "node_id": node_id,
        "session_id": session_id,
        "item_id": item_id,
        "source_node": source_node,
        "target_node": target_node,
        "acquire_id": acquire_id,
        "payload": payload or {},
    }


def test_interval_union_avoids_continuous_batching_double_count() -> None:
    assert interval_union_duration([(0.0, 4.0), (1.0, 3.0), (3.0, 6.0)]) == 6.0
    assert interval_union_duration([(0.0, 1.0), (2.0, 4.0)]) == 3.0


def test_trace_summary_uses_per_replica_interval_unions(tmp_path: Path) -> None:
    trace_path = tmp_path / "workflow_trace.jsonl"
    write_jsonl(
        trace_path,
        [
            event("run_started", 0.0),
            event("model_load_started", 0.0, replica_id="replica-1"),
            event(
                "task_execution_finished",
                4.0,
                replica_id="replica-1",
                node_id="chunk",
                payload={
                    "duration_sec": 4.0,
                    "started_at": 0.0,
                    "finished_at": 4.0,
                },
            ),
            event(
                "task_execution_finished",
                5.0,
                replica_id="replica-1",
                node_id="chunk",
                payload={
                    "duration_sec": 4.0,
                    "started_at": 1.0,
                    "finished_at": 5.0,
                },
            ),
            event("session_completed", 5.0, payload={"latency_sec": 5.0}),
            event("model_evicted", 6.0, replica_id="replica-1"),
            event("run_finished", 7.0),
        ],
    )

    summary = summarize_trace(trace_path)

    assert summary["active_gpu_seconds"] == 5.0
    assert summary["resident_gpu_seconds"] == 6.0
    assert summary["makespan_sec"] == 5.0
    assert summary["run_duration_sec"] == 7.0
    assert summary["cleanup_duration_sec"] == 2.0
    assert summary["pipeline_bubble_gpu_seconds"] == 0.0
    assert summary["pipeline_bubble_ratio"] == 0.0


def test_capacity_slope_and_energy_integration_are_deterministic(
    tmp_path: Path,
) -> None:
    timestamps = [float(index) for index in range(10)]
    telemetry_path = tmp_path / "gpu_telemetry.jsonl"
    write_jsonl(
        telemetry_path,
        [
            {"gpu_index": 0, "ts": 0.0, "power_watts": 100.0},
            {"gpu_index": 0, "ts": 2.0, "power_watts": 200.0},
        ],
    )

    assert steady_state_completion_slope(
        timestamps,
        start_fraction=0.2,
        end_fraction=0.8,
    ) == pytest.approx(1.0)
    assert integrate_energy_joules(telemetry_path) == 300.0


def test_trace_summary_derives_scheduler_and_lifecycle_metrics(
    tmp_path: Path,
) -> None:
    trace_path = tmp_path / "workflow_trace.jsonl"
    write_jsonl(
        trace_path,
        [
            event("run_started", 0.0),
            event("model_load_started", 0.0, replica_id="r1"),
            event("prefetch_started", 0.0, replica_id="r1"),
            event(
                "model_load_finished",
                1.0,
                replica_id="r1",
                payload={"duration_sec": 1.0},
            ),
            event(
                "prefetch_finished",
                1.0,
                replica_id="r1",
                payload={"duration_sec": 1.0},
            ),
            event("model_load_started", 1.0, replica_id="r2"),
            event("prefetch_started", 1.0, replica_id="r2"),
            event(
                "model_load_finished",
                2.0,
                replica_id="r2",
                payload={"duration_sec": 1.0},
            ),
            event(
                "prefetch_finished",
                2.0,
                replica_id="r2",
                payload={"duration_sec": 1.0},
            ),
            event("acquire_requested", 1.0, acquire_id="a1"),
            event(
                "acquire_granted",
                2.0,
                replica_id="r1",
                acquire_id="a1",
                payload={"admitted_batch_size": 2},
            ),
            event(
                "task_execution_finished",
                5.0,
                replica_id="r1",
                node_id="stage",
                payload={
                    "started_at": 2.0,
                    "finished_at": 5.0,
                    "duration_sec": 3.0,
                    "input_tokens": 10,
                    "output_tokens": 4,
                    "hit_token_limit": False,
                    "queue_time_sec": 0.5,
                    "time_to_first_token_sec": 1.0,
                    "replica_inflight_at_start": 2,
                },
            ),
            event(
                "task_execution_finished",
                6.0,
                replica_id="r1",
                node_id="stage",
                payload={
                    "started_at": 3.0,
                    "finished_at": 6.0,
                    "duration_sec": 3.0,
                    "input_tokens": 12,
                    "output_tokens": 5,
                    "hit_token_limit": True,
                    "queue_time_sec": 0.7,
                    "time_to_first_token_sec": 1.2,
                    "replica_inflight_at_start": 3,
                },
            ),
            event(
                "item_emitted",
                5.0,
                item_id="i1",
                source_node="stage",
                target_node="merge",
            ),
            event(
                "item_enqueued",
                6.0,
                item_id="i1",
                source_node="stage",
                target_node="merge",
            ),
            event("fanin_wait", 4.0, session_id="s1", node_id="merge"),
            event("fanin_ready", 7.0, session_id="s1", node_id="merge"),
            event("prefetch_skipped", 3.0),
            event("model_reused", 3.0, replica_id="r1"),
            event("model_evicted", 4.0, replica_id="r2"),
            event("session_completed", 6.0, payload={"latency_sec": 6.0}),
            event("session_completed", 7.0, payload={"latency_sec": 7.0}),
            event("model_eviction_started", 8.0, replica_id="r1"),
            event("model_evicted", 9.0, replica_id="r1"),
            event("run_finished", 10.0),
        ],
    )

    summary = summarize_trace(trace_path)

    assert summary["active_gpu_seconds"] == 4.0
    assert summary["resident_gpu_seconds"] == 9.0
    assert summary["idle_resident_gpu_seconds"] == 5.0
    assert summary["workload_resident_gpu_seconds"] == 8.0
    assert summary["pipeline_bubble_gpu_seconds"] == 4.0
    assert summary["pipeline_bubble_ratio"] == 0.5
    assert summary["loading_gpu_seconds"] == 2.0
    assert summary["evicting_gpu_seconds"] == 1.0
    assert summary["stage_utilization"] == {"stage": pytest.approx(4.0 / 7.0)}
    assert cast(dict[str, Any], summary["acquire_wait_sec"])["mean"] == 1.0
    assert cast(dict[str, Any], summary["fanin_wait_sec"])["mean"] == 3.0
    assert cast(dict[str, Any], summary["item_enqueue_blocked_sec"])["mean"] == 1.0
    assert summary["backpressure_duration_sec"] == 1.0
    assert summary["model_load_count"] == 2
    assert summary["model_reuse_count"] == 1
    assert summary["model_eviction_count"] == 2
    assert summary["prefetch_count"] == 2
    assert summary["prefetch_skip_count"] == 1
    assert summary["wasted_prefetch_count"] == 1
    assert summary["peak_replica_inflight"] == 3
    assert summary["peak_resident_replica_count"] == 2
    assert summary["input_token_count"] == 22
    assert summary["output_token_count"] == 9
    assert summary["token_limit_hit_count"] == 1


def test_telemetry_reports_raw_and_idle_subtracted_integrals(
    tmp_path: Path,
) -> None:
    gpu_path = tmp_path / "gpu_telemetry.jsonl"
    queue_path = tmp_path / "queue_telemetry.jsonl"
    write_jsonl(
        gpu_path,
        [
            {
                "gpu_index": 0,
                "ts": 0.0,
                "power_watts": 50.0,
                "memory_used_mb": 1024.0,
                "gpu_utilization_percent": 0.0,
                "memory_utilization_percent": 0.0,
            },
            {
                "gpu_index": 0,
                "ts": 2.0,
                "power_watts": 150.0,
                "memory_used_mb": 3072.0,
                "gpu_utilization_percent": 80.0,
                "memory_utilization_percent": 40.0,
            },
        ],
    )
    write_jsonl(
        queue_path,
        [
            {"node_id": "stage", "ts": 0.0, "queue_size": 0},
            {"node_id": "stage", "ts": 2.0, "queue_size": 4},
        ],
    )

    gpu = summarize_gpu_telemetry(gpu_path)
    queue = summarize_queue_telemetry(queue_path)

    assert integrate_energy_joules(gpu_path) == 200.0
    assert integrate_idle_subtracted_energy_joules(gpu_path) == 100.0
    assert integrate_memory_gib_seconds(gpu_path) == 4.0
    assert integrate_idle_subtracted_memory_gib_seconds(gpu_path) == 2.0
    idle_baseline = cast(dict[str, Any], gpu["telemetry_idle_baseline"])
    queue_by_node = cast(dict[str, Any], queue["by_node"])
    assert idle_baseline["method"] == ("per_gpu_first_sample_clamped_subtraction")
    assert queue["peak"] == 4.0
    assert queue_by_node["stage"]["time_weighted_mean"] == 0.0
    assert queue_by_node["stage"]["integration_method"] == (
        "left_sample_hold_between_samples"
    )


def test_static_baseline_marks_unobservable_metrics_unavailable(
    tmp_path: Path,
) -> None:
    trace_path = tmp_path / "workflow_trace.jsonl"
    write_jsonl(
        trace_path,
        [
            event("run_started", 0.0),
            event(
                "model_load_started",
                0.0,
                replica_id="baseline-model",
                payload={"reason": "static_baseline"},
            ),
            event(
                "model_load_finished",
                1.0,
                replica_id="baseline-model",
                payload={"duration_sec": 1.0},
            ),
            event("acquire_requested", 1.0, acquire_id="request"),
            event(
                "task_execution_finished",
                3.0,
                replica_id="baseline-model",
                payload={
                    "started_at": 1.0,
                    "finished_at": 3.0,
                    "duration_sec": 2.0,
                    "input_tokens": 4,
                    "output_tokens": 2,
                },
            ),
            event("session_completed", 3.0, payload={"latency_sec": 3.0}),
            event("model_evicted", 4.0, replica_id="baseline-model"),
            event("run_finished", 5.0),
        ],
    )

    summary = summarize_trace(trace_path)

    assert summary["makespan_sec"] == 3.0
    assert summary["run_duration_sec"] == 5.0
    assert summary["acquire_wait_sec"] is None
    assert summary["fanin_wait_sec"] is None
    assert summary["item_enqueue_blocked_sec"] is None
    assert summary["model_reuse_count"] is None
    availability = cast(dict[str, bool], summary["metric_availability"])
    assert not availability["acquire_wait_sec"]


def test_confidence_interval_uses_trials_as_repetitions() -> None:
    result = confidence_interval_95([1.0, 2.0, 3.0, 4.0, 5.0])

    assert result["mean"] == 3.0
    assert result["std"] == pytest.approx(1.58113883)
    assert result["ci95_low"] < 3.0 < result["ci95_high"]
