from __future__ import annotations

import json
import math
import statistics
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from itertools import pairwise
from pathlib import Path
from typing import Any

from experiment.workflow.artifacts import write_json_exclusive

T_CRITICAL_95_DF4 = 2.7764451051977987
TELEMETRY_BASELINE_METHOD = "per_gpu_first_sample_clamped_subtraction"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as file:
        for line in file:
            value = json.loads(line)
            if not isinstance(value, dict):
                raise TypeError(f"JSONL row must be an object: {path}")
            rows.append(value)
    return rows


def interval_union_duration(intervals: Iterable[tuple[float, float]]) -> float:
    return sum(end - start for start, end in _merge_intervals(intervals))


def summarize_trace(
    trace_path: Path,
    *,
    steady_state_start_fraction: float = 0.2,
    steady_state_end_fraction: float = 0.8,
) -> dict[str, object]:
    events = read_jsonl(trace_path)
    if not events:
        raise ValueError("workflow trace is empty")
    run_started = _single_timestamp(events, "run_started")
    run_finished = _single_timestamp(events, "run_finished")
    if run_finished < run_started:
        raise ValueError("run_finished precedes run_started")

    completion_times = sorted(
        _number(event["ts"])
        for event in events
        if event.get("event_type") == "session_completed"
    )
    session_latencies = [
        _number(_payload(event)["latency_sec"])
        for event in events
        if event.get("event_type") in ("session_completed", "session_failed")
    ]
    terminal_times = [
        _number(event["ts"])
        for event in events
        if event.get("event_type") in ("session_completed", "session_failed")
    ]
    business_finished = max(terminal_times, default=run_finished)
    if business_finished < run_started:
        raise ValueError("session terminal event precedes run_started")
    if business_finished > run_finished:
        raise ValueError("session terminal event follows run_finished")
    execution_intervals, stage_durations = _execution_metrics(events)
    resident_intervals = _resident_intervals(events, run_finished)
    active_by_replica = {
        replica_id: interval_union_duration(intervals)
        for replica_id, intervals in sorted(execution_intervals.items())
    }
    resident_by_replica = {
        replica_id: interval_union_duration(intervals)
        for replica_id, intervals in sorted(resident_intervals.items())
    }
    idle_by_replica = {
        replica_id: resident_by_replica[replica_id]
        - _interval_intersection_duration(
            resident_intervals[replica_id], execution_intervals.get(replica_id, ())
        )
        for replica_id in resident_by_replica
    }
    workload_resident_intervals = {
        replica_id: _clip_intervals(intervals, run_started, business_finished)
        for replica_id, intervals in resident_intervals.items()
    }
    workload_resident_by_replica = {
        replica_id: interval_union_duration(intervals)
        for replica_id, intervals in sorted(workload_resident_intervals.items())
    }
    pipeline_bubble_by_replica = {
        replica_id: max(
            0.0,
            workload_resident_by_replica[replica_id]
            - _interval_intersection_duration(
                intervals,
                execution_intervals.get(replica_id, ()),
            ),
        )
        for replica_id, intervals in workload_resident_intervals.items()
    }
    makespan = business_finished - run_started
    run_duration = run_finished - run_started
    cleanup_duration = run_finished - business_finished
    workload_resident_gpu_seconds = sum(workload_resident_by_replica.values())
    pipeline_bubble_gpu_seconds = sum(pipeline_bubble_by_replica.values())
    completed = len(completion_times)
    steady_state_rate = _steady_state_rate(
        completion_times,
        start_fraction=steady_state_start_fraction,
        end_fraction=steady_state_end_fraction,
    )
    acquire_waits, acquire_unmatched_starts, acquire_unmatched_ends = _paired_durations(
        events,
        "acquire_requested",
        "acquire_granted",
        ("acquire_id",),
        cancel_events=("acquire_cancelled",),
    )
    enqueue_blocked_intervals, item_unmatched_emits, item_unmatched_enqueues = (
        _item_enqueue_blocked_intervals(events)
    )
    fanin_waits, fanin_unmatched_waits = _fanin_wait_durations(events)
    queue_times, ttfts, input_tokens, output_tokens, token_limit_hits = (
        _inference_payload_metrics(events)
    )
    lifecycle = _lifecycle_metrics(events, run_finished, execution_intervals)
    stage_intervals = _stage_intervals(events)
    static_baseline = _is_static_baseline(events)
    event_types = {event.get("event_type") for event in events}
    availability = {
        "acquire_wait_sec": not static_baseline or "acquire_granted" in event_types,
        "fanin_wait_sec": not static_baseline or "fanin_ready" in event_types,
        "item_enqueue_blocked_sec": (
            not static_baseline or "item_emitted" in event_types
        ),
        "backpressure_duration_sec": (
            not static_baseline or "item_emitted" in event_types
        ),
        "model_reuse_count": not static_baseline or "model_reused" in event_types,
        "evicting_gpu_seconds": (
            not static_baseline or "model_eviction_started" in event_types
        ),
    }

    return {
        "makespan_sec": makespan,
        "run_duration_sec": run_duration,
        "cleanup_duration_sec": cleanup_duration,
        "completed_session_count": completed,
        "sessions_per_min": completed * 60.0 / makespan if makespan else 0.0,
        "steady_state_sessions_per_sec": steady_state_rate,
        "steady_state_sessions_per_min": steady_state_rate * 60.0,
        "first_completion_latency_sec": (
            min(completion_times) - run_started if completion_times else 0.0
        ),
        "session_latency_sec": _percentiles(session_latencies),
        "active_gpu_seconds": sum(active_by_replica.values()),
        "active_gpu_seconds_by_replica": active_by_replica,
        "resident_gpu_seconds": sum(resident_by_replica.values()),
        "resident_gpu_seconds_by_replica": resident_by_replica,
        "idle_resident_gpu_seconds": sum(idle_by_replica.values()),
        "idle_resident_gpu_seconds_by_replica": idle_by_replica,
        "workload_resident_gpu_seconds": workload_resident_gpu_seconds,
        "pipeline_bubble_gpu_seconds": pipeline_bubble_gpu_seconds,
        "pipeline_bubble_gpu_seconds_by_replica": pipeline_bubble_by_replica,
        "pipeline_bubble_ratio": (
            pipeline_bubble_gpu_seconds / workload_resident_gpu_seconds
            if workload_resident_gpu_seconds
            else 0.0
        ),
        "loading_gpu_seconds": lifecycle["loading_gpu_seconds"],
        "evicting_gpu_seconds": (
            lifecycle["evicting_gpu_seconds"]
            if availability["evicting_gpu_seconds"]
            else None
        ),
        "stage_duration_sec": {
            node_id: _distribution(values)
            for node_id, values in sorted(stage_durations.items())
        },
        "stage_utilization": {
            node_id: (
                interval_union_duration(intervals) / makespan if makespan else 0.0
            )
            for node_id, intervals in sorted(stage_intervals.items())
        },
        "acquire_wait_sec": (
            _timing_distribution(acquire_waits)
            if availability["acquire_wait_sec"]
            else None
        ),
        "vllm_queue_time_sec": _timing_distribution(queue_times),
        "ttft_sec": _timing_distribution(ttfts),
        "fanin_wait_sec": (
            _timing_distribution(fanin_waits)
            if availability["fanin_wait_sec"]
            else None
        ),
        "item_enqueue_blocked_sec": (
            _timing_distribution(
                [end - start for start, end in enqueue_blocked_intervals]
            )
            if availability["item_enqueue_blocked_sec"]
            else None
        ),
        "backpressure_duration_sec": (
            interval_union_duration(enqueue_blocked_intervals)
            if availability["backpressure_duration_sec"]
            else None
        ),
        "model_load_count": lifecycle["model_load_count"],
        "model_reuse_count": (
            lifecycle["model_reuse_count"]
            if availability["model_reuse_count"]
            else None
        ),
        "model_eviction_count": lifecycle["model_eviction_count"],
        "prefetch_count": lifecycle["prefetch_count"],
        "prefetch_skip_count": lifecycle["prefetch_skip_count"],
        "wasted_prefetch_count": lifecycle["wasted_prefetch_count"],
        "model_load_duration_sec": lifecycle["model_load_duration_sec"],
        "prefetch_load_duration_sec": lifecycle["prefetch_load_duration_sec"],
        "peak_replica_inflight": lifecycle["peak_replica_inflight"],
        "peak_replica_inflight_by_replica": lifecycle[
            "peak_replica_inflight_by_replica"
        ],
        "peak_resident_replica_count": lifecycle["peak_resident_replica_count"],
        "input_token_count": input_tokens,
        "output_token_count": output_tokens,
        "token_limit_hit_count": token_limit_hits,
        "runtime_kind": "static_baseline" if static_baseline else "workflow",
        "metric_availability": availability,
        "trace_pairing": {
            "acquire_unmatched_start_count": acquire_unmatched_starts,
            "acquire_unmatched_end_count": acquire_unmatched_ends,
            "fanin_unmatched_wait_count": fanin_unmatched_waits,
            "item_unmatched_emitted_count": item_unmatched_emits,
            "item_unmatched_enqueued_count": item_unmatched_enqueues,
        },
        "completion_timestamps": completion_times,
    }


def summarize_trial(
    trial_dir: Path,
    *,
    steady_state_start_fraction: float = 0.2,
    steady_state_end_fraction: float = 0.8,
) -> dict[str, object]:
    summary = summarize_trace(
        trial_dir / "workflow_trace.jsonl",
        steady_state_start_fraction=steady_state_start_fraction,
        steady_state_end_fraction=steady_state_end_fraction,
    )
    telemetry_path = trial_dir / "gpu_telemetry.jsonl"
    if telemetry_path.is_file():
        summary.update(summarize_gpu_telemetry(telemetry_path))
        completed = _integer(summary["completed_session_count"])
        energy = _number(summary["energy_joules"])
        adjusted_energy = _number(summary["idle_subtracted_energy_joules"])
        summary["energy_per_completion_joules"] = (
            energy / completed if completed else None
        )
        summary["idle_subtracted_energy_per_completion_joules"] = (
            adjusted_energy / completed if completed else None
        )
        summary["gpu_seconds_per_completion"] = (
            _number(summary["active_gpu_seconds"]) / completed if completed else None
        )
    queue_path = trial_dir / "queue_telemetry.jsonl"
    if queue_path.is_file():
        summary["queue_occupancy"] = summarize_queue_telemetry(queue_path)
    else:
        summary["queue_occupancy"] = None
    return summary


def steady_state_completion_slope(
    completion_timestamps: Sequence[float],
    *,
    start_fraction: float,
    end_fraction: float,
) -> float:
    if not 0 <= start_fraction < end_fraction <= 1:
        raise ValueError("steady-state fractions must be ordered within [0, 1]")
    ordered = sorted(completion_timestamps)
    start = math.floor(len(ordered) * start_fraction)
    end = math.ceil(len(ordered) * end_fraction)
    window = ordered[start:end]
    if len(window) < 2:
        raise ValueError("steady-state window requires at least two completions")
    origin = window[0]
    x = [timestamp - origin for timestamp in window]
    y = list(range(1, len(window) + 1))
    x_mean = statistics.fmean(x)
    y_mean = statistics.fmean(y)
    denominator = sum((value - x_mean) ** 2 for value in x)
    if denominator == 0:
        raise ValueError("completion timestamps do not define a slope")
    return (
        sum(
            (x_value - x_mean) * (y_value - y_mean)
            for x_value, y_value in zip(x, y, strict=True)
        )
        / denominator
    )


def integrate_energy_joules(telemetry_path: Path) -> float:
    return _integrate_gpu_metric(telemetry_path, "power_watts")


def integrate_memory_gib_seconds(telemetry_path: Path) -> float:
    return _integrate_gpu_metric(telemetry_path, "memory_used_mb") / 1024.0


def integrate_idle_subtracted_energy_joules(telemetry_path: Path) -> float:
    return _integrate_idle_subtracted_gpu_metric(telemetry_path, "power_watts")


def integrate_idle_subtracted_memory_gib_seconds(telemetry_path: Path) -> float:
    return (
        _integrate_idle_subtracted_gpu_metric(telemetry_path, "memory_used_mb") / 1024.0
    )


def summarize_gpu_telemetry(telemetry_path: Path) -> dict[str, object]:
    rows = read_jsonl(telemetry_path)
    memory = [_number(row["memory_used_mb"]) / 1024.0 for row in rows]
    gpu_utilization = [_number(row["gpu_utilization_percent"]) for row in rows]
    memory_utilization = [_number(row["memory_utilization_percent"]) for row in rows]
    power_baselines = _first_gpu_values(rows, "power_watts")
    memory_baselines = _first_gpu_values(rows, "memory_used_mb")
    return {
        "energy_joules": integrate_energy_joules(telemetry_path),
        "idle_subtracted_energy_joules": (
            integrate_idle_subtracted_energy_joules(telemetry_path)
        ),
        "memory_gib_seconds": integrate_memory_gib_seconds(telemetry_path),
        "idle_subtracted_memory_gib_seconds": (
            integrate_idle_subtracted_memory_gib_seconds(telemetry_path)
        ),
        "telemetry_idle_baseline": {
            "method": TELEMETRY_BASELINE_METHOD,
            "power_watts_by_gpu": power_baselines,
            "memory_used_gib_by_gpu": {
                gpu_index: value / 1024.0
                for gpu_index, value in memory_baselines.items()
            },
        },
        "gpu_memory_used_gib": _timing_distribution(memory),
        "gpu_utilization_percent": _timing_distribution(gpu_utilization),
        "gpu_memory_utilization_percent": _timing_distribution(memory_utilization),
    }


def summarize_queue_telemetry(telemetry_path: Path) -> dict[str, object]:
    samples: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for row in read_jsonl(telemetry_path):
        node_id = row.get("node_id")
        if not isinstance(node_id, str):
            raise TypeError("queue telemetry node_id must be a string")
        samples[node_id].append((_number(row["ts"]), _number(row["queue_size"])))
    by_node: dict[str, object] = {}
    for node_id, values in sorted(samples.items()):
        ordered = sorted(values)
        duration = ordered[-1][0] - ordered[0][0]
        integral = sum(
            (right_ts - left_ts) * left_value
            for (left_ts, left_value), (right_ts, _) in pairwise(ordered)
        )
        by_node[node_id] = {
            **_timing_distribution([value for _, value in ordered]),
            "time_weighted_mean": integral / duration if duration else ordered[0][1],
            "integration_method": "left_sample_hold_between_samples",
        }
    return {
        "peak": max(
            (value for values in samples.values() for _, value in values),
            default=0.0,
        ),
        "by_node": by_node,
    }


def confidence_interval_95(values: Sequence[float]) -> dict[str, float]:
    if len(values) != 5:
        raise ValueError("formal confidence intervals require five trials")
    mean = statistics.fmean(values)
    std = statistics.stdev(values)
    half_width = T_CRITICAL_95_DF4 * std / math.sqrt(len(values))
    return {
        "mean": mean,
        "std": std,
        "ci95_low": mean - half_width,
        "ci95_high": mean + half_width,
    }


def write_trial_analysis(trial_dir: Path) -> dict[str, object]:
    summary = summarize_trial(trial_dir)
    write_json_exclusive(trial_dir / "experiment_summary.json", summary)
    return summary


def _execution_metrics(
    events: Sequence[Mapping[str, Any]],
) -> tuple[
    dict[str, list[tuple[float, float]]],
    dict[str, list[float]],
]:
    intervals: dict[str, list[tuple[float, float]]] = defaultdict(list)
    stage_durations: dict[str, list[float]] = defaultdict(list)
    for event in events:
        if event.get("event_type") != "task_execution_finished":
            continue
        start, end = _execution_interval(event)
        node_id = event.get("node_id")
        if isinstance(node_id, str):
            stage_durations[node_id].append(end - start)
        replica_id = event.get("replica_id")
        if isinstance(replica_id, str):
            intervals[replica_id].append((start, end))
    return intervals, stage_durations


def _stage_intervals(
    events: Sequence[Mapping[str, Any]],
) -> dict[str, list[tuple[float, float]]]:
    intervals: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for event in events:
        if event.get("event_type") != "task_execution_finished":
            continue
        node_id = event.get("node_id")
        if isinstance(node_id, str):
            intervals[node_id].append(_execution_interval(event))
    return intervals


def _execution_interval(event: Mapping[str, Any]) -> tuple[float, float]:
    payload = _payload(event)
    started_at = payload.get("started_at")
    finished_at = payload.get("finished_at")
    if started_at is None or finished_at is None:
        end = _number(event["ts"])
        start = end - _number(payload["duration_sec"])
    else:
        start = _number(started_at)
        end = _number(finished_at)
    if end < start:
        raise ValueError("task execution finish precedes its start")
    return start, end


def _resident_intervals(
    events: Sequence[Mapping[str, Any]],
    run_finished: float,
) -> dict[str, list[tuple[float, float]]]:
    load_starts: dict[str, float] = {}
    resident_starts: dict[str, float] = {}
    intervals: dict[str, list[tuple[float, float]]] = defaultdict(list)
    ended: set[str] = set()
    for event in events:
        replica_id = event.get("replica_id")
        if not isinstance(replica_id, str):
            continue
        event_type = event.get("event_type")
        timestamp = _number(event["ts"])
        if event_type == "model_load_started":
            load_starts[replica_id] = timestamp
        elif event_type == "model_load_finished":
            resident_starts[replica_id] = timestamp
        elif event_type in ("model_eviction_started", "model_evicted"):
            if replica_id in ended:
                continue
            start = resident_starts.pop(
                replica_id, load_starts.get(replica_id, timestamp)
            )
            intervals[replica_id].append((start, timestamp))
            load_starts.pop(replica_id, None)
            ended.add(replica_id)
    for replica_id in set(load_starts) | set(resident_starts):
        if replica_id in ended:
            continue
        start = resident_starts.get(replica_id, load_starts[replica_id])
        intervals[replica_id].append((start, run_finished))
    return intervals


def _paired_durations(
    events: Sequence[Mapping[str, Any]],
    start_event: str,
    end_event: str,
    keys: Sequence[str],
    *,
    cancel_events: Sequence[str] = (),
) -> tuple[list[float], int, int]:
    starts: dict[tuple[object, ...], float] = {}
    durations = []
    unmatched_ends = 0
    for event in events:
        event_type = event.get("event_type")
        identity = tuple(event.get(key) for key in keys)
        if any(value is None for value in identity):
            continue
        if event_type == start_event:
            if identity in starts:
                raise ValueError(f"duplicate {start_event} identity")
            starts[identity] = _number(event["ts"])
        elif event_type == end_event:
            started_at = starts.pop(identity, None)
            if started_at is None:
                unmatched_ends += 1
                continue
            duration = _number(event["ts"]) - started_at
            if duration < 0:
                raise ValueError(f"{end_event} precedes {start_event}")
            durations.append(duration)
        elif event_type in cancel_events:
            if starts.pop(identity, None) is None:
                unmatched_ends += 1
    return durations, len(starts), unmatched_ends


def _item_enqueue_blocked_intervals(
    events: Sequence[Mapping[str, Any]],
) -> tuple[list[tuple[float, float]], int, int]:
    starts: dict[tuple[object, ...], float] = {}
    intervals = []
    unmatched_enqueues = 0
    keys = ("item_id", "source_node", "target_node")
    for event in events:
        event_type = event.get("event_type")
        identity = tuple(event.get(key) for key in keys)
        if any(value is None for value in identity):
            continue
        if event_type == "item_emitted":
            if identity in starts:
                raise ValueError("duplicate item_emitted identity")
            starts[identity] = _number(event["ts"])
        elif event_type == "item_enqueued":
            start = starts.pop(identity, None)
            if start is None:
                unmatched_enqueues += 1
                continue
            end = _number(event["ts"])
            if end < start:
                raise ValueError("item_enqueued precedes item_emitted")
            intervals.append((start, end))
    return intervals, len(starts), unmatched_enqueues


def _fanin_wait_durations(
    events: Sequence[Mapping[str, Any]],
) -> tuple[list[float], int]:
    starts: dict[tuple[object, object], float] = {}
    durations = []
    for event in events:
        identity = (event.get("session_id"), event.get("node_id"))
        if None in identity:
            continue
        event_type = event.get("event_type")
        if event_type == "fanin_wait":
            starts.setdefault(identity, _number(event["ts"]))
        elif event_type == "fanin_ready" and identity in starts:
            duration = _number(event["ts"]) - starts.pop(identity)
            if duration < 0:
                raise ValueError("fanin_ready precedes fanin_wait")
            durations.append(duration)
    return durations, len(starts)


def _inference_payload_metrics(
    events: Sequence[Mapping[str, Any]],
) -> tuple[list[float], list[float], int, int, int]:
    queue_times = []
    ttfts = []
    input_tokens = 0
    output_tokens = 0
    token_limit_hits = 0
    for event in events:
        if event.get("event_type") != "task_execution_finished":
            continue
        payload = _payload(event)
        input_value = payload.get("input_tokens")
        output_value = payload.get("output_tokens")
        if input_value is None and output_value is None:
            continue
        if input_value is not None:
            input_tokens += _integer(input_value)
        if output_value is not None:
            output_tokens += _integer(output_value)
        queue_time = payload.get("queue_time_sec")
        ttft = payload.get("time_to_first_token_sec")
        if queue_time is not None:
            queue_times.append(_number(queue_time))
        if ttft is not None:
            ttfts.append(_number(ttft))
        if payload.get("hit_token_limit") is True:
            token_limit_hits += 1
    return queue_times, ttfts, input_tokens, output_tokens, token_limit_hits


def _lifecycle_metrics(
    events: Sequence[Mapping[str, Any]],
    run_finished: float,
    execution_intervals: Mapping[str, Sequence[tuple[float, float]]],
) -> dict[str, object]:
    load_durations = [
        _number(_payload(event)["duration_sec"])
        for event in events
        if event.get("event_type") == "model_load_finished"
    ]
    prefetch_durations = [
        _number(_payload(event)["duration_sec"])
        for event in events
        if event.get("event_type") == "prefetch_finished"
    ]
    loading_intervals = _lifecycle_intervals(
        events,
        "model_load_started",
        ("model_load_finished", "model_load_failed"),
        run_finished,
    )
    evicting_intervals = _lifecycle_intervals(
        events,
        "model_eviction_started",
        ("model_evicted", "model_eviction_failed"),
        run_finished,
    )
    resident_intervals = _resident_intervals(events, run_finished)
    trace_peaks = _trace_inflight_peaks(events)
    overlap_peaks = {
        replica_id: _peak_overlapping_intervals(intervals)
        for replica_id, intervals in execution_intervals.items()
    }
    replica_ids = set(trace_peaks) | set(overlap_peaks)
    peak_by_replica = {
        replica_id: max(
            trace_peaks.get(replica_id, 0), overlap_peaks.get(replica_id, 0)
        )
        for replica_id in sorted(replica_ids)
    }
    return {
        "loading_gpu_seconds": sum(
            interval_union_duration(values) for values in loading_intervals.values()
        ),
        "evicting_gpu_seconds": sum(
            interval_union_duration(values) for values in evicting_intervals.values()
        ),
        "model_load_count": _event_count(events, "model_load_finished"),
        "model_reuse_count": _event_count(events, "model_reused"),
        "model_eviction_count": _event_count(events, "model_evicted"),
        "prefetch_count": _event_count(events, "prefetch_started"),
        "prefetch_skip_count": _event_count(events, "prefetch_skipped"),
        "wasted_prefetch_count": _wasted_prefetch_count(events),
        "model_load_duration_sec": _timing_distribution(load_durations),
        "prefetch_load_duration_sec": _timing_distribution(prefetch_durations),
        "peak_replica_inflight": max(peak_by_replica.values(), default=0),
        "peak_replica_inflight_by_replica": peak_by_replica,
        "peak_resident_replica_count": _peak_across_intervals(
            interval
            for intervals in resident_intervals.values()
            for interval in intervals
        ),
    }


def _lifecycle_intervals(
    events: Sequence[Mapping[str, Any]],
    start_event: str,
    end_events: Sequence[str],
    run_finished: float,
) -> dict[str, list[tuple[float, float]]]:
    starts: dict[str, float] = {}
    intervals: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for event in events:
        replica_id = event.get("replica_id")
        if not isinstance(replica_id, str):
            continue
        event_type = event.get("event_type")
        if event_type == start_event:
            starts[replica_id] = _number(event["ts"])
        elif event_type in end_events and replica_id in starts:
            intervals[replica_id].append((starts.pop(replica_id), _number(event["ts"])))
    for replica_id, started_at in starts.items():
        intervals[replica_id].append((started_at, run_finished))
    return intervals


def _trace_inflight_peaks(
    events: Sequence[Mapping[str, Any]],
) -> dict[str, int]:
    peaks: dict[str, int] = defaultdict(int)
    for event in events:
        replica_id = event.get("replica_id")
        if not isinstance(replica_id, str):
            continue
        payload = _payload(event)
        for key in (
            "replica_inflight_at_start",
            "replica_inflight_at_grant",
            "admitted_batch_size",
        ):
            value = payload.get(key)
            if value is not None:
                peaks[replica_id] = max(peaks[replica_id], _integer(value))
    return dict(peaks)


def _wasted_prefetch_count(events: Sequence[Mapping[str, Any]]) -> int:
    prefetched: set[str] = set()
    wasted = 0
    for event in events:
        replica_id = event.get("replica_id")
        if not isinstance(replica_id, str):
            continue
        event_type = event.get("event_type")
        if event_type == "prefetch_finished":
            prefetched.add(replica_id)
        elif event_type == "acquire_granted":
            prefetched.discard(replica_id)
        elif event_type == "model_evicted" and replica_id in prefetched:
            prefetched.remove(replica_id)
            wasted += 1
    return wasted + len(prefetched)


def _steady_state_rate(
    completion_times: Sequence[float],
    *,
    start_fraction: float,
    end_fraction: float,
) -> float:
    start = math.floor(len(completion_times) * start_fraction)
    end = math.ceil(len(completion_times) * end_fraction)
    if end - start < 2:
        return 0.0
    return steady_state_completion_slope(
        completion_times,
        start_fraction=start_fraction,
        end_fraction=end_fraction,
    )


def _integrate_gpu_metric(telemetry_path: Path, field: str) -> float:
    samples: dict[int, list[tuple[float, float]]] = defaultdict(list)
    for row in read_jsonl(telemetry_path):
        samples[_integer(row["gpu_index"])].append(
            (_number(row["ts"]), _number(row[field]))
        )
    total = 0.0
    for values in samples.values():
        ordered = sorted(values)
        total += sum(
            (right_ts - left_ts) * (left_value + right_value) / 2.0
            for (left_ts, left_value), (right_ts, right_value) in pairwise(ordered)
        )
    return total


def _integrate_idle_subtracted_gpu_metric(
    telemetry_path: Path,
    field: str,
) -> float:
    samples: dict[int, list[tuple[float, float]]] = defaultdict(list)
    for row in read_jsonl(telemetry_path):
        samples[_integer(row["gpu_index"])].append(
            (_number(row["ts"]), _number(row[field]))
        )
    total = 0.0
    for values in samples.values():
        ordered = sorted(values)
        baseline = ordered[0][1]
        adjusted = [
            (timestamp, max(0.0, value - baseline)) for timestamp, value in ordered
        ]
        total += sum(
            (right_ts - left_ts) * (left_value + right_value) / 2.0
            for (left_ts, left_value), (right_ts, right_value) in pairwise(adjusted)
        )
    return total


def _first_gpu_values(
    rows: Sequence[Mapping[str, Any]],
    field: str,
) -> dict[str, float]:
    samples: dict[int, list[tuple[float, float]]] = defaultdict(list)
    for row in rows:
        samples[_integer(row["gpu_index"])].append(
            (_number(row["ts"]), _number(row[field]))
        )
    return {
        str(gpu_index): min(values)[1] for gpu_index, values in sorted(samples.items())
    }


def _event_count(events: Sequence[Mapping[str, Any]], event_type: str) -> int:
    return sum(event.get("event_type") == event_type for event in events)


def _is_static_baseline(events: Sequence[Mapping[str, Any]]) -> bool:
    return any(
        event.get("event_type") == "model_load_started"
        and _payload(event).get("reason") == "static_baseline"
        for event in events
    )


def _peak_overlapping_intervals(intervals: Iterable[tuple[float, float]]) -> int:
    return _peak_across_intervals(intervals)


def _peak_across_intervals(intervals: Iterable[tuple[float, float]]) -> int:
    points = []
    for start, end in intervals:
        if end < start:
            raise ValueError("interval end precedes its start")
        if end == start:
            continue
        points.extend(((start, 1), (end, -1)))
    current = 0
    peak = 0
    for _, delta in sorted(points, key=lambda point: (point[0], point[1])):
        current += delta
        peak = max(peak, current)
    return peak


def _interval_intersection_duration(
    left: Iterable[tuple[float, float]],
    right: Iterable[tuple[float, float]],
) -> float:
    left_intervals = _merge_intervals(left)
    right_intervals = _merge_intervals(right)
    left_index = 0
    right_index = 0
    total = 0.0
    while left_index < len(left_intervals) and right_index < len(right_intervals):
        left_start, left_end = left_intervals[left_index]
        right_start, right_end = right_intervals[right_index]
        total += max(0.0, min(left_end, right_end) - max(left_start, right_start))
        if left_end <= right_end:
            left_index += 1
        else:
            right_index += 1
    return total


def _clip_intervals(
    intervals: Iterable[tuple[float, float]],
    window_start: float,
    window_end: float,
) -> list[tuple[float, float]]:
    if window_end < window_start:
        raise ValueError("interval window end precedes its start")
    return [
        (max(start, window_start), min(end, window_end))
        for start, end in intervals
        if max(start, window_start) < min(end, window_end)
    ]


def _merge_intervals(
    intervals: Iterable[tuple[float, float]],
) -> list[tuple[float, float]]:
    ordered = sorted(intervals)
    if not ordered:
        return []
    start, end = ordered[0]
    if end < start:
        raise ValueError("interval end precedes its start")
    merged = []
    for next_start, next_end in ordered[1:]:
        if next_end < next_start:
            raise ValueError("interval end precedes its start")
        if next_start <= end:
            end = max(end, next_end)
            continue
        merged.append((start, end))
        start, end = next_start, next_end
    merged.append((start, end))
    return merged


def _single_timestamp(events: Sequence[Mapping[str, Any]], event_type: str) -> float:
    values = [
        _number(event["ts"])
        for event in events
        if event.get("event_type") == event_type
    ]
    if len(values) != 1:
        raise ValueError(f"trace requires exactly one {event_type} event")
    return values[0]


def _payload(event: Mapping[str, Any]) -> Mapping[str, Any]:
    payload = event.get("payload")
    if not isinstance(payload, dict):
        raise TypeError("trace payload must be an object")
    return payload


def _number(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"expected a number, got {type(value)}")
    return float(value)


def _integer(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"expected an integer, got {type(value)}")
    return value


def _percentiles(values: Sequence[float]) -> dict[str, float]:
    if not values:
        return {"p50": 0.0, "p95": 0.0, "max": 0.0}
    ordered = sorted(values)
    return {
        "p50": ordered[math.ceil(0.50 * len(ordered)) - 1],
        "p95": ordered[math.ceil(0.95 * len(ordered)) - 1],
        "max": ordered[-1],
    }


def _distribution(values: Sequence[float]) -> dict[str, float]:
    return {
        "mean": statistics.fmean(values),
        **_percentiles(values),
    }


def _timing_distribution(values: Sequence[float]) -> dict[str, float | int]:
    return {
        "count": len(values),
        "total": sum(values),
        "mean": statistics.fmean(values) if values else 0.0,
        **_percentiles(values),
    }
