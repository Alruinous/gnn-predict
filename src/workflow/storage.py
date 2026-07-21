from __future__ import annotations

import json
from collections.abc import Iterator, Sequence
from math import ceil
from pathlib import Path
from typing import TextIO, cast

import langchain_core
import ray
from langchain.agents import AgentState
from langchain_core.load import dumpd

from workflow.types import TraceEvent

RESULTS_FILENAME = "session_results.jsonl"
TRACE_FILENAME = "workflow_trace.jsonl"
SUMMARY_FILENAME = "run_summary.json"


class ResultPersistenceError(RuntimeError):
    pass


def _contains_not_implemented(value: object) -> bool:
    if isinstance(value, dict):
        mapping = cast(dict[object, object], value)
        if mapping.get("type") == "not_implemented":
            return True
        return any(_contains_not_implemented(item) for item in mapping.values())
    if isinstance(value, list):
        return any(_contains_not_implemented(item) for item in value)
    return False


class ResultStore:
    def __init__(self, output_dir: Path | str, run_id: str) -> None:
        self.run_id = run_id
        self._results: dict[str, AgentState] = {}
        self._file: TextIO = (Path(output_dir) / RESULTS_FILENAME).open(
            "x", encoding="utf-8"
        )

    def put(self, session_id: str, node_id: str, state: AgentState) -> None:
        if session_id in self._results:
            raise ValueError(f"session {session_id!r} already has a result")
        serialized_state: object = dumpd(state)
        assert not _contains_not_implemented(serialized_state), serialized_state
        row: dict[str, object] = {
            "run_id": self.run_id,
            "session_id": session_id,
            "node_id": node_id,
            "serialization_format": "langchain_dump",
            "langchain_core_version": langchain_core.__version__,
            "state": serialized_state,
        }
        line = json.dumps(row)
        try:
            self._file.write(f"{line}\n")
        except (OSError, ValueError) as error:
            raise ResultPersistenceError("result write failed") from error
        try:
            self._file.flush()
        except (OSError, ValueError) as error:
            raise ResultPersistenceError("result flush failed") from error
        self._results[session_id] = state

    def has_result(self, session_id: str) -> bool:
        return session_id in self._results

    def get_result(self, session_id: str) -> AgentState:
        return self._results[session_id]

    def result_count(self) -> int:
        return len(self._results)

    def close(self) -> None:
        self._file.close()


ResultStoreActor = ray.remote(ResultStore)


class TraceWriter:
    def __init__(self, output_dir: Path | str, run_id: str) -> None:
        self.run_id = run_id
        self._next_event_seq = 0
        self._file: TextIO = (Path(output_dir) / TRACE_FILENAME).open(
            "x", encoding="utf-8"
        )

    def append_batch(self, events: list[TraceEvent]) -> int:
        expected_seq = self._next_event_seq
        for event in events:
            if event.run_id != self.run_id:
                raise ValueError(
                    f"event run_id {event.run_id!r} does not match {self.run_id!r}"
                )
            if event.event_seq != expected_seq:
                raise ValueError(
                    f"expected event_seq {expected_seq}, got {event.event_seq}"
                )
            expected_seq += 1

        lines = [json.dumps(event.model_dump()) for event in events]
        for line in lines:
            self._file.write(f"{line}\n")
        self._file.flush()
        self._next_event_seq = expected_seq
        return expected_seq - 1

    def close(self) -> None:
        self._file.close()


TraceWriterActor = ray.remote(TraceWriter)


def _read_jsonl(path: Path) -> Iterator[dict[str, object]]:
    with path.open(encoding="utf-8") as file:
        for line in file:
            value: object = json.loads(line)
            if not isinstance(value, dict):
                raise TypeError(f"expected JSON object in {path}, got {type(value)}")
            yield cast(dict[str, object], value)


def _session_latency(event: TraceEvent) -> float:
    value = event.payload["latency_sec"]
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
        raise TypeError(f"invalid latency_sec for event {event.event_id}")
    return float(value)


def _execution_interval(event: TraceEvent) -> tuple[float, float]:
    started_at = event.payload["started_at"]
    finished_at = event.payload["finished_at"]
    if (
        isinstance(started_at, bool)
        or not isinstance(started_at, (int, float))
        or isinstance(finished_at, bool)
        or not isinstance(finished_at, (int, float))
        or finished_at < started_at
    ):
        raise TypeError(f"invalid execution interval for event {event.event_id}")
    return float(started_at), float(finished_at)


def _interval_union_duration(intervals: list[tuple[float, float]]) -> float:
    if not intervals:
        return 0.0
    ordered = sorted(intervals)
    merged_start, merged_end = ordered[0]
    duration = 0.0
    for started_at, finished_at in ordered[1:]:
        if started_at > merged_end:
            duration += merged_end - merged_start
            merged_start, merged_end = started_at, finished_at
        else:
            merged_end = max(merged_end, finished_at)
    return duration + merged_end - merged_start


def _percentile(values: list[float], quantile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[ceil(quantile * len(ordered)) - 1]


def write_run_summary(
    output_dir: Path | str,
    run_id: str,
    *,
    result_directories: Sequence[Path | str] | None = None,
) -> None:
    directory = Path(output_dir)
    directories = (
        [directory]
        if result_directories is None
        else [Path(item) for item in result_directories]
    )
    result_count = 0
    for result_directory in directories:
        for result in _read_jsonl(result_directory / RESULTS_FILENAME):
            if result.get("run_id") != run_id:
                raise ValueError(
                    f"result run_id {result.get('run_id')!r} does not match {run_id!r}"
                )
            result_count += 1

    submitted_session_count = 0
    completed_session_count = 0
    failed_session_count = 0
    per_node_task_counts: dict[str, dict[str, int]] = {}
    placement_count = 0
    request_infeasible_count = 0
    oom_count = 0
    model_load_count = 0
    model_reuse_count = 0
    model_evict_count = 0
    batched_admission_count = 0
    peak_replica_inflight = 0
    active_intervals: dict[str, list[tuple[float, float]]] = {}
    resident_gpu_seconds = 0.0
    residency_starts: dict[str, float] = {}
    run_finished_ts: float | None = None
    session_latencies: list[float] = []

    for raw_event in _read_jsonl(directory / TRACE_FILENAME):
        event = TraceEvent.model_validate(raw_event)
        if event.run_id != run_id:
            raise ValueError(f"trace run_id {event.run_id!r} does not match {run_id!r}")

        if event.event_type == "session_submitted":
            submitted_session_count += 1
        elif event.event_type == "session_completed":
            completed_session_count += 1
            session_latencies.append(_session_latency(event))
        elif event.event_type == "session_failed":
            failed_session_count += 1
            session_latencies.append(_session_latency(event))
        elif event.event_type == "task_execution_finished":
            if event.accelerator_id is not None:
                if event.replica_id is None:
                    raise ValueError("GPU execution event requires replica_id")
                active_intervals.setdefault(event.replica_id, []).append(
                    _execution_interval(event)
                )
            if event.payload.get("status") == "oom":
                oom_count += 1
        elif event.event_type in ("task_completed", "task_failed"):
            if event.node_id is None:
                raise ValueError(f"{event.event_type} requires node_id")
            node_counts = per_node_task_counts.setdefault(
                event.node_id, {"completed": 0, "failed": 0}
            )
            status = "completed" if event.event_type == "task_completed" else "failed"
            node_counts[status] += 1
            if (
                event.event_type == "task_failed"
                and event.payload.get("status") == "oom"
                and not event.payload.get("execution_event_recorded", False)
            ):
                oom_count += 1
        elif event.event_type == "placement_selected":
            placement_count += 1
            admitted_batch_size = event.payload.get("admitted_batch_size", 1)
            if (
                isinstance(admitted_batch_size, bool)
                or not isinstance(admitted_batch_size, int)
                or admitted_batch_size < 1
            ):
                raise TypeError("invalid admitted_batch_size")
            peak_replica_inflight = max(
                peak_replica_inflight,
                admitted_batch_size,
            )
            if admitted_batch_size > 1:
                batched_admission_count += 1
        elif event.event_type == "request_infeasible":
            request_infeasible_count += 1
        elif event.event_type == "model_load_started":
            if event.replica_id is not None:
                residency_starts.setdefault(event.replica_id, event.ts)
        elif event.event_type == "model_load_finished":
            model_load_count += 1
        elif event.event_type == "model_reused":
            model_reuse_count += 1
        elif event.event_type == "model_evicted":
            model_evict_count += 1
            if event.replica_id in residency_starts:
                assert event.replica_id is not None
                resident_gpu_seconds += event.ts - residency_starts.pop(
                    event.replica_id
                )
        elif event.event_type == "run_finished":
            run_finished_ts = event.ts

    if residency_starts:
        if run_finished_ts is None:
            raise ValueError("run_finished is required for resident model accounting")
        resident_gpu_seconds += sum(
            run_finished_ts - started_at for started_at in residency_starts.values()
        )
    active_gpu_seconds = sum(
        _interval_union_duration(intervals) for intervals in active_intervals.values()
    )

    summary: dict[str, object] = {
        "run_id": run_id,
        "submitted_session_count": submitted_session_count,
        "completed_session_count": completed_session_count,
        "failed_session_count": failed_session_count,
        "result_count": result_count,
        "per_node_task_counts": per_node_task_counts,
        "placement_count": placement_count,
        "request_infeasible_count": request_infeasible_count,
        "oom_count": oom_count,
        "model_load_count": model_load_count,
        "model_reuse_count": model_reuse_count,
        "model_evict_count": model_evict_count,
        "batched_admission_count": batched_admission_count,
        "peak_replica_inflight": peak_replica_inflight,
        "active_gpu_seconds": active_gpu_seconds,
        "resident_gpu_seconds": resident_gpu_seconds,
        "session_latency_sec": {
            "p50": _percentile(session_latencies, 0.5),
            "p95": _percentile(session_latencies, 0.95),
            "max": max(session_latencies, default=0.0),
        },
    }
    with (directory / SUMMARY_FILENAME).open("x", encoding="utf-8") as file:
        file.write(json.dumps(summary))
        file.flush()
