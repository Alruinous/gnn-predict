from __future__ import annotations

import json
from pathlib import Path
from typing import Any, TextIO, cast
from uuid import UUID

import langchain_core
import pytest
import ray
from langchain.agents import AgentState
from langchain_core.load import dumpd
from langchain_core.messages import AIMessage
from pydantic import ValidationError

from workflow.storage import (
    ResultStore,
    ResultStoreActor,
    TraceWriter,
    TraceWriterActor,
    write_run_summary,
)
from workflow.types import TraceEvent


class FailingResultFile:
    def __init__(self, wrapped: TextIO, operation: str) -> None:
        self.wrapped = wrapped
        self.operation = operation

    def write(self, value: str) -> int:
        if self.operation == "write":
            raise OSError("result write failed")
        return self.wrapped.write(value)

    def flush(self) -> None:
        if self.operation == "flush":
            raise OSError("result flush failed")
        self.wrapped.flush()

    def close(self) -> None:
        self.wrapped.close()


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def _event(
    event_seq: int,
    event_type: str,
    *,
    run_id: str = "run-1",
    ts: float = 0.0,
    node_id: str | None = None,
    replica_id: str | None = None,
    accelerator_id: str | None = None,
    payload: dict[str, object] | None = None,
) -> TraceEvent:
    return TraceEvent(
        run_id=run_id,
        event_seq=event_seq,
        event_type=event_type,
        ts=ts,
        node_id=node_id,
        replica_id=replica_id,
        accelerator_id=accelerator_id,
        payload=payload or {},
    )


def test_trace_event_defaults_identity_fields_and_validates_sequence() -> None:
    event = _event(0, "future_event")

    UUID(event.event_id)
    assert event.session_id is None
    assert event.item_id is None
    assert event.task_id is None
    assert event.source_node is None
    assert event.target_node is None
    assert event.model_key is None
    assert event.gpu_kind is None
    assert event.acquire_id is None
    with pytest.raises(ValidationError):
        _event(-1, "run_started")


def test_result_store_persists_each_session_once(tmp_path: Path) -> None:
    store = ResultStore(tmp_path, "run-1")
    state = AgentState(messages=[AIMessage(content="answer")])
    try:
        assert not store.has_result("session-1")
        store.put("session-1", "merge", state)

        assert store.has_result("session-1")
        assert store.get_result("session-1") == state
        assert store.result_count() == 1
        with pytest.raises(ValueError, match="already has a result"):
            store.put("session-1", "merge", state)
    finally:
        store.close()

    rows = _read_jsonl(tmp_path / "session_results.jsonl")
    assert len(rows) == 1
    assert rows[0]["run_id"] == "run-1"
    assert rows[0]["session_id"] == "session-1"
    assert rows[0]["node_id"] == "merge"
    assert rows[0]["serialization_format"] == "langchain_dump"
    assert rows[0]["langchain_core_version"] == langchain_core.__version__
    assert rows[0]["state"] == dumpd(state)


def test_result_store_rejects_nested_not_implemented_marker(tmp_path: Path) -> None:
    store = ResultStore(tmp_path, "run-1")
    state = cast(
        AgentState,
        {"messages": [], "nested": [{"unsupported": object()}]},
    )
    try:
        with pytest.raises(AssertionError):
            store.put("session-1", "merge", state)
        assert store.result_count() == 0
    finally:
        store.close()

    assert (tmp_path / "session_results.jsonl").read_text() == ""


@pytest.mark.parametrize("operation", ["write", "flush"])
def test_result_store_exposes_run_level_persistence_failure_without_caching(
    tmp_path: Path,
    operation: str,
) -> None:
    from workflow.storage import ResultPersistenceError

    store = ResultStore(tmp_path, "run-1")
    store._file = cast(TextIO, FailingResultFile(store._file, operation))
    state = AgentState(messages=[AIMessage(content="answer")])
    try:
        with pytest.raises(ResultPersistenceError, match=operation) as captured:
            store.put("session-1", "merge", state)

        assert isinstance(captured.value.__cause__, OSError)
        assert not store.has_result("session-1")
        assert store.result_count() == 0
    finally:
        store.close()


def test_result_store_uses_exclusive_create(tmp_path: Path) -> None:
    store = ResultStore(tmp_path, "run-1")
    store.close()

    with pytest.raises(FileExistsError):
        ResultStore(tmp_path, "run-1")


def test_trace_writer_requires_matching_run_and_contiguous_sequence(
    tmp_path: Path,
) -> None:
    writer = TraceWriter(tmp_path, "run-1")
    try:
        with pytest.raises(ValueError, match="expected event_seq 0"):
            writer.append_batch([_event(1, "run_started")])
        with pytest.raises(ValueError, match="run_id"):
            writer.append_batch([_event(0, "run_started", run_id="run-2")])

        assert writer.append_batch([_event(0, "run_started")]) == 0
        with pytest.raises(ValueError, match="expected event_seq 1"):
            writer.append_batch([_event(0, "session_submitted")])
        with pytest.raises(ValueError, match="expected event_seq 1"):
            writer.append_batch([_event(2, "session_submitted")])
        assert writer.append_batch([_event(1, "run_finished")]) == 1
    finally:
        writer.close()

    rows = _read_jsonl(tmp_path / "workflow_trace.jsonl")
    assert [row["event_seq"] for row in rows] == [0, 1]
    assert [row["event_type"] for row in rows] == ["run_started", "run_finished"]


def test_trace_writer_rejects_non_json_payload_without_partial_batch(
    tmp_path: Path,
) -> None:
    writer = TraceWriter(tmp_path, "run-1")
    try:
        with pytest.raises(TypeError):
            writer.append_batch(
                [
                    _event(0, "run_started"),
                    _event(1, "scheduler_decision", payload={"value": object()}),
                ]
            )
    finally:
        writer.close()

    assert (tmp_path / "workflow_trace.jsonl").read_text() == ""


def test_write_run_summary_aggregates_trace_and_results(tmp_path: Path) -> None:
    store = ResultStore(tmp_path, "run-1")
    try:
        store.put(
            "session-1",
            "merge",
            AgentState(messages=[AIMessage(content="answer")]),
        )
    finally:
        store.close()

    events = [
        _event(0, "run_started", ts=0.0),
        _event(1, "session_submitted", ts=0.1),
        _event(2, "session_submitted", ts=0.2),
        _event(
            3,
            "task_completed",
            ts=1.0,
            node_id="agent",
            accelerator_id="gpu-0",
            payload={"duration_sec": 2.5},
        ),
        _event(
            4,
            "task_failed",
            ts=1.1,
            node_id="agent",
            payload={"status": "oom"},
        ),
        _event(
            5,
            "task_completed",
            ts=1.2,
            node_id="merge",
            payload={"duration_sec": 9.0},
        ),
        _event(6, "token_budget_selected", payload={"action": "fixed"}),
        _event(7, "token_budget_selected", payload={"action": "upscaled"}),
        _event(8, "token_budget_selected", payload={"action": "downscaled"}),
        _event(9, "token_budget_infeasible"),
        _event(10, "model_load_started", ts=1.0, replica_id="replica-1"),
        _event(11, "model_load_finished", ts=2.0, replica_id="replica-1"),
        _event(12, "model_reused", ts=3.0, replica_id="replica-1"),
        _event(13, "model_evicted", ts=5.0, replica_id="replica-1"),
        _event(14, "model_load_started", ts=6.0, replica_id="replica-2"),
        _event(15, "session_completed", ts=8.0, payload={"latency_sec": 3.0}),
        _event(16, "session_failed", ts=9.0, payload={"latency_sec": 3.0}),
        _event(17, "run_finished", ts=10.0),
    ]
    writer = TraceWriter(tmp_path, "run-1")
    try:
        writer.append_batch(events)
    finally:
        writer.close()

    write_run_summary(tmp_path, "run-1")

    summary = json.loads((tmp_path / "run_summary.json").read_text())
    assert summary == {
        "run_id": "run-1",
        "submitted_session_count": 2,
        "completed_session_count": 1,
        "failed_session_count": 1,
        "result_count": 1,
        "per_node_task_counts": {
            "agent": {"completed": 1, "failed": 1},
            "merge": {"completed": 1, "failed": 0},
        },
        "token_budget_action_counts": {
            "fixed": 1,
            "upscaled": 1,
            "downscaled": 1,
            "infeasible": 1,
        },
        "oom_count": 1,
        "model_load_count": 1,
        "model_reuse_count": 1,
        "model_evict_count": 1,
        "active_gpu_seconds": 2.5,
        "resident_gpu_seconds": 8.0,
        "session_latency_sec": {"p50": 3.0, "p95": 3.0, "max": 3.0},
    }
    with pytest.raises(FileExistsError):
        write_run_summary(tmp_path, "run-1")


def test_write_run_summary_rejects_trace_for_another_run(tmp_path: Path) -> None:
    store = ResultStore(tmp_path, "run-1")
    store.close()
    writer = TraceWriter(tmp_path, "run-2")
    try:
        writer.append_batch([_event(0, "run_finished", run_id="run-2", ts=1.0)])
    finally:
        writer.close()

    with pytest.raises(ValueError, match="trace run_id"):
        write_run_summary(tmp_path, "run-1")
    assert not (tmp_path / "run_summary.json").exists()


def test_storage_actors_persist_through_ray(ray_session: None, tmp_path: Path) -> None:
    result_actor: Any = ResultStoreActor.remote(tmp_path, "run-actor")
    trace_actor: Any = TraceWriterActor.remote(tmp_path, "run-actor")
    state = AgentState(messages=[AIMessage(content="actor answer")])
    try:
        ray.get(result_actor.put.remote("session-1", "merge", state))
        assert ray.get(result_actor.get_result.remote("session-1")) == state
        assert ray.get(result_actor.result_count.remote()) == 1
        acknowledged = ray.get(
            trace_actor.append_batch.remote(
                [
                    TraceEvent(
                        run_id="run-actor",
                        event_seq=0,
                        event_type="run_finished",
                        ts=1.0,
                    )
                ]
            )
        )
        assert acknowledged == 0
    finally:
        ray.get([result_actor.close.remote(), trace_actor.close.remote()])

    assert len(_read_jsonl(tmp_path / "session_results.jsonl")) == 1
    assert len(_read_jsonl(tmp_path / "workflow_trace.jsonl")) == 1
