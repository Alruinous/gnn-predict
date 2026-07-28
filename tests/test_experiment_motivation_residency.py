from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from experiment.workflow.motivation_residency import analyze_residency, belady_misses

WORKFLOW = "wf"


def event(event_type: str, ts: float, **fields: Any) -> dict[str, Any]:
    row: dict[str, Any] = {"event_type": event_type, "ts": ts}
    row.setdefault("workflow_name", WORKFLOW)
    row.update(fields)
    return row


def agent_task(
    *,
    session_id: str,
    node_id: str,
    task_id: str,
    acquire_id: str,
    replica_id: str,
    model_key: str,
    requested_at: float,
    granted_at: float,
    finished_at: float,
    gpu_kind: str = "v100",
) -> list[dict[str, Any]]:
    return [
        event(
            "acquire_requested",
            requested_at,
            session_id=session_id,
            node_id=node_id,
            task_id=task_id,
            acquire_id=acquire_id,
        ),
        event("task_started", granted_at, session_id=session_id, node_id=node_id),
        event(
            "acquire_granted",
            granted_at,
            session_id=session_id,
            node_id=node_id,
            task_id=task_id,
            acquire_id=acquire_id,
            replica_id=replica_id,
            model_key=model_key,
            gpu_kind=gpu_kind,
        ),
        event(
            "task_execution_finished",
            finished_at,
            session_id=session_id,
            node_id=node_id,
            task_id=task_id,
            replica_id=replica_id,
            payload={"started_at": granted_at, "finished_at": finished_at},
        ),
    ]


def model_load(
    *,
    replica_id: str,
    model_key: str,
    accelerator_id: str,
    started_at: float,
    finished_at: float,
    reason: str = "ready_load",
) -> list[dict[str, Any]]:
    return [
        event(
            "model_load_started",
            started_at,
            workflow_name=None,
            replica_id=replica_id,
            model_key=model_key,
            accelerator_id=accelerator_id,
            payload={"reason": reason},
        ),
        event(
            "model_load_finished",
            finished_at,
            workflow_name=None,
            replica_id=replica_id,
            model_key=model_key,
            accelerator_id=accelerator_id,
        ),
    ]


def write_trace(path: Path, events: list[dict[str, Any]]) -> Path:
    trace = path / "workflow_trace.jsonl"
    with trace.open("w", encoding="utf-8") as stream:
        for row in events:
            stream.write(json.dumps(row) + "\n")
    return trace


def chain_trace(
    *,
    upstream_start: float,
    upstream_finish: float,
    load_start: float,
    load_finish: float,
    competitor: tuple[float, float] | None = None,
) -> list[dict[str, Any]]:
    """Session with up -> down; ``down``'s model is knowable once ``up`` starts.

    ``competitor`` optionally keeps the only eligible card busy for a window, which is
    what separates "the need was foreseeable" from "capacity existed to act on it".
    """
    events = [event("run_started", 0.0, workflow_name=None)]
    events.append(event("session_submitted", 0.0, session_id="s0"))
    events.append(
        event(
            "item_enqueued",
            upstream_start,
            session_id="s0",
            source_node="up",
            target_node="down",
        )
    )
    events.extend(
        agent_task(
            session_id="s0",
            node_id="up",
            task_id="t-up",
            acquire_id="a-up",
            replica_id="r-up",
            model_key="key-up",
            requested_at=upstream_start,
            granted_at=upstream_start,
            finished_at=upstream_finish,
        )
    )
    events.extend(
        model_load(
            replica_id="r-down",
            model_key="key-down",
            accelerator_id="gpu-0",
            started_at=load_start,
            finished_at=load_finish,
        )
    )
    if competitor is not None:
        start, finish = competitor
        events.extend(
            agent_task(
                session_id="s1",
                node_id="other",
                task_id="t-other",
                acquire_id="a-other",
                replica_id="r-other",
                model_key="key-other",
                requested_at=start,
                granted_at=start,
                finished_at=finish,
            )
        )
        events.extend(
            model_load(
                replica_id="r-other",
                model_key="key-other",
                accelerator_id="gpu-0",
                started_at=start - 1.0,
                finished_at=start,
            )
        )
    events.extend(
        agent_task(
            session_id="s0",
            node_id="down",
            task_id="t-down",
            acquire_id="a-down",
            replica_id="r-down",
            model_key="key-down",
            requested_at=upstream_finish,
            granted_at=load_finish,
            finished_at=load_finish + 5.0,
        )
    )
    events.append(event("run_finished", load_finish + 20.0, workflow_name=None))
    return events


def test_belady_matches_optimal_replacement_on_a_known_sequence():
    # A B C B C A at capacity 2: C evicts A (next use furthest), so B and C then hit.
    assert belady_misses(list("ABCBCA"), 2) == 4
    assert belady_misses(list("ABCBCA"), 3) == 3
    assert belady_misses(list("AAAA"), 1) == 1
    assert belady_misses(list("ABAB"), 1) == 4


def test_belady_rejects_a_non_positive_capacity():
    with pytest.raises(ValueError):
        belady_misses(list("AB"), 0)


def test_preload_reports_lead_and_capacity_when_both_are_available(tmp_path):
    # up runs 0->100, down's 10 s load starts at 100: 100 s of lead, card idle throughout.
    trace = write_trace(
        tmp_path,
        chain_trace(
            upstream_start=0.0, upstream_finish=100.0, load_start=100.0, load_finish=110.0
        ),
    )
    preload = analyze_residency(trace)["preload"]
    assert preload["attributed_load_count"] == 1
    assert preload["lead_sufficient_count"] == 1
    assert preload["lead_and_capacity_count"] == 1
    assert preload["hideable_load_sec"] == pytest.approx(10.0)


def test_preload_separates_sufficient_lead_from_missing_capacity(tmp_path):
    # Same 100 s lead, but the only card this model ever used is busy for 0->100.
    trace = write_trace(
        tmp_path,
        chain_trace(
            upstream_start=0.0,
            upstream_finish=100.0,
            load_start=100.0,
            load_finish=110.0,
            competitor=(1.0, 99.5),
        ),
    )
    preload = analyze_residency(trace)["preload"]
    assert preload["lead_sufficient_count"] == 1
    assert preload["lead_and_capacity_count"] == 0
    assert preload["hideable_load_sec"] == pytest.approx(0.0)


def test_preload_reports_insufficient_lead_when_the_load_outlasts_the_bubble(tmp_path):
    # Lead is 2 s, the load takes 10 s: no amount of capacity makes it hideable.
    trace = write_trace(
        tmp_path,
        chain_trace(
            upstream_start=0.0, upstream_finish=2.0, load_start=2.0, load_finish=12.0
        ),
    )
    preload = analyze_residency(trace)["preload"]
    assert preload["lead_sufficient_count"] == 0
    assert preload["lead_and_capacity_fraction"] == pytest.approx(0.0)


def test_demand_concurrency_is_time_weighted_over_distinct_models(tmp_path):
    # key-a spans 0->30, key-b spans 10->20: two models are demanded for 10 s of 30.
    events = [event("run_started", 0.0, workflow_name=None)]
    events.append(event("session_submitted", 0.0, session_id="s0"))
    events.extend(
        agent_task(
            session_id="s0",
            node_id="a",
            task_id="t-a",
            acquire_id="a-a",
            replica_id="r-a",
            model_key="key-a",
            requested_at=0.0,
            granted_at=0.0,
            finished_at=30.0,
        )
    )
    events.extend(
        agent_task(
            session_id="s0",
            node_id="b",
            task_id="t-b",
            acquire_id="a-b",
            replica_id="r-b",
            model_key="key-b",
            requested_at=10.0,
            granted_at=10.0,
            finished_at=20.0,
        )
    )
    events.append(event("run_finished", 40.0, workflow_name=None))
    demand = analyze_residency(write_trace(tmp_path, events))["demand"]

    assert demand["distinct_model_keys"] == 2
    assert demand["min_static_cards"] == 2
    assert demand["seconds_at_concurrency"] == {1: pytest.approx(20.0), 2: pytest.approx(10.0)}
    assert demand["fraction_over_pool_size"][1] == pytest.approx(10.0 / 30.0)
    assert demand["fraction_over_pool_size"][2] == pytest.approx(0.0)


def test_wait_breakdown_attributes_blocking_to_the_right_model(tmp_path):
    # down waits 0->110; 100->110 of that is its own load, 20->60 is an unrelated load.
    events = chain_trace(
        upstream_start=0.0, upstream_finish=0.0, load_start=100.0, load_finish=110.0
    )
    events.extend(
        model_load(
            replica_id="r-noise",
            model_key="key-other",
            accelerator_id="gpu-1",
            started_at=20.0,
            finished_at=60.0,
        )
    )
    breakdown = analyze_residency(write_trace(tmp_path, events))["wait_breakdown"]

    assert breakdown["own_model_load_sec"] == pytest.approx(10.0)
    assert breakdown["head_of_line_other_key_sec"] == pytest.approx(40.0)
    assert breakdown["other_sec"] == pytest.approx(60.0)
