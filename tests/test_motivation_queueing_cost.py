from __future__ import annotations

from pathlib import Path

import pytest

from scripts.motivation.plot_queueing_cost import (
    SessionFusionCost,
    extract_session_costs,
    latency_quantile_session,
    load_baseline_run,
    nearest_rank,
)

SESSION_ID = "qmsum_lane1-0000"


def task_event(
    node_id: str,
    started_at: float,
    finished_at: float,
    replica_id: str,
    model_key: str,
) -> dict[str, object]:
    return {
        "workflow_name": "qmsum_lane1",
        "event_type": "task_execution_finished",
        "session_id": SESSION_ID,
        "node_id": node_id,
        "task_id": f"task-{node_id}",
        "replica_id": replica_id,
        "model_key": model_key,
        "acquire_id": f"acquire-{node_id}",
        "payload": {
            "started_at": started_at,
            "finished_at": finished_at,
        },
    }


def acquire_events(
    node_id: str,
    requested_at: float,
    granted_at: float,
    replica_id: str,
    model_key: str,
) -> list[dict[str, object]]:
    common = {
        "workflow_name": "qmsum_lane1",
        "session_id": SESSION_ID,
        "node_id": node_id,
        "task_id": f"task-{node_id}",
        "acquire_id": f"acquire-{node_id}",
    }
    return [
        {
            **common,
            "event_type": "acquire_requested",
            "ts": requested_at,
        },
        {
            **common,
            "event_type": "acquire_granted",
            "ts": granted_at,
            "replica_id": replica_id,
            "model_key": model_key,
        },
    ]


def agent_events(
    node_id: str,
    started_at: float,
    finished_at: float,
    requested_at: float,
    granted_at: float,
    replica_id: str,
    model_key: str,
) -> list[dict[str, object]]:
    return [
        *acquire_events(
            node_id,
            requested_at,
            granted_at,
            replica_id,
            model_key,
        ),
        task_event(node_id, started_at, finished_at, replica_id, model_key),
    ]


def load_events(
    replica_id: str,
    started_at: float,
    finished_at: float,
    duration_sec: float,
) -> list[dict[str, object]]:
    return [
        {
            "event_type": "model_load_started",
            "replica_id": replica_id,
            "ts": started_at,
            "payload": {"reason": "ready_load"},
        },
        {
            "event_type": "model_load_finished",
            "replica_id": replica_id,
            "ts": finished_at,
            "payload": {"duration_sec": duration_sec},
        },
    ]


def eviction_events(
    replica_id: str,
    started_at: float,
    finished_at: float,
) -> list[dict[str, object]]:
    return [
        {
            "event_type": "model_eviction_started",
            "replica_id": replica_id,
            "ts": started_at,
            "payload": {"reason": "ready_load"},
        },
        {
            "event_type": "model_evicted",
            "replica_id": replica_id,
            "ts": finished_at,
        },
    ]


def completion_event(latency_sec: float) -> dict[str, object]:
    return {
        "workflow_name": "qmsum_lane1",
        "event_type": "session_completed",
        "session_id": SESSION_ID,
        "payload": {"latency_sec": latency_sec},
    }


def trace_events() -> list[dict[str, object]]:
    return [
        completion_event(30.0),
        *eviction_events("r4a", 4.2, 4.5),
        *load_events("r4b", 4.6, 5.8, 1.0),
        *eviction_events("r8a", 11.2, 11.8),
        *load_events("r8b", 12.0, 14.0, 1.8),
        *agent_events("lane_a_0", 0.0, 1.0, -5.0, -0.1, "r4a", "4b"),
        *agent_events("lane_a_1", 3.0, 4.0, 1.4, 2.8, "r4a", "4b"),
        *agent_events("lane_a_2", 6.0, 7.0, 4.2, 5.8, "r4b", "4b"),
        *agent_events("lane_b_0", 0.0, 1.0, -4.0, -0.1, "r4c", "4b"),
        *agent_events("lane_b_1", 2.0, 3.0, 1.4, 1.8, "r4c", "4b"),
        *agent_events("lane_b_2", 5.0, 6.0, 4.4, 4.8, "r4c", "4b"),
        *agent_events("merge", 9.0, 11.0, 7.0, 8.8, "r8a", "8b"),
        *agent_events("expand", 15.0, 16.0, 11.2, 14.8, "r8b", "8b"),
        *agent_events("finalize", 20.0, 21.0, 16.2, 19.8, "r8b", "8b"),
    ]


def test_extract_session_costs_recomputes_parallel_join() -> None:
    costs = extract_session_costs(trace_events(), "fifo_base_r1")

    assert len(costs) == 1
    assert costs[0].latency_sec == pytest.approx(30.0)
    assert costs[0].intermediate_acquire_sec == pytest.approx(9.0)
    assert costs[0].reload_seconds == pytest.approx(2.8)


def test_extract_session_costs_requires_acquire_evidence() -> None:
    events = [
        event
        for event in trace_events()
        if not (
            event.get("event_type") == "acquire_granted"
            and event.get("node_id") == "lane_a_2"
        )
    ]

    with pytest.raises(ValueError, match="acquire event identities are unmatched"):
        extract_session_costs(events, "fifo_base_r1")


def test_extract_session_costs_requires_reload_evidence() -> None:
    events = [
        event
        for event in trace_events()
        if not (
            event.get("event_type") == "model_load_finished"
            and event.get("replica_id") == "r4b"
        )
    ]

    with pytest.raises(ValueError, match="lacks complete ready-load evidence"):
        extract_session_costs(events, "fifo_base_r1")


def test_extract_session_costs_requires_eviction_evidence() -> None:
    events = [
        event
        for event in trace_events()
        if not (
            event.get("event_type") == "model_evicted"
            and event.get("replica_id") == "r4a"
        )
    ]

    with pytest.raises(ValueError, match="lacks complete ready-load evidence"):
        extract_session_costs(events, "fifo_base_r1")


def test_latency_quantile_session_ranks_by_e2e_latency() -> None:
    costs = [
        SessionFusionCost("r1", "s1", 10.0, 9.0, 0.0),
        SessionFusionCost("r1", "s2", 20.0, 1.0, 0.0),
        SessionFusionCost("r1", "s3", 30.0, 2.0, 0.0),
    ]

    assert latency_quantile_session(costs, 0.50).session_id == "s2"
    assert latency_quantile_session(costs, 0.95).session_id == "s3"


def test_nearest_rank_matches_trace_summary_convention() -> None:
    assert nearest_rank(list(range(1, 21)), 0.50) == 10
    assert nearest_rank(list(range(1, 21)), 0.95) == 19


def test_motivation_analysis_rejects_fusion_runs(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="cannot read a fusion run"):
        load_baseline_run(tmp_path, "fifo_fuse_r1")
