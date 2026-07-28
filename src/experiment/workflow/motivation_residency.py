"""Trace analysis for the model-residency motivation experiment.

Answers four falsifiable questions about a completed serve run, using only the trace:

1. Does concurrent model demand exceed the GPU pool, and by how much over time?
   Plus ``B(N)``: the loads an optimal clairvoyant replacement policy would still pay
   at pool size ``N``, over the grant order this run actually produced.
2. What share of the run does model loading cost, and where does acquire wait go?
3. (Cross-arm; the caller compares group 2 across policies.)
4. Was each load foreseeable, and was there idle capacity while it was foreseeable?

Two deliberate approximations, both stated so they can be checked:

- A model's eligible accelerators are taken to be those it was *observed* on during the
  run. Conservative: a card never used for a key is never counted as available for it.
- A card counts as available for a load whenever no generation is in flight on it, which
  ignores the cost of evicting whatever sits there. Group 4 is therefore an upper bound
  on preloading opportunity, not an achievable schedule.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from experiment.workflow.analysis import (
    _execution_interval,
    _interval_intersection_duration,
    _merge_intervals,
    read_jsonl,
)

Event = Mapping[str, Any]
Interval = tuple[float, float]


@dataclass(frozen=True, slots=True)
class LoadEvent:
    replica_id: str
    model_key: str
    accelerator_id: str
    started_at: float
    finished_at: float
    reason: str

    @property
    def duration_sec(self) -> float:
        return self.finished_at - self.started_at


@dataclass(frozen=True, slots=True)
class Demand:
    """One task's window of needing a model: request until generation ends."""

    model_key: str
    gpu_kind: str
    replica_id: str
    session_id: str
    node_id: str
    workflow_name: str
    requested_at: float
    granted_at: float
    released_at: float


def analyze_residency(
    trace_path: Path,
    *,
    pool_sizes: Sequence[int] = (1, 2, 3, 4, 5, 6),
) -> dict[str, Any]:
    events = read_jsonl(trace_path)
    if not events:
        raise ValueError("workflow trace is empty")
    run_started = _single_ts(events, "run_started")
    run_finished = _single_ts(events, "run_finished")
    demands = _demands(events)
    if not demands:
        raise ValueError("trace contains no granted acquires")
    loads = _loads(events)
    graph = _graph_from_trace(events)
    key_accelerators = _key_accelerators(demands, loads)
    busy_by_accelerator = _busy_by_accelerator(events)

    return {
        "run_duration_sec": run_finished - run_started,
        "demand": _demand_metrics(demands, pool_sizes=pool_sizes),
        "belady": _belady_metrics(demands, pool_sizes=pool_sizes),
        "loading": _loading_metrics(loads, demands, run_started, run_finished),
        "wait_breakdown": _wait_breakdown(demands, loads),
        "preload": _preload_metrics(
            loads,
            demands,
            events,
            graph=graph,
            key_accelerators=key_accelerators,
            busy_by_accelerator=busy_by_accelerator,
        ),
    }


def _demands(events: Sequence[Event]) -> list[Demand]:
    requested: dict[str, float] = {}
    task_of_acquire: dict[str, str] = {}
    granted: dict[str, Event] = {}
    released: dict[str, float] = {}
    for event in events:
        event_type = event.get("event_type")
        if event_type == "acquire_requested":
            requested[_text(event, "acquire_id")] = _number(event["ts"])
            task_of_acquire[_text(event, "acquire_id")] = _text(event, "task_id")
        elif event_type == "acquire_granted":
            granted[_text(event, "acquire_id")] = event
        elif event_type == "task_execution_finished":
            released[_text(event, "task_id")] = _number(event["ts"])

    demands: list[Demand] = []
    for acquire_id, event in granted.items():
        task_id = task_of_acquire.get(acquire_id)
        request_ts = requested.get(acquire_id)
        if task_id is None or request_ts is None:
            raise ValueError(f"acquire_granted has no matching request: {acquire_id}")
        release_ts = released.get(task_id)
        if release_ts is None:
            # A task granted but never finished cannot bound its own demand window.
            continue
        demands.append(
            Demand(
                model_key=_text(event, "model_key"),
                gpu_kind=_text(event, "gpu_kind"),
                replica_id=_text(event, "replica_id"),
                session_id=_text(event, "session_id"),
                node_id=_text(event, "node_id"),
                workflow_name=_text(event, "workflow_name"),
                requested_at=request_ts,
                granted_at=_number(event["ts"]),
                released_at=release_ts,
            )
        )
    demands.sort(key=lambda demand: demand.granted_at)
    return demands


def _loads(events: Sequence[Event]) -> list[LoadEvent]:
    started: dict[str, Event] = {}
    loads: list[LoadEvent] = []
    for event in events:
        event_type = event.get("event_type")
        if event_type == "model_load_started":
            started[_text(event, "replica_id")] = event
        elif event_type == "model_load_finished":
            replica_id = _text(event, "replica_id")
            start_event = started.pop(replica_id, None)
            if start_event is None:
                raise ValueError(f"load finished without a start: {replica_id}")
            loads.append(
                LoadEvent(
                    replica_id=replica_id,
                    model_key=_text(start_event, "model_key"),
                    accelerator_id=_text(start_event, "accelerator_id"),
                    started_at=_number(start_event["ts"]),
                    finished_at=_number(event["ts"]),
                    reason=str(_payload(start_event).get("reason", "")),
                )
            )
    loads.sort(key=lambda load: load.started_at)
    return loads


def _demand_metrics(
    demands: Sequence[Demand], *, pool_sizes: Sequence[int]
) -> dict[str, Any]:
    overall = _concurrency_profile(
        [
            (demand.requested_at, demand.released_at, demand.model_key)
            for demand in demands
        ]
    )
    gpu_kinds = {demand.gpu_kind for demand in demands}
    by_gpu_kind = {
        gpu_kind: _concurrency_profile(
            [
                (demand.requested_at, demand.released_at, demand.model_key)
                for demand in demands
                if demand.gpu_kind == gpu_kind
            ]
        )
        for gpu_kind in sorted(gpu_kinds)
    }
    total = overall["observed_sec"]
    return {
        "distinct_model_keys": len({demand.model_key for demand in demands}),
        "min_static_cards": overall["max_concurrent"],
        "mean_concurrent_models": overall["mean_concurrent"],
        "seconds_at_concurrency": overall["seconds_at"],
        "fraction_over_pool_size": {
            size: (
                sum(
                    seconds
                    for level, seconds in overall["seconds_at"].items()
                    if level > size
                )
                / total
                if total
                else 0.0
            )
            for size in pool_sizes
        },
        "by_gpu_kind": {
            gpu_kind: {
                "min_static_cards": profile["max_concurrent"],
                "mean_concurrent_models": profile["mean_concurrent"],
            }
            for gpu_kind, profile in by_gpu_kind.items()
        },
    }


def _concurrency_profile(
    windows: Sequence[tuple[float, float, str]],
) -> dict[str, Any]:
    """Time spent at each distinct-model-demand level, by sweeping window endpoints."""
    transitions: list[tuple[float, int, str]] = []
    for start, end, key in windows:
        if end <= start:
            continue
        transitions.append((start, 1, key))
        transitions.append((end, -1, key))
    transitions.sort(key=lambda item: (item[0], item[1]))

    seconds_at: dict[int, float] = defaultdict(float)
    active: dict[str, int] = defaultdict(int)
    distinct = 0
    cursor = transitions[0][0] if transitions else 0.0
    for time, delta, key in transitions:
        if time > cursor:
            seconds_at[distinct] += time - cursor
            cursor = time
        active[key] += delta
        if delta == 1 and active[key] == 1:
            distinct += 1
        elif delta == -1 and active[key] == 0:
            distinct -= 1
    observed = sum(seconds_at.values())
    mean = (
        sum(level * seconds for level, seconds in seconds_at.items()) / observed
        if observed
        else 0.0
    )
    return {
        "seconds_at": dict(sorted(seconds_at.items())),
        "max_concurrent": max(seconds_at, default=0),
        "mean_concurrent": mean,
        "observed_sec": observed,
    }


def _belady_metrics(
    demands: Sequence[Demand], *, pool_sizes: Sequence[int]
) -> dict[str, Any]:
    accesses = [demand.model_key for demand in demands]
    return {
        "access_count": len(accesses),
        "distinct_model_keys": len(set(accesses)),
        "note": "lower bound conditional on this run's grant order",
        "loads_by_pool_size": {
            size: belady_misses(accesses, size) for size in pool_sizes
        },
    }


def belady_misses(accesses: Sequence[str], capacity: int) -> int:
    """Cache misses under optimal (furthest-in-future) replacement.

    No online policy can beat this on the same access sequence, so it lower-bounds the
    loads any residency manager must pay at ``capacity`` exclusive cards.
    """
    if capacity < 1:
        raise ValueError("capacity must be positive")
    next_use: list[int] = [len(accesses)] * len(accesses)
    seen: dict[str, int] = {}
    for index in range(len(accesses) - 1, -1, -1):
        key = accesses[index]
        next_use[index] = seen.get(key, len(accesses))
        seen[key] = index

    cached: dict[str, int] = {}
    misses = 0
    for index, key in enumerate(accesses):
        if key in cached:
            cached[key] = next_use[index]
            continue
        misses += 1
        if len(cached) >= capacity:
            victim = max(cached, key=lambda candidate: cached[candidate])
            del cached[victim]
        cached[key] = next_use[index]
    return misses


def _loading_metrics(
    loads: Sequence[LoadEvent],
    demands: Sequence[Demand],
    run_started: float,
    run_finished: float,
) -> dict[str, Any]:
    loading_sec = sum(load.duration_sec for load in loads)
    generation_sec = sum(demand.released_at - demand.granted_at for demand in demands)
    run_sec = run_finished - run_started
    return {
        "model_load_count": len(loads),
        "loading_gpu_seconds": loading_sec,
        "generation_gpu_seconds": generation_sec,
        "loading_over_generation": (
            loading_sec / generation_sec if generation_sec else 0.0
        ),
        "loading_over_run_duration": loading_sec / run_sec if run_sec else 0.0,
        "load_duration_sec": _distribution([load.duration_sec for load in loads]),
        "loads_by_model_key": _counts(load.model_key for load in loads),
        "loads_by_reason": _counts(load.reason for load in loads),
    }


def _wait_breakdown(
    demands: Sequence[Demand], loads: Sequence[LoadEvent]
) -> dict[str, Any]:
    """Split acquire wait into time behind this key's load, another key's, or neither.

    Head-of-line time is the residual: overlap with any load minus overlap with this
    key's own loads, so a wait blocked by an unrelated model is separated from a wait
    for the model the request actually needs.
    """
    loads_by_key: dict[str, list[Interval]] = defaultdict(list)
    for load in loads:
        loads_by_key[load.model_key].append((load.started_at, load.finished_at))
    all_loads = [(load.started_at, load.finished_at) for load in loads]

    own = other_key = 0.0
    total = 0.0
    for demand in demands:
        window = [(demand.requested_at, demand.granted_at)]
        total += demand.granted_at - demand.requested_at
        own_overlap = _overlap(window, loads_by_key.get(demand.model_key, []))
        any_overlap = _overlap(window, all_loads)
        own += own_overlap
        other_key += max(0.0, any_overlap - own_overlap)
    return {
        "total_sec": total,
        "own_model_load_sec": own,
        "head_of_line_other_key_sec": other_key,
        "other_sec": max(0.0, total - own - other_key),
        "own_model_load_fraction": own / total if total else 0.0,
        "head_of_line_other_key_fraction": other_key / total if total else 0.0,
    }


def _preload_metrics(
    loads: Sequence[LoadEvent],
    demands: Sequence[Demand],
    events: Sequence[Event],
    *,
    graph: Mapping[str, Mapping[str, tuple[str, ...]]],
    key_accelerators: Mapping[str, frozenset[str]],
    busy_by_accelerator: Mapping[str, list[Interval]],
) -> dict[str, Any]:
    task_started = _task_started(events)
    session_submitted = _session_submitted(events)
    # A load is charged to the first activation it unblocked on that replica.
    first_demand: dict[str, Demand] = {}
    for demand in demands:
        first_demand.setdefault(demand.replica_id, demand)

    records: list[dict[str, Any]] = []
    for load in loads:
        demand = first_demand.get(load.replica_id)
        if demand is None:
            continue
        known_at = _known_at(
            demand, graph=graph, task_started=task_started, submitted=session_submitted
        )
        if known_at is None:
            continue
        lead = load.started_at - known_at
        duration = load.duration_sec
        capacity_sec = _available_capacity(
            known_at,
            load.started_at,
            duration,
            accelerators=key_accelerators.get(load.model_key, frozenset()),
            busy_by_accelerator=busy_by_accelerator,
        )
        records.append(
            {
                "model_key": load.model_key,
                "node_id": demand.node_id,
                "workflow_name": demand.workflow_name,
                "load_duration_sec": duration,
                "information_lead_sec": lead,
                "lead_sufficient": lead >= duration,
                "capacity_sufficient": capacity_sec >= duration,
                "hideable_sec": min(max(lead, 0.0), duration)
                if capacity_sec >= duration
                else 0.0,
            }
        )

    attributed = len(records)
    lead_ok = sum(1 for record in records if record["lead_sufficient"])
    both_ok = sum(
        1
        for record in records
        if record["lead_sufficient"] and record["capacity_sufficient"]
    )
    total_load_sec = sum(record["load_duration_sec"] for record in records)
    return {
        "load_count": len(loads),
        "attributed_load_count": attributed,
        "lead_sufficient_count": lead_ok,
        "lead_and_capacity_count": both_ok,
        "lead_sufficient_fraction": lead_ok / attributed if attributed else 0.0,
        "lead_and_capacity_fraction": both_ok / attributed if attributed else 0.0,
        "information_lead_sec": _distribution(
            [record["information_lead_sec"] for record in records]
        ),
        "load_duration_sec": _distribution(
            [record["load_duration_sec"] for record in records]
        ),
        "attributed_load_sec": total_load_sec,
        "hideable_load_sec": sum(record["hideable_sec"] for record in records),
        "hideable_load_fraction": (
            sum(record["hideable_sec"] for record in records) / total_load_sec
            if total_load_sec
            else 0.0
        ),
        "records": records,
    }


def _known_at(
    demand: Demand,
    *,
    graph: Mapping[str, Mapping[str, tuple[str, ...]]],
    task_started: Mapping[tuple[str, str], float],
    submitted: Mapping[str, float],
) -> float | None:
    """When this activation's model became determined: first predecessor start."""
    predecessors = graph.get(demand.workflow_name, {}).get(demand.node_id, ())
    if not predecessors:
        return submitted.get(demand.session_id)
    starts = [
        task_started[(demand.session_id, predecessor)]
        for predecessor in predecessors
        if (demand.session_id, predecessor) in task_started
    ]
    return min(starts) if starts else submitted.get(demand.session_id)


def _available_capacity(
    window_start: float,
    window_end: float,
    required_sec: float,
    *,
    accelerators: Iterable[str],
    busy_by_accelerator: Mapping[str, list[Interval]],
) -> float:
    """Longest idle stretch on any eligible card inside the window."""
    if window_end <= window_start:
        return 0.0
    best = 0.0
    for accelerator_id in accelerators:
        free = _complement(
            busy_by_accelerator.get(accelerator_id, []), window_start, window_end
        )
        longest = max((end - start for start, end in free), default=0.0)
        best = max(best, longest)
        if best >= required_sec:
            break
    return best


def _complement(
    busy: Sequence[Interval], window_start: float, window_end: float
) -> list[Interval]:
    free: list[Interval] = []
    cursor = window_start
    for start, end in _merge_intervals(busy):
        if end <= window_start or start >= window_end:
            continue
        start = max(start, window_start)
        if start > cursor:
            free.append((cursor, start))
        cursor = max(cursor, min(end, window_end))
    if cursor < window_end:
        free.append((cursor, window_end))
    return free


def _graph_from_trace(
    events: Sequence[Event],
) -> dict[str, dict[str, tuple[str, ...]]]:
    """Reconstruct per-workflow predecessor sets from the items that flowed on edges."""
    edges: dict[str, set[tuple[str, str]]] = defaultdict(set)
    for event in events:
        if event.get("event_type") != "item_enqueued":
            continue
        source = event.get("source_node")
        target = event.get("target_node")
        workflow_name = event.get("workflow_name")
        if (
            isinstance(source, str)
            and isinstance(target, str)
            and isinstance(workflow_name, str)
        ):
            edges[workflow_name].add((source, target))
    graph: dict[str, dict[str, tuple[str, ...]]] = {}
    for workflow_name, pairs in edges.items():
        predecessors: dict[str, set[str]] = defaultdict(set)
        for source, target in pairs:
            predecessors[target].add(source)
        graph[workflow_name] = {
            node: tuple(sorted(sources)) for node, sources in predecessors.items()
        }
    return graph


def _key_accelerators(
    demands: Sequence[Demand], loads: Sequence[LoadEvent]
) -> dict[str, frozenset[str]]:
    observed: dict[str, set[str]] = defaultdict(set)
    for load in loads:
        observed[load.model_key].add(load.accelerator_id)
    return {key: frozenset(values) for key, values in observed.items()}


def _busy_by_accelerator(events: Sequence[Event]) -> dict[str, list[Interval]]:
    accelerator_of: dict[str, str] = {}
    for event in events:
        replica_id = event.get("replica_id")
        accelerator_id = event.get("accelerator_id")
        if isinstance(replica_id, str) and isinstance(accelerator_id, str):
            accelerator_of[replica_id] = accelerator_id
    busy: dict[str, list[Interval]] = defaultdict(list)
    for event in events:
        if event.get("event_type") != "task_execution_finished":
            continue
        replica_id = event.get("replica_id")
        if not isinstance(replica_id, str):
            continue
        accelerator_id = accelerator_of.get(replica_id)
        if accelerator_id is None:
            continue
        busy[accelerator_id].append(_execution_interval(event))
    return busy


def _task_started(events: Sequence[Event]) -> dict[tuple[str, str], float]:
    started: dict[tuple[str, str], float] = {}
    for event in events:
        if event.get("event_type") != "task_started":
            continue
        session_id = event.get("session_id")
        node_id = event.get("node_id")
        if isinstance(session_id, str) and isinstance(node_id, str):
            key = (session_id, node_id)
            started[key] = min(started.get(key, math.inf), _number(event["ts"]))
    return started


def _session_submitted(events: Sequence[Event]) -> dict[str, float]:
    return {
        _text(event, "session_id"): _number(event["ts"])
        for event in events
        if event.get("event_type") == "session_submitted"
    }


def _overlap(left: Sequence[Interval], right: Sequence[Interval]) -> float:
    if not left or not right:
        return 0.0
    return _interval_intersection_duration(left, right)


def _distribution(values: Sequence[float]) -> dict[str, float]:
    if not values:
        return {"count": 0, "mean": 0.0, "p50": 0.0, "p95": 0.0, "max": 0.0, "sum": 0.0}
    ordered = sorted(values)
    return {
        "count": len(ordered),
        "mean": sum(ordered) / len(ordered),
        "p50": ordered[len(ordered) // 2],
        "p95": ordered[min(len(ordered) - 1, int(len(ordered) * 0.95))],
        "max": ordered[-1],
        "sum": sum(ordered),
    }


def _counts(values: Iterable[str]) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for value in values:
        counts[value] += 1
    return dict(sorted(counts.items()))


def _single_ts(events: Sequence[Event], event_type: str) -> float:
    matches = [event for event in events if event.get("event_type") == event_type]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one {event_type} event")
    return _number(matches[0]["ts"])


def _payload(event: Event) -> Mapping[str, Any]:
    payload = event.get("payload")
    if payload is None:
        return {}
    if not isinstance(payload, Mapping):
        raise TypeError("trace payload must be a mapping")
    return payload


def _number(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError(f"expected a number: {value!r}")
    return float(value)


def _text(event: Event, field: str) -> str:
    value = event.get(field)
    if not isinstance(value, str):
        raise TypeError(f"trace field {field} must be text: {value!r}")
    return value


__all__ = ["Demand", "LoadEvent", "analyze_residency", "belady_misses"]
