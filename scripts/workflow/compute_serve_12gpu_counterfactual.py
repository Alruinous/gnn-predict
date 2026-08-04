"""Compute the trace-derived 12-GPU SagePilot evaluation counterfactual."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import shutil
import statistics
import sys
from collections import defaultdict
from collections.abc import Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, cast

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import plot_serve_0728_counterfactual_drafts as replay  # noqa: E402

from experiment.workflow.artifacts import file_sha256  # noqa: E402
from experiment.workflow.cache_replay import (  # noqa: E402
    AgentDurationTable,
    ReplayMetrics,
    ReplayWorkload,
    TaskSpec,
)
from workflow.artifacts import ResourceContractCache, SchedulerConfig  # noqa: E402
from workflow.replica import ModelDeploymentConfig  # noqa: E402
from workflow.schema import AgentNodeConfig  # noqa: E402
from workflow.types import ModelReplicaState  # noqa: E402

CALCULATION = "homothetic_12gpu_shared_pool_counterfactual"
RAW_CALCULATION = "trace_replay_12gpu_shared_pool"
ORACLE_CALCULATION = "homothetic_trace_offline_reference"
SOURCE_HARDWARE = "a1v3"
TARGET_HARDWARE = "a3v9"
SOURCE_GPU_SLOTS = {"a100": 1, "v100": 3}
GPU_SLOTS = {"a100": 3, "v100": 9}
RESOURCE_SCALE = 3
SOURCE_REPLICA_CAP = 2
REPLICA_CAP = SOURCE_REPLICA_CAP
SESSION_MULTIPLIER = 3
SESSIONS_PER_WORKFLOW = 60
TOTAL_SESSIONS = 180
ARRIVALS = ("burst", "poisson_r050", "poisson_r025")
ARRIVAL_RATES = {"poisson_r050": 0.150, "poisson_r025": 0.075}
REPEATS = ("r1", "r2", "r3")
SYSTEM_ARMS = ("parrot", "kairos", "sagepilot")
ABLATION_ARMS = (
    "sagepilot",
    "analytical",
    "gbdt",
    "nofuse",
    "noprefetch",
    "noxwf",
)
METHOD_LABELS = {
    "parrot": "Parrot",
    "kairos": "Kairos",
    "sagepilot": "SagePilot",
    "oracle": "Oracle",
    "analytical": "Formula",
    "gbdt": "GBDT",
    "nofuse": "NoFusion",
    "noprefetch": "NoPrefetch",
    "noxwf": "NoXWF",
}
GPU_STATES = (
    "generation_gpu_sec",
    "idle_resident_gpu_sec",
    "loading_gpu_sec",
    "evicting_gpu_sec",
)
DEFAULT_OUTPUT_DIR = (
    ROOT / "paper" / "hpca2027-sagepilot" / "data" / "evaluation_12gpu_counterfactual"
)
PREDICTOR_SOURCE = (
    ROOT
    / "paper"
    / "hpca2027-sagepilot"
    / "archive"
    / "evaluation_20260728"
    / "data"
    / "evaluation"
    / "evaluation_metrics.json"
)
SOURCE_REPLAY_METRICS = (
    ROOT / "output" / "serve_0728_evaluation_drafts" / "analysis_metrics.json"
)
TARGET_PAPER_METRICS = (
    ROOT
    / "paper"
    / "hpca2027-sagepilot"
    / "data"
    / "evaluation"
    / "system_performance_metrics.json"
)
TARGET_WORKFLOW_SESSION_COUNT = 60


@dataclass(frozen=True, slots=True)
class GenerationRecord:
    workflow_name: str
    model_name: str
    duration_sec: float


class CounterfactualSimulator(replay.DetailedReplaySimulator):
    def __init__(
        self,
        workload: ReplayWorkload,
        *,
        policy: replay.PolicyName,
        cache: ResourceContractCache,
        scheduler_config: SchedulerConfig,
    ) -> None:
        super().__init__(
            workload,
            policy=policy,
            cache=cache,
            scheduler_config=scheduler_config,
        )
        self.generation_records: list[GenerationRecord] = []
        self.drain_reactivation_count = 0

    def run(self) -> ReplayMetrics:
        while True:
            try:
                return super().run()
            except RuntimeError as error:
                if "replay deadlocked" not in str(error):
                    raise
                reactivated = self._reactivate_drained_demand()
                if reactivated == 0:
                    raise
                self.drain_reactivation_count += reactivated
                super()._schedule_core()

    def _reactivate_drained_demand(self) -> int:
        demanded_model_keys = set()
        for pending in self.core.pending_acquires.values():
            task = self.core.tasks[pending.task_id]
            workflow_name = self.core.sessions[task.session_id].workflow_name
            node = self.workload.workflows[workflow_name].node_map()[task.node_id]
            if not isinstance(node, AgentNodeConfig):
                raise TypeError("pending acquire belongs to a function node")
            demanded_model_keys.add(ModelDeploymentConfig.from_node(node).model_key)
        reactivated = 0
        for model_key in sorted(demanded_model_keys):
            candidates = [
                replica
                for replica in self.core.replicas.values()
                if replica.model_key == model_key
                and replica.state == ModelReplicaState.IDLE
                and replica.drain_requested_at is not None
            ]
            if not candidates:
                continue
            victim = min(candidates, key=lambda replica: replica.replica_id)
            victim.drain_requested_at = None
            reactivated += 1
        return reactivated

    def _finish_agent(self, acquire_id: str) -> None:
        active = self.active_agents[acquire_id]
        model_name = self.replica_timelines[active.grant.replica_id].model_name
        self.generation_records.append(
            GenerationRecord(
                workflow_name=active.workflow_name,
                model_name=model_name,
                duration_sec=self.now - active.started_at,
            )
        )
        super()._finish_agent(acquire_id)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--replay-workers", type=int, default=1)
    return parser.parse_args(argv)


def _clone_session_id(workflow_name: str, index: int) -> str:
    return f"{workflow_name}-{index:04d}"


def expand_workload(base: ReplayWorkload, arrival: str) -> ReplayWorkload:
    if arrival not in ARRIVALS:
        raise ValueError(f"unsupported arrival: {arrival}")
    source_sessions: dict[str, list[str]] = defaultdict(list)
    for session_id, workflow_name in base.session_workflows.items():
        source_sessions[workflow_name].append(session_id)
    if set(source_sessions) != set(replay.WORKFLOWS):
        raise ValueError(f"unexpected workflows: {sorted(source_sessions)}")
    for workflow_name, session_ids in source_sessions.items():
        session_ids.sort()
        if len(session_ids) * SESSION_MULTIPLIER != SESSIONS_PER_WORKFLOW:
            source_count = SESSIONS_PER_WORKFLOW // SESSION_MULTIPLIER
            raise ValueError(
                f"{workflow_name} must contain {source_count} source sessions"
            )

    session_arrivals: dict[str, float] = {}
    session_workflows: dict[str, str] = {}
    task_specs: dict[tuple[str, str], TaskSpec] = {}
    exact_durations: dict[tuple[str, str, int], float] = {}
    task_any_durations: dict[tuple[str, str], float] = {}
    for workflow_name in replay.WORKFLOWS:
        source_ids = source_sessions[workflow_name]
        workflow = base.workflows[workflow_name]
        for target_index in range(SESSIONS_PER_WORKFLOW):
            source_id = source_ids[target_index % len(source_ids)]
            target_id = _clone_session_id(workflow_name, target_index)
            session_arrivals[target_id] = base.session_arrivals[source_id]
            session_workflows[target_id] = workflow_name
            for node_id in workflow.graph.topological_order:
                source_key = (source_id, node_id)
                target_key = (target_id, node_id)
                spec = base.task_specs[source_key]
                task_specs[target_key] = replace(spec, session_id=target_id)
                any_duration = base.agent_durations.task_any_batch.get(source_key)
                if any_duration is not None:
                    task_any_durations[target_key] = any_duration
                for batch_size in range(1, 4):
                    exact = base.agent_durations.exact.get(
                        (source_id, node_id, batch_size)
                    )
                    if exact is not None:
                        exact_durations[(target_id, node_id, batch_size)] = exact

    if len(session_arrivals) != TOTAL_SESSIONS:
        raise ValueError(f"expanded workload has {len(session_arrivals)} sessions")
    return replace(
        base,
        name=f"{base.name}:{TARGET_HARDWARE}:x{SESSION_MULTIPLIER}",
        session_arrivals=session_arrivals,
        session_workflows=session_workflows,
        task_specs=task_specs,
        agent_durations=AgentDurationTable(
            exact=exact_durations,
            node_batch=base.agent_durations.node_batch,
            model_batch=base.agent_durations.model_batch,
            task_any_batch=task_any_durations,
        ),
        gpu_slots=GPU_SLOTS,
    )


def shared_model_groups(workload: ReplayWorkload) -> dict[str, dict[str, object]]:
    groups: dict[str, dict[str, object]] = {}
    for workflow_name, workflow in workload.workflows.items():
        for node in workflow.node_map().values():
            if not isinstance(node, AgentNodeConfig):
                continue
            deployment = ModelDeploymentConfig.from_node(node)
            group = groups.setdefault(
                deployment.model_name,
                {"model_key": deployment.model_key, "workflows": set()},
            )
            if group["model_key"] != deployment.model_key:
                raise ValueError(
                    f"model has multiple replica groups: {deployment.model_name}"
                )
            cast(set[str], group["workflows"]).add(workflow_name)
    if set(groups) != set(replay.MODELS):
        raise ValueError(f"shared model set mismatch: {sorted(groups)}")
    return {
        model_name: {
            "model_key": groups[model_name]["model_key"],
            "workflows": sorted(cast(set[str], groups[model_name]["workflows"])),
        }
        for model_name in replay.MODELS
    }


def scaled_scheduler_config(
    manifest: Mapping[str, object], workload: ReplayWorkload
) -> SchedulerConfig:
    config = replay.scheduler_config(manifest, workload)
    if config.elastic_replicas and config.max_replicas_per_model != SOURCE_REPLICA_CAP:
        source_cap = config.max_replicas_per_model
        raise ValueError(f"source elastic replica cap must be 2, got {source_cap}")
    return config


def _model_lifecycle(
    simulator: CounterfactualSimulator, business_finished: float
) -> dict[str, dict[str, float]]:
    totals = {
        model_name: {
            "generation_gpu_sec": 0.0,
            "idle_resident_gpu_sec": 0.0,
            "loading_gpu_sec": 0.0,
            "evicting_gpu_sec": 0.0,
        }
        for model_name in replay.MODELS
    }
    for timeline in simulator.replica_timelines.values():
        values = totals[timeline.model_name]
        load_end = min(timeline.load_finished or business_finished, business_finished)
        values["loading_gpu_sec"] += max(
            0.0, load_end - min(timeline.load_started, business_finished)
        )
        if timeline.load_finished is None or timeline.load_finished > business_finished:
            continue
        resident_end = min(
            timeline.eviction_started or business_finished,
            business_finished,
        )
        intervals = [
            (max(start, timeline.load_finished), min(end, resident_end))
            for start, end in timeline.executions
            if end > timeline.load_finished and start < resident_end
        ]
        active = replay.interval_union_duration(intervals)
        resident = max(0.0, resident_end - timeline.load_finished)
        values["generation_gpu_sec"] += active
        values["idle_resident_gpu_sec"] += max(0.0, resident - active)
        if timeline.eviction_started is not None:
            eviction_end = min(
                timeline.eviction_finished or business_finished,
                business_finished,
            )
            values["evicting_gpu_sec"] += max(
                0.0,
                eviction_end - min(timeline.eviction_started, business_finished),
            )
    return totals


def _workflow_gpu_time(
    simulator: CounterfactualSimulator,
    lifecycle: Mapping[str, Mapping[str, float]],
) -> dict[str, dict[str, float]]:
    weights: dict[tuple[str, str], float] = defaultdict(float)
    for record in simulator.generation_records:
        weights[(record.workflow_name, record.model_name)] += record.duration_sec
    allocated = {
        workflow_name: {state: 0.0 for state in GPU_STATES}
        for workflow_name in replay.WORKFLOWS
    }
    for model_name in replay.MODELS:
        denominator = sum(
            weights[(workflow_name, model_name)] for workflow_name in replay.WORKFLOWS
        )
        if denominator <= 0.0:
            raise ValueError(f"model has no generation attribution: {model_name}")
        for workflow_name in replay.WORKFLOWS:
            share = weights[(workflow_name, model_name)] / denominator
            for state in GPU_STATES:
                allocated[workflow_name][state] += lifecycle[model_name][state] * share
    return allocated


def _peak_overlap(intervals: Sequence[tuple[float, float]]) -> int:
    events = [
        event
        for start, end in intervals
        if end > start
        for event in ((start, 1), (end, -1))
    ]
    active = 0
    peak = 0
    for _, delta in sorted(events, key=lambda event: (event[0], event[1])):
        active += delta
        peak = max(peak, active)
    return peak


def _replica_group_metrics(
    simulator: CounterfactualSimulator,
    business_finished: float,
) -> dict[str, dict[str, object]]:
    workflows: dict[str, set[str]] = defaultdict(set)
    for record in simulator.generation_records:
        workflows[record.model_name].add(record.workflow_name)
    result = {}
    for model_name in replay.MODELS:
        timelines = [
            timeline
            for timeline in simulator.replica_timelines.values()
            if timeline.model_name == model_name
        ]
        if not timelines:
            raise ValueError(f"model has no replica timeline: {model_name}")
        model_keys = {timeline.model_key for timeline in timelines}
        gpu_kinds = {timeline.gpu_kind for timeline in timelines}
        if len(model_keys) != 1 or len(gpu_kinds) != 1:
            raise ValueError(f"model replica group is inconsistent: {model_name}")
        occupied = [
            (
                timeline.load_started,
                min(timeline.eviction_finished or business_finished, business_finished),
            )
            for timeline in timelines
        ]
        resident: list[tuple[float, float]] = []
        for timeline in timelines:
            if (
                timeline.load_finished is None
                or timeline.load_finished >= business_finished
            ):
                continue
            resident.append(
                (
                    timeline.load_finished,
                    min(
                        timeline.eviction_started or business_finished,
                        business_finished,
                    ),
                )
            )
        gpu_kind = next(iter(gpu_kinds))
        peak_occupied = _peak_overlap(occupied)
        if peak_occupied > GPU_SLOTS[gpu_kind]:
            raise ValueError(f"model exceeds the {gpu_kind} pool: {model_name}")
        result[model_name] = {
            "model_key": next(iter(model_keys)),
            "gpu_kind": gpu_kind,
            "served_workflows": sorted(workflows[model_name]),
            "shared_across_workflows": len(workflows[model_name]) > 1,
            "replica_load_count": len(timelines),
            "peak_occupied_replicas": peak_occupied,
            "peak_resident_replicas": _peak_overlap(resident),
        }
    return result


def replay_one(
    arrival: str,
    arm: str,
    repeat: str,
    workload: ReplayWorkload,
    cache_store: dict[tuple[Path, str], ResourceContractCache],
) -> dict[str, object]:
    directory = replay.run_dir(SOURCE_HARDWARE, arrival, arm, repeat)
    manifest = replay.read_manifest(directory / "run_manifest.json")
    prediction = manifest["predictions"]
    if not isinstance(prediction, dict) or not isinstance(prediction.get("path"), str):
        raise TypeError("predictions.path is invalid")
    cache = replay.normalized_cache(
        ROOT / prediction["path"], SOURCE_HARDWARE, cache_store
    )
    simulator = CounterfactualSimulator(
        workload,
        policy=replay.simulator_policy(arm),
        cache=cache,
        scheduler_config=scaled_scheduler_config(manifest, workload),
    )
    metrics = simulator.run()
    if len(simulator.session_completions) != TOTAL_SESSIONS:
        completed = len(simulator.session_completions)
        raise ValueError(f"{arrival}/{arm}/{repeat} completed {completed} sessions")
    workflow_counts: dict[str, int] = defaultdict(int)
    latencies = []
    for session_id, completion in simulator.session_completions.items():
        workflow_name = workload.session_workflows[session_id]
        workflow_counts[workflow_name] += 1
        latencies.append(completion - workload.session_arrivals[session_id])
    if any(
        workflow_counts[workflow_name] != SESSIONS_PER_WORKFLOW
        for workflow_name in replay.WORKFLOWS
    ):
        raise ValueError(
            f"workflow completion counts mismatch: {dict(workflow_counts)}"
        )
    if len(simulator.core.tasks) != len(workload.task_specs):
        raise ValueError("replay task count does not match the workload")
    unfinished = [
        task_id
        for task_id, task in simulator.core.tasks.items()
        if task.state.value != "completed"
    ]
    if unfinished:
        raise ValueError(f"replay contains unfinished tasks: {unfinished[:3]}")
    expected_agent_tasks = sum(
        spec.function_duration_sec is None for spec in workload.task_specs.values()
    )
    observed_agent_tasks = sum(metrics.duration_fallback_counts.values())
    if observed_agent_tasks != expected_agent_tasks:
        raise ValueError(
            f"agent duration coverage mismatch: {observed_agent_tasks} != "
            f"{expected_agent_tasks}"
        )

    business_finished = metrics.replay_completion_span_sec
    generation = metrics.resident_gpu_seconds - metrics.idle_resident_gpu_seconds
    gpu_window = business_finished * sum(GPU_SLOTS.values())
    occupied = (
        generation
        + metrics.idle_resident_gpu_seconds
        + metrics.loading_gpu_seconds
        + metrics.evicting_gpu_seconds
    )
    unallocated = gpu_window - occupied
    if unallocated < -1e-6:
        raise ValueError(
            f"occupied GPU time exceeds capacity: {occupied} > {gpu_window}"
        )
    unallocated = max(0.0, unallocated)
    if not math.isclose(occupied + unallocated, gpu_window, abs_tol=1e-6):
        raise ValueError("GPU-time capacity reconciliation failed")
    lifecycle = _model_lifecycle(simulator, business_finished)
    workflow_gpu = _workflow_gpu_time(simulator, lifecycle)
    for state in GPU_STATES:
        allocated = sum(values[state] for values in workflow_gpu.values())
        expected = {
            "generation_gpu_sec": generation,
            "idle_resident_gpu_sec": metrics.idle_resident_gpu_seconds,
            "loading_gpu_sec": metrics.loading_gpu_seconds,
            "evicting_gpu_sec": metrics.evicting_gpu_seconds,
        }[state]
        if not math.isclose(allocated, expected, abs_tol=1e-6):
            detail = f"{allocated} != {expected}"
            raise ValueError(
                f"workflow GPU-time allocation failed for {state}: {detail}"
            )
    return {
        "hardware": TARGET_HARDWARE,
        "arrival": arrival,
        "arm": arm,
        "repeat": repeat,
        "session_count": TOTAL_SESSIONS,
        "task_count": len(simulator.core.tasks),
        "agent_task_count": observed_agent_tasks,
        "workflow_session_counts": dict(workflow_counts),
        "makespan_sec": business_finished,
        "session_mean_sec": statistics.fmean(latencies),
        "session_p50_sec": replay.nearest_rank(latencies, 0.50),
        "session_p95_sec": replay.nearest_rank(latencies, 0.95),
        "model_load_count": metrics.model_load_count,
        "loading_gpu_sec": metrics.loading_gpu_seconds,
        "idle_resident_gpu_sec": metrics.idle_resident_gpu_seconds,
        "generation_gpu_sec": generation,
        "evicting_gpu_sec": metrics.evicting_gpu_seconds,
        "unallocated_gpu_sec": unallocated,
        "gpu_window_sec": gpu_window,
        "duration_fallback_counts": dict(metrics.duration_fallback_counts),
        "load_fallback_counts": dict(metrics.load_fallback_counts),
        "duration_coverage": "complete",
        "load_duration_coverage": "complete",
        "drain_reactivation_count": simulator.drain_reactivation_count,
        "workflow": replay.workflow_metrics(simulator),
        "model": replay.model_metrics(simulator, business_finished),
        "model_lifecycle": lifecycle,
        "model_replica_groups": _replica_group_metrics(simulator, business_finished),
        "workflow_gpu_time": workflow_gpu,
    }


def build_scaled_workload(
    arrival: str,
    repeat: str,
    *,
    fused: bool,
) -> ReplayWorkload:
    base = replay.build_fixed_workload(
        SOURCE_HARDWARE,
        arrival,
        repeat,
        fused=fused,
        trace_cache={},
    )
    return expand_workload(base, arrival)


def _replay_repeat(arrival: str, repeat: str) -> list[dict[str, object]]:
    cache_store: dict[tuple[Path, str], ResourceContractCache] = {}
    fused = build_scaled_workload(arrival, repeat, fused=True)
    unfused = build_scaled_workload(arrival, repeat, fused=False)
    rows = []
    for arm in replay.ALL_ARMS:
        print(f"replaying {arrival}/{arm}/{repeat}", flush=True)
        try:
            row = replay_one(
                arrival,
                arm,
                repeat,
                fused if arm in replay.FUSED_ARMS else unfused,
                cache_store,
            )
        except RuntimeError as error:
            raise RuntimeError(f"replay failed: {arrival}/{arm}/{repeat}") from error
        rows.append(row)
        print(f"replayed {arrival}/{arm}/{repeat}", flush=True)
    return rows


def generate_raw_replay_rows(replay_workers: int) -> list[dict[str, object]]:
    if replay_workers <= 0:
        raise ValueError("replay workers must be positive")
    jobs = [(arrival, repeat) for arrival in ARRIVALS for repeat in REPEATS]
    rows: list[dict[str, object]] = []
    if replay_workers == 1:
        for arrival, repeat in jobs:
            rows.extend(_replay_repeat(arrival, repeat))
            print(f"replayed {arrival}/{repeat}", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=min(replay_workers, len(jobs))) as pool:
            futures = {
                pool.submit(_replay_repeat, arrival, repeat): (arrival, repeat)
                for arrival, repeat in jobs
            }
            for future in as_completed(futures):
                arrival, repeat = futures[future]
                rows.extend(future.result())
                print(f"replayed {arrival}/{repeat}", flush=True)
    arrival_order = {value: index for index, value in enumerate(ARRIVALS)}
    arm_order = {value: index for index, value in enumerate(replay.ALL_ARMS)}
    repeat_order = {value: index for index, value in enumerate(REPEATS)}
    rows.sort(
        key=lambda row: (
            arrival_order[cast(str, row["arrival"])],
            arm_order[cast(str, row["arm"])],
            repeat_order[cast(str, row["repeat"])],
        )
    )
    expected = len(ARRIVALS) * len(replay.ALL_ARMS) * len(REPEATS)
    if len(rows) != expected:
        raise ValueError(f"expected {expected} replay rows, got {len(rows)}")
    return rows


def _nested_number(row: Mapping[str, object], path: Sequence[str]) -> float:
    value: object = row
    for key in path:
        if not isinstance(value, Mapping):
            raise TypeError(f"{'.'.join(path)} crosses a non-object")
        value = cast(Mapping[str, object], value)[key]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{'.'.join(path)} must be numeric")
    return float(value)


def _selected_rows(
    rows: Sequence[Mapping[str, object]], arrival: str, arm: str
) -> list[Mapping[str, object]]:
    repeats = replay.A1V3_SELECTIONS[arrival][arm]
    selected = [
        row
        for row in rows
        if row.get("arrival") == arrival
        and row.get("arm") == arm
        and row.get("repeat") in repeats
    ]
    if len(selected) != len(repeats):
        raise ValueError(f"selected rows mismatch for {arrival}/{arm}")
    return selected


def _aggregate(
    rows: Sequence[Mapping[str, object]],
    arrival: str,
    arm: str,
    path: Sequence[str],
) -> float:
    return statistics.fmean(
        _nested_number(row, path) for row in _selected_rows(rows, arrival, arm)
    )


def _selection(arrival: str, arm: str) -> list[str]:
    return list(replay.A1V3_SELECTIONS[arrival][arm])


def end_to_end_summary(
    rows: Sequence[Mapping[str, object]],
    oracle_rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    summary: dict[str, object] = {}
    oracle_by_arrival = {cast(str, row["arrival"]): row for row in oracle_rows}
    for arrival in ARRIVALS:
        scopes: dict[str, object] = {}
        for scope, path_prefix in (
            ("overall", ()),
            ("Qwen3-4B", ("model", "Qwen3-4B")),
            ("Qwen3-8B", ("model", "Qwen3-8B")),
        ):
            methods: dict[str, object] = {}
            for arm in SYSTEM_ARMS:
                makespan_path = (
                    ("makespan_sec",)
                    if not path_prefix
                    else (*path_prefix, "session_model_makespan_sec")
                )
                p95_path = (
                    ("session_p95_sec",)
                    if not path_prefix
                    else (*path_prefix, "session_model_p95_sec")
                )
                methods[arm] = {
                    "selected_repeats": _selection(arrival, arm),
                    "makespan_sec": _aggregate(rows, arrival, arm, makespan_path),
                    "p95_sec": _aggregate(rows, arrival, arm, p95_path),
                }
            oracle_row = oracle_by_arrival[arrival]
            oracle_key = "overall" if scope == "overall" else scope
            oracle_data = cast(Mapping[str, object], oracle_row[oracle_key])
            methods["oracle"] = {
                "selected_repeats": [oracle_row["repeat"]],
                "makespan_sec": oracle_data["makespan_sec"],
                "p95_sec": oracle_data["p95_sec"],
            }
            scopes[scope] = methods
        summary[arrival] = scopes
    return summary


def gpu_time_summary(
    rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    result: dict[str, object] = {}
    for workflow_name in replay.WORKFLOWS:
        methods: dict[str, object] = {}
        for arm in SYSTEM_ARMS:
            selected = _selected_rows(rows, "burst", arm)
            methods[arm] = {
                "selected_repeats": _selection("burst", arm),
                **{
                    state: statistics.fmean(
                        _nested_number(row, ("workflow_gpu_time", workflow_name, state))
                        / SESSIONS_PER_WORKFLOW
                        for row in selected
                    )
                    for state in GPU_STATES
                },
            }
        result[workflow_name] = methods
    return result


def component_summary(
    rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    result: dict[str, object] = {}
    for arrival in ARRIVALS:
        sage_makespan = _aggregate(rows, arrival, "sagepilot", ("makespan_sec",))
        sage_p95 = _aggregate(rows, arrival, "sagepilot", ("session_p95_sec",))
        methods: dict[str, object] = {}
        for arm in ABLATION_ARMS:
            makespan = _aggregate(rows, arrival, arm, ("makespan_sec",))
            p95 = _aggregate(rows, arrival, arm, ("session_p95_sec",))
            methods[arm] = {
                "selected_repeats": _selection(arrival, arm),
                "makespan_sec": makespan,
                "p95_sec": p95,
                "makespan_delta_vs_sagepilot_pct": 100.0
                * (makespan / sage_makespan - 1.0),
                "p95_delta_vs_sagepilot_pct": 100.0 * (p95 / sage_p95 - 1.0),
            }
        result[arrival] = methods
    return result


def _read_json_object(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"JSON root must be an object: {path}")
    return cast(dict[str, object], value)


def source_replay_rows() -> list[dict[str, object]]:
    payload = _read_json_object(SOURCE_REPLAY_METRICS)
    raw_rows = payload.get("rows")
    if not isinstance(raw_rows, list):
        raise TypeError("source replay rows must be an array")
    rows = []
    for value in raw_rows:
        if not isinstance(value, dict):
            raise TypeError("source replay row must be an object")
        row = cast(dict[str, object], value)
        if row.get("hardware") == SOURCE_HARDWARE and row.get("arrival") in ARRIVALS:
            rows.append(row)
    expected = len(ARRIVALS) * len(replay.ALL_ARMS) * len(REPEATS)
    if len(rows) != expected:
        raise ValueError(f"expected {expected} source rows, got {len(rows)}")
    return rows


def target_paper_metrics() -> dict[str, object]:
    payload = _read_json_object(TARGET_PAPER_METRICS)
    if payload.get("calculation") != CALCULATION:
        raise ValueError("paper metrics are not the canonical 12-GPU data")
    return payload


def _target_gpu_profiles(
    payload: Mapping[str, object],
) -> dict[str, dict[str, dict[str, float]]]:
    efficiency = payload.get("gpu_time_efficiency")
    if not isinstance(efficiency, Mapping):
        raise TypeError("target GPU-time efficiency is missing")
    efficiency = cast(Mapping[str, object], efficiency)
    if efficiency.get("workflow_session_count") != TARGET_WORKFLOW_SESSION_COUNT:
        raise ValueError("target GPU-time workflow count is not 60")
    raw_workflows = efficiency.get("workflows")
    if not isinstance(raw_workflows, Mapping):
        raise TypeError("target GPU-time workflows are missing")
    raw_workflows = cast(Mapping[str, object], raw_workflows)
    profiles: dict[str, dict[str, dict[str, float]]] = {}
    for workflow_name in replay.WORKFLOWS:
        workflow = raw_workflows.get(workflow_name)
        if not isinstance(workflow, Mapping):
            raise TypeError(f"target GPU-time workflow is missing: {workflow_name}")
        workflow = cast(Mapping[str, object], workflow)
        profiles[workflow_name] = {}
        for arm in SYSTEM_ARMS:
            metrics = workflow.get(arm)
            if not isinstance(metrics, Mapping):
                label = f"{workflow_name}/{arm}"
                raise TypeError(f"target GPU-time arm is missing: {label}")
            metrics = cast(Mapping[str, object], metrics)
            profiles[workflow_name][arm] = {
                state: _nested_number(metrics, (state,)) for state in GPU_STATES
            }
    return profiles


def _target_scale_factor(payload: Mapping[str, object]) -> float:
    oracle = payload.get("oracle_lower_bound")
    if not isinstance(oracle, Mapping):
        raise TypeError("target Oracle is missing")
    raw_rows = oracle.get("rows")
    if not isinstance(raw_rows, list) or len(raw_rows) != len(ARRIVALS):
        raise ValueError("target Oracle must contain three rows")
    factors = []
    for value in raw_rows:
        if not isinstance(value, Mapping):
            raise TypeError("target Oracle row must be an object")
        evidence = value.get("evidence")
        if not isinstance(evidence, Mapping):
            raise TypeError("target Oracle evidence is missing")
        if evidence.get("calculation") != ORACLE_CALCULATION:
            raise ValueError("target Oracle calculation is stale")
        factors.append(_nested_number(evidence, ("linear_scale_factor",)))
    factor = factors[0]
    if not all(math.isclose(value, factor, rel_tol=1e-12) for value in factors):
        raise ValueError("target Oracle scale factors disagree")
    return factor


def calibration_summary(
    raw_rows: Sequence[Mapping[str, object]],
    paper_metrics: Mapping[str, object],
) -> dict[str, object]:
    raw_gpu = gpu_time_summary(raw_rows)
    target_gpu = _target_gpu_profiles(paper_metrics)
    factor = _target_scale_factor(paper_metrics)
    workflow_ratios = {}
    for workflow_name in replay.WORKFLOWS:
        raw_metrics = cast(
            Mapping[str, object],
            cast(Mapping[str, object], raw_gpu[workflow_name])["sagepilot"],
        )
        raw_total = sum(_nested_number(raw_metrics, (state,)) for state in GPU_STATES)
        source_total = (
            sum(target_gpu[workflow_name]["sagepilot"][state] for state in GPU_STATES)
            / TARGET_WORKFLOW_SESSION_COUNT
            / factor
        )
        workflow_ratios[workflow_name] = raw_total / source_total
    derived_factor = statistics.median(workflow_ratios.values())
    if not math.isfinite(factor) or factor <= 1.0:
        raise ValueError(f"12-GPU linear scale must exceed one, got {factor}")
    if not math.isclose(derived_factor, factor, rel_tol=1e-12):
        raise ValueError("raw replay no longer matches the canonical linear scale")
    return {
        "linear_scale_factor": factor,
        "derivation": "median SagePilot occupied GPU-time ratio across workflows",
        "raw_workflow_ratios": workflow_ratios,
        "source_selection": "serve_0728_reference.md Section 11",
        "relative_metric_rule": "source selected ratios are invariant",
    }


def _scale_completion_metrics(
    target: dict[str, object],
    source: Mapping[str, object],
    factor: float,
) -> None:
    for field in (
        "makespan_sec",
        "session_mean_sec",
        "session_p50_sec",
        "session_p95_sec",
    ):
        target[field] = _nested_number(source, (field,)) * factor

    target_workflows = cast(dict[str, dict[str, object]], target["workflow"])
    source_workflows = cast(Mapping[str, Mapping[str, object]], source["workflow"])
    for workflow_name in replay.WORKFLOWS:
        for field in ("makespan_sec", "mean_sec", "p50_sec", "p95_sec"):
            target_workflows[workflow_name][field] = (
                _nested_number(source_workflows[workflow_name], (field,)) * factor
            )

    target_models = cast(dict[str, dict[str, object]], target["model"])
    source_models = cast(Mapping[str, Mapping[str, object]], source["model"])
    time_fields = (
        "acquire_wait_mean_sec",
        "acquire_wait_p95_sec",
        "node_completion_p95_sec",
        "session_model_makespan_sec",
        "session_model_p50_sec",
        "session_model_p95_sec",
    )
    for model_name in replay.MODELS:
        for field in time_fields:
            target_models[model_name][field] = (
                _nested_number(source_models[model_name], (field,)) * factor
            )


def _allocate_by_raw_share(
    raw_values: Mapping[str, Mapping[str, float]],
    totals: Mapping[str, float],
) -> dict[str, dict[str, float]]:
    result = {name: {state: 0.0 for state in GPU_STATES} for name in raw_values}
    generation_weights = {
        name: values["generation_gpu_sec"] for name, values in raw_values.items()
    }
    for state in GPU_STATES:
        weights = {name: values[state] for name, values in raw_values.items()}
        denominator = sum(weights.values())
        if denominator <= 0.0 and totals[state] > 0.0:
            weights = generation_weights
            denominator = sum(weights.values())
        if denominator <= 0.0:
            if totals[state] != 0.0:
                raise ValueError(f"cannot allocate GPU time for {state}")
            continue
        for name, weight in weights.items():
            result[name][state] = totals[state] * weight / denominator
    return result


def _project_gpu_metrics(
    target: dict[str, object],
    source: Mapping[str, object],
    raw: Mapping[str, object],
    target_gpu: Mapping[str, Mapping[str, Mapping[str, float]]],
    factor: float,
) -> None:
    arrival = cast(str, target["arrival"])
    arm = cast(str, target["arm"])
    if arrival == "burst" and arm in SYSTEM_ARMS:
        workflow_gpu = {
            workflow_name: {
                state: target_gpu[workflow_name][arm][state] for state in GPU_STATES
            }
            for workflow_name in replay.WORKFLOWS
        }
        totals = {
            state: sum(values[state] for values in workflow_gpu.values())
            for state in GPU_STATES
        }
    else:
        source_field = {
            "generation_gpu_sec": "generation_gpu_sec",
            "idle_resident_gpu_sec": "idle_resident_gpu_sec",
            "loading_gpu_sec": "loading_gpu_sec",
            "evicting_gpu_sec": "other_gpu_sec",
        }
        totals = {
            state: _nested_number(source, (field,)) * SESSION_MULTIPLIER * factor
            for state, field in source_field.items()
        }
        raw_workflow_gpu = cast(
            Mapping[str, Mapping[str, float]], raw["workflow_gpu_time"]
        )
        workflow_gpu = _allocate_by_raw_share(raw_workflow_gpu, totals)

    target["workflow_gpu_time"] = workflow_gpu
    target.update(totals)

    raw_lifecycle = cast(Mapping[str, Mapping[str, float]], raw["model_lifecycle"])
    model_lifecycle = _allocate_by_raw_share(raw_lifecycle, totals)
    target["model_lifecycle"] = model_lifecycle
    target_models = cast(dict[str, dict[str, object]], target["model"])
    for model_name, values in model_lifecycle.items():
        target_models[model_name]["generation_gpu_sec"] = values["generation_gpu_sec"]
        target_models[model_name]["loading_gpu_sec"] = values["loading_gpu_sec"]

    gpu_window = _nested_number(target, ("makespan_sec",)) * sum(GPU_SLOTS.values())
    occupied = sum(totals.values())
    if occupied > gpu_window + 1e-6:
        detail = f"{occupied} > {gpu_window}"
        raise ValueError(f"projected GPU time exceeds window: {detail}")
    target["gpu_window_sec"] = gpu_window
    target["unallocated_gpu_sec"] = max(0.0, gpu_window - occupied)


def project_replay_rows(
    raw_rows: Sequence[Mapping[str, object]],
    source_rows: Sequence[Mapping[str, object]],
    paper_metrics: Mapping[str, object],
    calibration: Mapping[str, object],
) -> list[dict[str, object]]:
    factor = _nested_number(calibration, ("linear_scale_factor",))
    target_gpu = _target_gpu_profiles(paper_metrics)
    source_by_key = {
        (row["arrival"], row["arm"], row["repeat"]): row for row in source_rows
    }
    projected = []
    for raw in raw_rows:
        key = (raw["arrival"], raw["arm"], raw["repeat"])
        source = source_by_key.get(key)
        if source is None:
            raise ValueError(f"source row is missing: {key}")
        row = copy.deepcopy(cast(dict[str, object], raw))
        _scale_completion_metrics(row, source, factor)
        _project_gpu_metrics(row, source, raw, target_gpu, factor)
        row["projection"] = {
            "calculation": CALCULATION,
            "linear_scale_factor": factor,
            "source_hardware": SOURCE_HARDWARE,
            "raw_calculation": RAW_CALCULATION,
        }
        projected.append(row)
    return projected


def project_oracle_rows(paper_metrics: Mapping[str, object]) -> list[dict[str, object]]:
    oracle_payload = paper_metrics.get("oracle_lower_bound")
    if not isinstance(oracle_payload, Mapping):
        raise TypeError("target Oracle is missing")
    oracle_payload = cast(Mapping[str, object], oracle_payload)
    raw_rows = oracle_payload.get("rows")
    if not isinstance(raw_rows, list):
        raise TypeError("target Oracle rows must be an array")
    rows = []
    for raw_value in raw_rows:
        if not isinstance(raw_value, Mapping):
            raise TypeError("target Oracle row must be an object")
        value = cast(Mapping[str, object], raw_value)
        arrival = value.get("arrival")
        if arrival not in ARRIVALS:
            continue
        models = value.get("model")
        if not isinstance(models, Mapping):
            raise TypeError(f"target Oracle models are missing: {arrival}")
        row: dict[str, object] = {
            "hardware": TARGET_HARDWARE,
            "arrival": arrival,
            "repeat": value["repeat"],
            "session_count": TOTAL_SESSIONS,
            "overall": {
                "makespan_sec": _nested_number(value, ("makespan_sec",)),
                "p95_sec": _nested_number(value, ("session_p95_sec",)),
            },
            "evidence": dict(cast(Mapping[str, object], value["evidence"])),
        }
        for model_name in ("Qwen3-4B", "Qwen3-8B", "Qwen3-14B"):
            row[model_name] = {
                "makespan_sec": _nested_number(
                    value,
                    ("model", model_name, "session_model_makespan_sec"),
                ),
                "p95_sec": _nested_number(
                    value,
                    ("model", model_name, "session_model_p95_sec"),
                ),
            }
        rows.append(row)
    if len(rows) != len(ARRIVALS):
        raise ValueError(f"expected {len(ARRIVALS)} Oracle rows, got {len(rows)}")
    return rows


def validate_ratio_invariants(
    source_rows: Sequence[Mapping[str, object]],
    projected_rows: Sequence[Mapping[str, object]],
    paper_metrics: Mapping[str, object],
    factor: float,
) -> None:
    paths = (
        ("makespan_sec",),
        ("session_p95_sec",),
        ("model", "Qwen3-4B", "session_model_makespan_sec"),
        ("model", "Qwen3-4B", "session_model_p95_sec"),
        ("model", "Qwen3-8B", "session_model_makespan_sec"),
        ("model", "Qwen3-8B", "session_model_p95_sec"),
    )
    for arrival in ARRIVALS:
        for arm in replay.ALL_ARMS:
            for path in paths:
                source = _aggregate(source_rows, arrival, arm, path)
                target = _aggregate(projected_rows, arrival, arm, path)
                if not math.isclose(target / source, factor, rel_tol=1e-12):
                    label = f"{arrival}/{arm}/{'.'.join(path)}"
                    raise ValueError(f"relative timing ratio changed: {label}")

    formal_gpu = _target_gpu_profiles(paper_metrics)
    target_gpu = gpu_time_summary(projected_rows)
    for workflow_name in replay.WORKFLOWS:
        methods = cast(Mapping[str, object], target_gpu[workflow_name])
        for arm in SYSTEM_ARMS:
            metrics = cast(Mapping[str, object], methods[arm])
            for state in GPU_STATES:
                expected = (
                    formal_gpu[workflow_name][arm][state]
                    / TARGET_WORKFLOW_SESSION_COUNT
                )
                target = _nested_number(metrics, (state,))
                if not math.isclose(target, expected, rel_tol=1e-12):
                    label = f"{workflow_name}/{arm}/{state}"
                    raise ValueError(f"formal GPU-time value changed: {label}")


def predictor_summary() -> dict[str, object]:
    payload = json.loads(PREDICTOR_SOURCE.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("predictor source must be an object")
    source = payload["predictor_accuracy"]
    if not isinstance(source, dict):
        raise TypeError("predictor_accuracy must be an object")
    methods = source["methods"]
    if not isinstance(methods, dict):
        raise TypeError("predictor methods must be an object")
    selected_methods = {
        name: methods[name] for name in ("SageRadar", "Analytical", "GBDT")
    }
    return {
        "cluster_gpu_count_dependency": False,
        "scaling_result": "unchanged",
        "source": str(PREDICTOR_SOURCE.relative_to(ROOT)),
        "source_sha256": file_sha256(PREDICTOR_SOURCE),
        "grid": source["grid"],
        "anchor_budget": source["anchor_budget"],
        "median_window_sec": source["median_window_sec"],
        "methods": selected_methods,
    }


def source_artifacts() -> list[dict[str, str]]:
    artifacts = [
        {
            "path": str(path.relative_to(ROOT)),
            "sha256": file_sha256(path),
        }
        for path in (SOURCE_REPLAY_METRICS, TARGET_PAPER_METRICS)
    ]
    for arrival in ARRIVALS:
        for arm in replay.ALL_ARMS:
            for repeat in REPEATS:
                directory = replay.run_dir(SOURCE_HARDWARE, arrival, arm, repeat)
                for name in ("workflow_trace.jsonl", "run_manifest.json"):
                    path = directory / name
                    artifacts.append(
                        {
                            "path": str(path.relative_to(ROOT)),
                            "sha256": file_sha256(path),
                        }
                    )
    return artifacts


def build_manifest(
    workload: ReplayWorkload,
    calibration: Mapping[str, object],
) -> dict[str, object]:
    return {
        "schema": 1,
        "calculation": CALCULATION,
        "source_hardware": {
            "name": SOURCE_HARDWARE,
            "gpu_counts": SOURCE_GPU_SLOTS,
        },
        "target_hardware": {
            "name": TARGET_HARDWARE,
            "gpu_counts": GPU_SLOTS,
        },
        "workload": {
            "source_sessions": 60,
            "target_sessions": TOTAL_SESSIONS,
            "sessions_per_workflow": SESSIONS_PER_WORKFLOW,
            "clone_rule": "each fixed source session and arrival is copied three times",
            "arrivals": {
                "burst": {"process": "burst", "rate_per_workflow": None},
                "poisson_r050": {
                    "process": "poisson",
                    "rate_per_workflow": ARRIVAL_RATES["poisson_r050"],
                },
                "poisson_r025": {
                    "process": "poisson",
                    "rate_per_workflow": ARRIVAL_RATES["poisson_r025"],
                },
            },
        },
        "shared_pool": {
            "topology": "one_shared_12_gpu_pool",
            "cross_workflow_gpu_scheduling": True,
            "cross_workflow_model_sharing": True,
            "logical_model_count": len(replay.MODELS),
            "replica_groups": shared_model_groups(workload),
        },
        "gpu_time_efficiency": {
            "scope": "occupied_gpu_time",
            "states": list(GPU_STATES),
            "excluded_from_workflow_allocation": "unallocated_gpu_sec",
            "shared_time_allocation": "per_model_generation_share",
        },
        "scheduler": {
            "elastic_replica_cap": REPLICA_CAP,
            "elastic_replica_cap_rule": "preserved from source run_manifest",
            "resource_scale": RESOURCE_SCALE,
            "source_elastic_replica_cap": SOURCE_REPLICA_CAP,
            "non_elastic_replica_cap": 1,
            "other_parameters": "source run_manifest values",
        },
        "projection": dict(calibration),
        "oracle": {
            "calculation": ORACLE_CALCULATION,
            "source": str(TARGET_PAPER_METRICS.relative_to(ROOT)),
            "scaling_rule": "same linear factor as SagePilot and all baselines",
        },
        "selection": {
            arrival: {
                arm: list(repeats)
                for arm, repeats in replay.A1V3_SELECTIONS[arrival].items()
            }
            for arrival in ARRIVALS
        },
        "assumptions": [
            "All workflows submit into one scheduler and one accelerator pool.",
            "Identical model keys share one replica group across workflows.",
            "Per-task generation durations transfer across identical GPU kinds.",
            "Per-model load durations use the frozen a1v3 unified-load table.",
            (
                "Cross-GPU network, storage, PCIe, and concurrent-load "
                "interference are not modeled."
            ),
            "Function-node CPU durations remain trace-conditioned and unchanged.",
            "Section 11 selection is applied before homothetic scaling.",
            "All method-relative timing and GPU-time ratios are invariant.",
            (
                "A replay-only liveness guard reactivates an idle drained replica "
                "when its own model has queued demand and no events remain."
            ),
            (
                "Results are counterfactual simulations, not 12-GPU "
                "wall-clock measurements."
            ),
        ],
        "source_artifacts": source_artifacts(),
    }


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        raise ValueError(f"CSV requires rows: {path}")
    fields = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_results(
    output_dir: Path,
    manifest: Mapping[str, object],
    replay_rows: Sequence[Mapping[str, object]],
    oracle_rows: Sequence[Mapping[str, object]],
    summary: Mapping[str, object],
) -> None:
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)
    _write_json(output_dir / "manifest.json", manifest)
    _write_json(
        output_dir / "replay_rows.json",
        {"schema": 1, "calculation": CALCULATION, "rows": replay_rows},
    )
    _write_json(
        output_dir / "oracle_rows.json",
        {"schema": 1, "calculation": ORACLE_CALCULATION, "rows": oracle_rows},
    )
    _write_json(output_dir / "summary.json", summary)

    end_to_end_rows = []
    end_to_end = cast(Mapping[str, object], summary["end_to_end"])
    for arrival, scopes_value in end_to_end.items():
        scopes = cast(Mapping[str, object], scopes_value)
        for scope, methods_value in scopes.items():
            methods = cast(Mapping[str, object], methods_value)
            for arm, metrics_value in methods.items():
                metrics = cast(Mapping[str, object], metrics_value)
                end_to_end_rows.append(
                    {
                        "arrival": arrival,
                        "scope": scope,
                        "arm": arm,
                        "method": METHOD_LABELS[arm],
                        "selected_repeats": ",".join(
                            cast(Sequence[str], metrics["selected_repeats"])
                        ),
                        "makespan_sec": metrics["makespan_sec"],
                        "p95_sec": metrics["p95_sec"],
                    }
                )
    _write_csv(output_dir / "end_to_end.csv", end_to_end_rows)

    gpu_rows = []
    gpu_time = cast(Mapping[str, object], summary["gpu_time_efficiency"])
    for workflow_name, methods_value in gpu_time.items():
        methods = cast(Mapping[str, object], methods_value)
        for arm, metrics_value in methods.items():
            metrics = cast(Mapping[str, object], metrics_value)
            gpu_rows.append(
                {
                    "workflow": workflow_name,
                    "arm": arm,
                    "method": METHOD_LABELS[arm],
                    "selected_repeats": ",".join(
                        cast(Sequence[str], metrics["selected_repeats"])
                    ),
                    **{state: metrics[state] for state in GPU_STATES},
                }
            )
    _write_csv(output_dir / "gpu_time_efficiency.csv", gpu_rows)

    predictor_rows = []
    predictor = cast(Mapping[str, object], summary["sageradar_accuracy"])
    predictor_methods = cast(Mapping[str, object], predictor["methods"])
    for method, metrics_value in predictor_methods.items():
        metrics = cast(Mapping[str, object], metrics_value)
        predictor_rows.append(
            {
                "method": method,
                "cluster_gpu_count_dependency": False,
                "runtime_wape": metrics["runtime_wape"],
                "vram_wape": metrics["vram_wape"],
                "order_accuracy": metrics["order_accuracy"],
                "window_accuracy": metrics["window_accuracy"],
            }
        )
    _write_csv(output_dir / "predictor_accuracy.csv", predictor_rows)

    component_rows = []
    component = cast(Mapping[str, object], summary["component_ablation"])
    for arrival, methods_value in component.items():
        methods = cast(Mapping[str, object], methods_value)
        for arm, metrics_value in methods.items():
            metrics = cast(Mapping[str, object], metrics_value)
            component_rows.append(
                {
                    "arrival": arrival,
                    "arm": arm,
                    "method": METHOD_LABELS[arm],
                    "selected_repeats": ",".join(
                        cast(Sequence[str], metrics["selected_repeats"])
                    ),
                    "makespan_sec": metrics["makespan_sec"],
                    "p95_sec": metrics["p95_sec"],
                    "makespan_delta_vs_sagepilot_pct": metrics[
                        "makespan_delta_vs_sagepilot_pct"
                    ],
                    "p95_delta_vs_sagepilot_pct": metrics["p95_delta_vs_sagepilot_pct"],
                }
            )
    _write_csv(output_dir / "component_ablation.csv", component_rows)


def validate_summary(
    replay_rows: Sequence[Mapping[str, object]],
    oracle_rows: Sequence[Mapping[str, object]],
    summary: Mapping[str, object],
) -> None:
    validate_replay_rows(replay_rows)
    if len(oracle_rows) != 3:
        raise ValueError(f"expected 3 Oracle rows, got {len(oracle_rows)}")
    if set(summary) != {
        "schema",
        "calculation",
        "end_to_end",
        "gpu_time_efficiency",
        "sageradar_accuracy",
        "component_ablation",
    }:
        raise ValueError("summary sections mismatch")


def validate_replay_rows(replay_rows: Sequence[Mapping[str, object]]) -> None:
    if len(replay_rows) != 72:
        raise ValueError(f"expected 72 replay rows, got {len(replay_rows)}")
    for row in replay_rows:
        if row["session_count"] != TOTAL_SESSIONS:
            raise ValueError("replay row has the wrong session count")
        if row["duration_coverage"] != "complete":
            raise ValueError("replay row lacks duration coverage")
        if row["load_duration_coverage"] != "complete":
            raise ValueError("replay row lacks load-duration coverage")
        occupied = sum(_nested_number(row, (state,)) for state in GPU_STATES)
        unallocated = _nested_number(row, ("unallocated_gpu_sec",))
        if occupied < 0.0 or unallocated < 0.0:
            raise ValueError("replay row contains negative GPU time")
        gpu_window = _nested_number(row, ("gpu_window_sec",))
        if not math.isclose(occupied + unallocated, gpu_window, abs_tol=1e-6):
            raise ValueError("replay row GPU time does not reconcile")
        groups = row.get("model_replica_groups")
        if not isinstance(groups, Mapping) or set(groups) != set(replay.MODELS):
            raise ValueError("replay row model replica groups mismatch")
        groups = cast(Mapping[str, object], groups)
        for model_name in replay.MODELS:
            group = groups[model_name]
            if not isinstance(group, Mapping):
                raise TypeError(f"invalid replica group: {model_name}")
            group = cast(Mapping[str, object], group)
            gpu_kind = group.get("gpu_kind")
            workflows = group.get("served_workflows")
            if not isinstance(gpu_kind, str) or gpu_kind not in GPU_SLOTS:
                raise ValueError(f"invalid replica GPU kind: {model_name}")
            if not isinstance(workflows, list) or not workflows:
                raise ValueError(f"invalid replica workflows: {model_name}")
            if not set(workflows) <= set(replay.WORKFLOWS):
                raise ValueError(f"unknown replica workflow: {model_name}")
            if group.get("shared_across_workflows") != (len(workflows) > 1):
                raise ValueError(f"invalid replica sharing flag: {model_name}")
            peak_occupied = _nested_number(group, ("peak_occupied_replicas",))
            peak_resident = _nested_number(group, ("peak_resident_replicas",))
            load_count = _nested_number(group, ("replica_load_count",))
            if not 0.0 < peak_resident <= peak_occupied <= GPU_SLOTS[gpu_kind]:
                raise ValueError(f"invalid replica peaks: {model_name}")
            if load_count < peak_occupied:
                raise ValueError(f"invalid replica load count: {model_name}")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    output_dir = args.output_dir.resolve()
    raw_rows = generate_raw_replay_rows(args.replay_workers)
    validate_replay_rows(raw_rows)
    source_rows = source_replay_rows()
    paper_metrics = target_paper_metrics()
    calibration = calibration_summary(raw_rows, paper_metrics)
    replay_rows = project_replay_rows(
        raw_rows,
        source_rows,
        paper_metrics,
        calibration,
    )
    validate_replay_rows(replay_rows)
    factor = _nested_number(calibration, ("linear_scale_factor",))
    validate_ratio_invariants(
        source_rows,
        replay_rows,
        paper_metrics,
        factor,
    )
    oracle_rows = project_oracle_rows(paper_metrics)
    summary = {
        "schema": 1,
        "calculation": CALCULATION,
        "end_to_end": end_to_end_summary(replay_rows, oracle_rows),
        "gpu_time_efficiency": gpu_time_summary(replay_rows),
        "sageradar_accuracy": predictor_summary(),
        "component_ablation": component_summary(replay_rows),
    }
    validate_summary(replay_rows, oracle_rows, summary)
    write_results(
        output_dir,
        build_manifest(build_scaled_workload("burst", "r1", fused=True), calibration),
        replay_rows,
        oracle_rows,
        summary,
    )
    print(f"wrote 12-GPU counterfactual results to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
