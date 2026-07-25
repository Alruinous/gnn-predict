from __future__ import annotations

import heapq
import json
import re
import shutil
import statistics
import tempfile
from collections import defaultdict, deque
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal, cast

import yaml

from experiment.workflow.analysis import interval_union_duration, read_jsonl
from experiment.workflow.artifacts import canonical_json, file_sha256
from workflow.artifacts import (
    AcceleratorConfig,
    ResourceContract,
    ResourceContractCache,
    SchedulerConfig,
    SchedulerPolicy,
    load_resource_contract_cache,
)
from workflow.replica import ModelDeploymentConfig, ReplicaLoadResult
from workflow.scheduler import (
    AgentTaskRuntimeReport,
    EvictReplicaAction,
    FunctionTaskRuntimeReport,
    GrantInfo,
    LoadReplicaAction,
    OutputReport,
    SchedulerCore,
)
from workflow.schema import AgentNodeConfig, FunctionNodeConfig, Workflow
from workflow.types import ModelReplicaState, NodeTaskState, WorkflowModelFeatureKey

PolicyName = Literal[
    "profile_cache",
    "trace_calibrated_cache",
    "trace_calibrated_cache_reclaim",
    "history",
    "fifo",
    "kairos",
]

# Replay arms that exercise a scheduler policy directly rather than swapping the
# prediction source; they let the counterfactual compare against the serve baselines.
SCHEDULER_POLICY_ARMS: Mapping[PolicyName, SchedulerPolicy] = {
    "fifo": "fifo",
    "kairos": "kairos",
    "history": "history",
}

MIN_CALIBRATION_SAMPLES = 3
GPU_MEMORY_MB = {"v100": 32_768, "a100": 81_920}
POLICIES: tuple[PolicyName, ...] = (
    "profile_cache",
    "trace_calibrated_cache",
    "trace_calibrated_cache_reclaim",
    "history",
)


@dataclass(frozen=True, slots=True)
class TaskSample:
    session_id: str
    workflow_name: str
    node_id: str
    duration_sec: float
    input_tokens: int | None
    output_tokens: int | None
    admitted_batch_size: int | None
    model_key: str | None
    gpu_kind: str | None
    prediction_key: WorkflowModelFeatureKey | None
    predicted_run_sec: float | None
    predicted_load_sec: float | None

    @property
    def is_agent(self) -> bool:
        return self.model_key is not None


@dataclass(frozen=True, slots=True)
class LoadSample:
    model_name: str
    model_key: str
    gpu_kind: str
    duration_sec: float
    cold_accelerator: bool


@dataclass(frozen=True, slots=True)
class TraceRun:
    path: Path
    run_id: str
    run_started: float
    run_finished: float
    session_arrivals: Mapping[str, float]
    session_workflows: Mapping[str, str]
    tasks: Mapping[tuple[str, str], TaskSample]
    loads: tuple[LoadSample, ...]
    eviction_durations: Mapping[str, tuple[float, ...]]
    model_names: Mapping[str, str]
    completed_session_count: int


@dataclass(frozen=True, slots=True)
class CalibrationRecord:
    prediction_key: WorkflowModelFeatureKey
    run_tier: str
    run_sample_count: int
    load_tier: str
    load_sample_count: int
    original_run_sec: float
    calibrated_run_sec: float
    original_load_sec: float
    calibrated_load_sec: float


@dataclass(frozen=True, slots=True)
class CalibrationResult:
    cache: ResourceContractCache
    records: tuple[CalibrationRecord, ...]


@dataclass(frozen=True, slots=True)
class CohortDefinition:
    name: str
    trace_paths: tuple[Path, ...]
    gpu_slots: Mapping[str, int]


@dataclass(frozen=True, slots=True)
class AgentDurationTable:
    exact: Mapping[tuple[str, str, int], float]
    node_batch: Mapping[tuple[str, str, int], float]
    model_batch: Mapping[tuple[str, str, int], float]
    task_any_batch: Mapping[tuple[str, str], float]


@dataclass(frozen=True, slots=True)
class TaskSpec:
    session_id: str
    workflow_name: str
    node_id: str
    input_tokens: int | None
    output_tokens: int | None
    function_duration_sec: float | None


@dataclass(frozen=True, slots=True)
class ReplayWorkload:
    name: str
    workflows: Mapping[str, Workflow]
    session_arrivals: Mapping[str, float]
    session_workflows: Mapping[str, str]
    task_specs: Mapping[tuple[str, str], TaskSpec]
    agent_durations: AgentDurationTable
    cold_load_sec: Mapping[tuple[str, str], float]
    warm_load_sec: Mapping[tuple[str, str], float]
    any_load_sec: Mapping[tuple[str, str], float]
    eviction_sec: Mapping[str, float]
    gpu_slots: Mapping[str, int]


@dataclass(frozen=True, slots=True)
class ReplayMetrics:
    policy: PolicyName
    replay_completion_span_sec: float
    mean_session_latency_sec: float
    completion_by_family_sec: Mapping[str, float]
    loading_gpu_seconds: float
    resident_gpu_seconds: float
    idle_resident_gpu_seconds: float
    evicting_gpu_seconds: float
    non_active_lifecycle_gpu_seconds: float
    pipeline_bubble_ratio: float
    model_load_count: int
    model_reuse_count: int
    model_eviction_count: int
    prefetch_count: int
    prefetch_ready_before_acquire_count: int
    prefetch_overrun_seconds: float
    reclaimed_replica_count: int
    duration_fallback_counts: Mapping[str, int]
    load_fallback_counts: Mapping[str, int]


@dataclass(slots=True)
class _ReplicaTimeline:
    model_key: str
    model_name: str
    gpu_kind: str
    load_reason: str
    load_started: float
    load_finished: float | None = None
    eviction_started: float | None = None
    eviction_finished: float | None = None
    executions: list[tuple[float, float]] = field(default_factory=list)
    first_grant_at: float | None = None
    first_acquire_at: float | None = None


@dataclass(order=True, slots=True)
class _Event:
    at: float
    sequence: int
    kind: str = field(compare=False)
    payload: object = field(compare=False)


@dataclass(frozen=True, slots=True)
class _ActiveFunction:
    task_id: str
    session_id: str
    workflow_name: str
    node_id: str
    started_at: float


@dataclass(frozen=True, slots=True)
class _ActiveAgent:
    grant: GrantInfo
    workflow_name: str
    started_at: float
    output_tokens: int


class _IndexedPredictionCache:
    def __init__(self, cache: ResourceContractCache) -> None:
        self.version = cache.version
        self.environment = cache.environment
        self.entries = cache.entries
        self._cache = cache
        sequences: dict[tuple[str, str, int], set[int]] = defaultdict(set)
        outputs: dict[tuple[str, str, int, int], set[int]] = defaultdict(set)
        for entry in cache.entries:
            key = entry.key
            if key.phase != "decode":
                continue
            sequences[(key.model_name, key.gpu_name, key.batch_size)].add(
                key.sequence_length
            )
            outputs[
                (
                    key.model_name,
                    key.gpu_name,
                    key.sequence_length,
                    key.batch_size,
                )
            ].add(key.decode_output_length)
        self._sequences = {
            key: tuple(sorted(values)) for key, values in sequences.items()
        }
        self._outputs = {key: tuple(sorted(values)) for key, values in outputs.items()}

    def lookup(self, key: WorkflowModelFeatureKey) -> ResourceContract:
        return self._cache.lookup(key)

    def lookup_decode(
        self,
        *,
        model_name: str,
        gpu_kind: str,
        batch_size: int = 1,
        sequence_length: int,
        decode_output_length: int,
    ) -> ResourceContract:
        return self._cache.lookup_decode(
            model_name=model_name,
            gpu_kind=gpu_kind,
            batch_size=batch_size,
            sequence_length=sequence_length,
            decode_output_length=decode_output_length,
        )

    def decode_sequence_lengths(
        self,
        model_name: str,
        gpu_kind: str,
        batch_size: int = 1,
    ) -> tuple[int, ...]:
        return self._sequences.get((model_name, gpu_kind, batch_size), ())

    def decode_output_lengths(
        self,
        model_name: str,
        gpu_kind: str,
        sequence_length: int,
        batch_size: int = 1,
    ) -> tuple[int, ...]:
        return self._outputs.get(
            (model_name, gpu_kind, sequence_length, batch_size),
            (),
        )


def read_trace_run(
    path: Path,
    *,
    expected_sessions: int | None = None,
) -> TraceRun:
    events = read_jsonl(path)
    if not events:
        raise ValueError(f"workflow trace is empty: {path}")
    run_starts = _event_timestamps(events, "run_started")
    run_finishes = _event_timestamps(events, "run_finished")
    if len(run_starts) != 1 or len(run_finishes) != 1:
        raise ValueError(f"trace must have one run start and finish: {path}")
    if run_finishes[0] < run_starts[0]:
        raise ValueError(f"run finish precedes start: {path}")

    run_ids = {event.get("run_id") for event in events}
    if len(run_ids) != 1 or not isinstance(next(iter(run_ids)), str):
        raise ValueError(f"trace has inconsistent run ids: {path}")
    run_id = next(iter(run_ids))
    assert isinstance(run_id, str)

    terminal_events = [
        event
        for event in events
        if event.get("event_type") in ("session_completed", "session_failed")
    ]
    failed_sessions = [
        event
        for event in terminal_events
        if event.get("event_type") == "session_failed"
    ]
    if failed_sessions:
        raise ValueError(f"trace contains failed sessions: {path}")
    if expected_sessions is not None and len(terminal_events) != expected_sessions:
        raise ValueError(
            f"trace has {len(terminal_events)} completed sessions, "
            f"expected {expected_sessions}: {path}"
        )

    session_arrivals: dict[str, float] = {}
    session_workflows: dict[str, str] = {}
    for event in events:
        if event.get("event_type") != "session_submitted":
            continue
        session_id = _required_str(event, "session_id", path)
        workflow_name = _required_str(event, "workflow_name", path)
        if session_id in session_arrivals:
            raise ValueError(f"duplicate session submission {session_id}: {path}")
        session_arrivals[session_id] = _number(event["ts"])
        session_workflows[session_id] = workflow_name
    if set(session_arrivals) != {
        _required_str(event, "session_id", path) for event in terminal_events
    }:
        raise ValueError(f"submitted and completed sessions differ: {path}")

    tasks: dict[tuple[str, str], TaskSample] = {}
    model_names: dict[str, str] = {}
    requested: set[str] = set()
    granted: set[str] = set()
    finished_acquires: set[str] = set()
    for event in events:
        event_type = event.get("event_type")
        if event_type == "acquire_requested":
            requested.add(_required_str(event, "acquire_id", path))
        elif event_type == "acquire_granted":
            granted.add(_required_str(event, "acquire_id", path))
        if event_type != "task_execution_finished":
            continue
        payload = _payload(event, path)
        status = payload.get("status")
        if status != "success":
            raise ValueError(f"trace contains non-success task execution: {path}")
        session_id = _required_str(event, "session_id", path)
        workflow_name = _required_str(event, "workflow_name", path)
        node_id = _required_str(event, "node_id", path)
        identity = (session_id, node_id)
        if identity in tasks:
            raise ValueError(f"duplicate completed task {identity}: {path}")

        model_key = _optional_str(event.get("model_key"), "model_key", path)
        gpu_kind = _optional_str(event.get("gpu_kind"), "gpu_kind", path)
        prediction_raw = payload.get("prediction_key")
        prediction_key = (
            WorkflowModelFeatureKey.model_validate(prediction_raw)
            if prediction_raw is not None
            else None
        )
        if (model_key is None) != (prediction_key is None):
            raise ValueError(f"agent task metadata is incomplete: {identity} in {path}")
        if prediction_key is not None:
            assert model_key is not None
            existing = model_names.setdefault(model_key, prediction_key.model_name)
            if existing != prediction_key.model_name:
                raise ValueError(f"model key maps to multiple names: {model_key}")
            finished_acquires.add(_required_str(event, "acquire_id", path))

        tasks[identity] = TaskSample(
            session_id=session_id,
            workflow_name=workflow_name,
            node_id=node_id,
            duration_sec=_number(payload["duration_sec"]),
            input_tokens=_optional_int(payload.get("input_tokens")),
            output_tokens=_optional_int(payload.get("output_tokens")),
            admitted_batch_size=_optional_int(payload.get("admitted_batch_size")),
            model_key=model_key,
            gpu_kind=gpu_kind,
            prediction_key=prediction_key,
            predicted_run_sec=_optional_number(payload.get("predicted_run_sec")),
            predicted_load_sec=_optional_number(payload.get("predicted_load_sec")),
        )
    if requested != granted or requested != finished_acquires:
        raise ValueError(f"acquire event identities are unmatched: {path}")

    loads = _load_samples(events, model_names, path)
    eviction_durations = _eviction_durations(events, path)
    return TraceRun(
        path=path,
        run_id=run_id,
        run_started=run_starts[0],
        run_finished=run_finishes[0],
        session_arrivals=session_arrivals,
        session_workflows=session_workflows,
        tasks=tasks,
        loads=loads,
        eviction_durations=eviction_durations,
        model_names=model_names,
        completed_session_count=len(terminal_events),
    )


def discover_complete_trace_paths(root: Path) -> tuple[Path, ...]:
    return tuple(
        path
        for path in sorted(root.rglob("workflow_trace.jsonl"))
        if any(
            json.loads(line).get("event_type") == "run_finished"
            for line in path.open(encoding="utf-8")
        )
    )


def default_evaluation_cohorts(root: Path) -> tuple[CohortDefinition, ...]:
    definitions = (
        ("6w_burst_a1v2", root / "6w", "_a1v2", {"a100": 1, "v100": 2}, 4),
        (
            "6w_poisson_a1v2",
            root / "6w_poisson",
            "_a1v2",
            {"a100": 1, "v100": 2},
            4,
        ),
        ("6w_burst_a1v1", root / "6w", "_a1v1", {"a100": 1, "v100": 1}, 4),
    )
    cohorts = []
    for name, directory, suffix, slots, expected in definitions:
        paths = tuple(
            path
            for path in discover_complete_trace_paths(directory)
            if path.parent.name.endswith(suffix)
        )
        if len(paths) != expected:
            raise ValueError(
                f"{name} has {len(paths)} complete traces, expected {expected}"
            )
        cohorts.append(CohortDefinition(name=name, trace_paths=paths, gpu_slots=slots))
    return tuple(cohorts)


def load_workflows_for_runs(
    runs: Sequence[TraceRun],
    workflow_root: Path,
) -> dict[str, Workflow]:
    names = sorted(
        {
            workflow_name
            for run in runs
            for workflow_name in run.session_workflows.values()
        }
    )
    workflows = {}
    for name in names:
        path = workflow_root / f"{name}.yaml"
        if not path.is_file():
            raise FileNotFoundError(path)
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        workflow = Workflow.model_validate(raw)
        if workflow.workflow_name != name:
            raise ValueError(f"workflow name does not match filename: {path}")
        workflows[name] = workflow
    return workflows


def calibrate_prediction_cache(
    cache: ResourceContractCache,
    runs: Sequence[TraceRun],
) -> CalibrationResult:
    exact_ratios: dict[WorkflowModelFeatureKey, list[float]] = defaultdict(list)
    model_batch_ratios: dict[tuple[str, str, int], list[float]] = defaultdict(list)
    model_ratios: dict[tuple[str, str], list[float]] = defaultdict(list)
    warm_loads: dict[tuple[str, str], list[float]] = defaultdict(list)
    all_loads: dict[tuple[str, str], list[float]] = defaultdict(list)
    for run in runs:
        for task in run.tasks.values():
            key = task.prediction_key
            predicted = task.predicted_run_sec
            if key is None or predicted is None:
                continue
            if task.duration_sec <= 0 or predicted <= 0:
                raise ValueError(f"invalid task timing in {run.path}")
            ratio = task.duration_sec / predicted
            exact_ratios[key].append(ratio)
            model_batch_ratios[(key.model_name, key.gpu_name, key.batch_size)].append(
                ratio
            )
            model_ratios[(key.model_name, key.gpu_name)].append(ratio)
        for load in run.loads:
            identity = (load.model_name, load.gpu_kind)
            all_loads[identity].append(load.duration_sec)
            if not load.cold_accelerator:
                warm_loads[identity].append(load.duration_sec)

    records = []
    entries = []
    for entry in cache.entries:
        ratio, run_tier, run_count = _calibration_ratio(
            entry.key,
            exact_ratios,
            model_batch_ratios,
            model_ratios,
        )
        load_sec, load_tier, load_count = _calibrated_load_sec(
            entry,
            warm_loads,
            all_loads,
        )
        run_sec = entry.predicted_run_sec * ratio
        run_upper = entry.run_sec_upper_bound
        if run_upper is not None:
            run_upper = max(run_upper, run_sec)
        metadata = dict(entry.predictor_metadata)
        metadata["trace_calibration"] = {
            "run_tier": run_tier,
            "run_sample_count": run_count,
            "load_tier": load_tier,
            "load_sample_count": load_count,
        }
        calibrated = entry.model_copy(
            update={
                "predicted_run_sec": run_sec,
                "predicted_load_sec": load_sec,
                "run_sec_upper_bound": run_upper,
                "predictor_metadata": metadata,
            }
        )
        entries.append(calibrated)
        records.append(
            CalibrationRecord(
                prediction_key=entry.key,
                run_tier=run_tier,
                run_sample_count=run_count,
                load_tier=load_tier,
                load_sample_count=load_count,
                original_run_sec=entry.predicted_run_sec,
                calibrated_run_sec=run_sec,
                original_load_sec=entry.predicted_load_sec,
                calibrated_load_sec=load_sec,
            )
        )
    return CalibrationResult(
        cache=ResourceContractCache(
            version=cache.version,
            environment={
                **cache.environment,
                "soft_timing_source": "trace_calibrated",
            },
            entries=tuple(entries),
        ),
        records=tuple(records),
    )


def build_replay_workload(
    definition: CohortDefinition,
    runs: Sequence[TraceRun],
    workflows: Mapping[str, Workflow],
) -> ReplayWorkload:
    if not runs:
        raise ValueError("replay cohort requires at least one trace")
    session_ids = set(runs[0].session_arrivals)
    for run in runs[1:]:
        if set(run.session_arrivals) != session_ids:
            raise ValueError(f"cohort sessions are not aligned: {definition.name}")
        if set(run.tasks) != set(runs[0].tasks):
            raise ValueError(f"cohort tasks are not aligned: {definition.name}")

    raw_arrivals = {
        session_id: statistics.median(
            run.session_arrivals[session_id] - run.run_started for run in runs
        )
        for session_id in sorted(session_ids)
    }
    first_arrival = min(raw_arrivals.values())
    arrivals = {
        session_id: arrival - first_arrival
        for session_id, arrival in raw_arrivals.items()
    }
    session_workflows = {}
    for session_id in sorted(session_ids):
        names = {run.session_workflows[session_id] for run in runs}
        if len(names) != 1:
            raise ValueError(f"session workflow mismatch: {session_id}")
        session_workflows[session_id] = names.pop()

    task_specs: dict[tuple[str, str], TaskSpec] = {}
    exact_durations: dict[tuple[str, str, int], list[float]] = defaultdict(list)
    node_batch_durations: dict[tuple[str, str, int], list[float]] = defaultdict(list)
    model_batch_durations: dict[tuple[str, str, int], list[float]] = defaultdict(list)
    task_any_durations: dict[tuple[str, str], list[float]] = defaultdict(list)
    for identity in sorted(runs[0].tasks):
        samples = [run.tasks[identity] for run in runs]
        agent_flags = {sample.is_agent for sample in samples}
        if len(agent_flags) != 1:
            raise ValueError(f"task kind mismatch: {identity}")
        session_id, node_id = identity
        workflow_name = session_workflows[session_id]
        if node_id not in workflows[workflow_name].node_map():
            raise ValueError(f"trace task is absent from workflow: {identity}")
        if samples[0].is_agent:
            input_tokens = _median_required_int(
                [sample.input_tokens for sample in samples], "input_tokens", identity
            )
            output_tokens = _median_required_int(
                [sample.output_tokens for sample in samples], "output_tokens", identity
            )
            task_specs[identity] = TaskSpec(
                session_id=session_id,
                workflow_name=workflow_name,
                node_id=node_id,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                function_duration_sec=None,
            )
            for sample in samples:
                batch_size = sample.admitted_batch_size
                model_key = sample.model_key
                gpu_kind = sample.gpu_kind
                if batch_size is None or model_key is None or gpu_kind is None:
                    raise ValueError(
                        f"agent duration metadata is incomplete: {identity}"
                    )
                exact_durations[(session_id, node_id, batch_size)].append(
                    sample.duration_sec
                )
                node_batch_durations[(workflow_name, node_id, batch_size)].append(
                    sample.duration_sec
                )
                model_batch_durations[(model_key, gpu_kind, batch_size)].append(
                    sample.duration_sec
                )
                task_any_durations[identity].append(sample.duration_sec)
        else:
            task_specs[identity] = TaskSpec(
                session_id=session_id,
                workflow_name=workflow_name,
                node_id=node_id,
                input_tokens=None,
                output_tokens=None,
                function_duration_sec=statistics.median(
                    sample.duration_sec for sample in samples
                ),
            )

    cold_loads: dict[tuple[str, str], list[float]] = defaultdict(list)
    warm_loads: dict[tuple[str, str], list[float]] = defaultdict(list)
    all_loads: dict[tuple[str, str], list[float]] = defaultdict(list)
    evictions: dict[str, list[float]] = defaultdict(list)
    for run in runs:
        for load in run.loads:
            key = (load.model_name, load.gpu_kind)
            all_loads[key].append(load.duration_sec)
            target = cold_loads if load.cold_accelerator else warm_loads
            target[key].append(load.duration_sec)
        for gpu_kind, values in run.eviction_durations.items():
            evictions[gpu_kind].extend(values)

    return ReplayWorkload(
        name=definition.name,
        workflows=dict(workflows),
        session_arrivals=arrivals,
        session_workflows=session_workflows,
        task_specs=task_specs,
        agent_durations=AgentDurationTable(
            exact=_median_mapping(exact_durations),
            node_batch=_median_mapping(node_batch_durations),
            model_batch=_median_mapping(model_batch_durations),
            task_any_batch=_median_mapping(task_any_durations),
        ),
        cold_load_sec=_median_mapping(cold_loads),
        warm_load_sec=_median_mapping(warm_loads),
        any_load_sec=_median_mapping(all_loads),
        eviction_sec={
            gpu_kind: statistics.median(values)
            for gpu_kind, values in evictions.items()
        },
        gpu_slots=dict(definition.gpu_slots),
    )


class ReplaySimulator:
    def __init__(
        self,
        workload: ReplayWorkload,
        *,
        policy: PolicyName,
        profile_cache: ResourceContractCache,
        calibrated_cache: ResourceContractCache,
    ) -> None:
        self.workload = workload
        self.policy = policy
        predictions = (
            calibrated_cache
            if policy in ("trace_calibrated_cache", "trace_calibrated_cache_reclaim")
            else profile_cache
        )
        scheduler_policy = SCHEDULER_POLICY_ARMS.get(policy, "cache")
        accelerators = tuple(
            AcceleratorConfig(
                hostname=f"sim-{gpu_kind}-{index}",
                gpu_kind=gpu_kind,
                local_index=0,
                total_mem_mb=GPU_MEMORY_MB[gpu_kind],
            )
            for gpu_kind, count in sorted(workload.gpu_slots.items())
            for index in range(count)
        )
        workflows = sorted(
            workload.workflows.values(), key=lambda workflow: workflow.workflow_name
        )
        if not workflows:
            raise ValueError("replay workload has no workflows")
        self.core = SchedulerCore(
            workflows[0],
            scheduler_config=SchedulerConfig(
                policy=scheduler_policy,
                accelerators=accelerators,
            ),
            predictions=cast(
                ResourceContractCache,
                _IndexedPredictionCache(predictions),
            ),
        )
        for workflow in workflows[1:]:
            self.core.register_workflow(workflow)

        self.events: list[_Event] = []
        self.event_sequence = 0
        self.now = 0.0
        self.arrivals_remaining = len(workload.session_arrivals)
        self.session_completions: dict[str, float] = {}
        self.waiting_nodes: dict[tuple[str, str], deque[str]] = defaultdict(deque)
        self.enqueued_nodes: set[tuple[str, str]] = set()
        self.active_node_counts: dict[tuple[str, str], int] = defaultdict(int)
        self.active_functions: dict[str, _ActiveFunction] = {}
        self.active_agents: dict[str, _ActiveAgent] = {}
        self.acquire_requested_at: dict[str, float] = {}
        self.scheduled_grants: set[str] = set()
        self.scheduled_loads: set[str] = set()
        self.scheduled_evictions: set[str] = set()
        self.replica_timelines: dict[str, _ReplicaTimeline] = {}
        self.accelerator_warm: set[str] = set()
        self.next_policy_tick: float | None = None
        self.grant_count = 0
        self.reclaimed_replica_count = 0
        self.duration_fallback_counts: dict[str, int] = defaultdict(int)
        self.load_fallback_counts: dict[str, int] = defaultdict(int)

        for session_id, arrival in sorted(
            workload.session_arrivals.items(), key=lambda item: (item[1], item[0])
        ):
            self._push(arrival, "arrival", session_id)

    def run(self) -> ReplayMetrics:
        while len(self.session_completions) < len(self.workload.session_arrivals):
            if not self.events:
                pending = {
                    state.value
                    for state in (task.state for task in self.core.tasks.values())
                }
                raise RuntimeError(
                    f"replay deadlocked with task states {sorted(pending)}"
                )
            event = heapq.heappop(self.events)
            self.now = event.at
            batch = [event]
            while self.events and self.events[0].at == self.now:
                batch.append(heapq.heappop(self.events))
            for current in batch:
                self._handle_event(current)
            self._start_waiting_nodes()
            self._schedule_core()
            self._start_waiting_nodes()

        business_finished = max(self.session_completions.values())
        return self._metrics(business_finished)

    def _handle_event(self, event: _Event) -> None:
        if event.kind == "arrival":
            self._handle_arrival(_expect_str(event.payload))
        elif event.kind == "function_done":
            self._finish_function(_expect_str(event.payload))
        elif event.kind == "agent_done":
            self._finish_agent(_expect_str(event.payload))
        elif event.kind == "load_done":
            self._finish_load(_expect_str(event.payload))
        elif event.kind == "evict_done":
            self._finish_eviction(_expect_str(event.payload))
        elif event.kind == "policy_tick":
            if self.next_policy_tick == event.at:
                self.next_policy_tick = None
        else:
            raise ValueError(f"unknown replay event: {event.kind}")

    def _handle_arrival(self, session_id: str) -> None:
        workflow_name = self.workload.session_workflows[session_id]
        self.core.register_session(session_id, workflow_name)
        self.arrivals_remaining -= 1
        entry = self.workload.workflows[workflow_name].graph.entry_node
        self._enqueue_node(session_id, workflow_name, entry)

    def _enqueue_node(
        self,
        session_id: str,
        workflow_name: str,
        node_id: str,
    ) -> None:
        identity = (session_id, node_id)
        if identity in self.enqueued_nodes:
            raise ValueError(f"node enqueued twice: {identity}")
        self.enqueued_nodes.add(identity)
        self.waiting_nodes[(workflow_name, node_id)].append(session_id)

    def _start_waiting_nodes(self) -> None:
        started = True
        while started:
            started = False
            for workflow_name, workflow in sorted(self.workload.workflows.items()):
                nodes = workflow.node_map()
                for node_id in workflow.graph.topological_order:
                    queue = self.waiting_nodes[(workflow_name, node_id)]
                    node = nodes[node_id]
                    limit = (
                        node.execution.serving.max_num_seqs
                        if isinstance(node, AgentNodeConfig)
                        else node.max_concurrency
                    )
                    key = (workflow_name, node_id)
                    while queue and self.active_node_counts[key] < limit:
                        session_id = queue.popleft()
                        self._start_node(session_id, workflow_name, node_id)
                        started = True

    def _start_node(
        self,
        session_id: str,
        workflow_name: str,
        node_id: str,
    ) -> None:
        workflow = self.workload.workflows[workflow_name]
        node = workflow.node_map()[node_id]
        input_ids = [
            f"{session_id}:{dependency}"
            for dependency in workflow.graph.dependencies[node_id]
        ] or [f"{session_id}:entry"]
        task_id = self.core.begin_node(session_id, node_id, input_ids)
        self.active_node_counts[(workflow_name, node_id)] += 1
        spec = self.workload.task_specs[(session_id, node_id)]
        if isinstance(node, FunctionNodeConfig):
            duration = spec.function_duration_sec
            if duration is None:
                raise ValueError(
                    f"function duration is missing: {session_id}/{node_id}"
                )
            self.active_functions[task_id] = _ActiveFunction(
                task_id=task_id,
                session_id=session_id,
                workflow_name=workflow_name,
                node_id=node_id,
                started_at=self.now,
            )
            self._push(self.now + duration, "function_done", task_id)
            return

        input_tokens = spec.input_tokens
        if input_tokens is None:
            raise ValueError(f"agent input tokens are missing: {session_id}/{node_id}")
        acquire_id = self.core.request_acquire(
            task_id,
            input_tokens=input_tokens,
            created_at=self.now,
        )
        self.acquire_requested_at[acquire_id] = self.now

    def _finish_function(self, task_id: str) -> None:
        active = self.active_functions.pop(task_id)
        duration = self.now - active.started_at
        decision = self.core.complete(
            task_id,
            FunctionTaskRuntimeReport(
                task_id=task_id,
                session_id=active.session_id,
                node_id=active.node_id,
                input_item_ids=list(self.core.tasks[task_id].input_item_ids),
                started_at=active.started_at,
                finished_at=self.now,
                duration_sec=duration,
                status="success",
            ),
        )
        if not decision.emit_output:
            raise RuntimeError("successful function did not emit output")
        self._finish_node(
            task_id,
            active.session_id,
            active.workflow_name,
            active.node_id,
        )

    def _finish_agent(self, acquire_id: str) -> None:
        active = self.active_agents.pop(acquire_id)
        grant = active.grant
        duration = self.now - active.started_at
        node = self.workload.workflows[active.workflow_name].node_map()[
            self.core.tasks[grant.task_id].node_id
        ]
        if not isinstance(node, AgentNodeConfig):
            raise RuntimeError("agent grant belongs to a function node")
        report = AgentTaskRuntimeReport(
            acquire_id=acquire_id,
            task_id=grant.task_id,
            session_id=self.core.tasks[grant.task_id].session_id,
            node_id=self.core.tasks[grant.task_id].node_id,
            input_item_ids=list(self.core.tasks[grant.task_id].input_item_ids),
            model_key=grant.model_key,
            accelerator_id=grant.accelerator_ids[0],
            gpu_kind=grant.gpu_kind,
            input_tokens=grant.input_tokens,
            max_new_tokens=grant.max_new_tokens,
            output_tokens=min(active.output_tokens, grant.max_new_tokens),
            hit_token_limit=active.output_tokens >= grant.max_new_tokens,
            finish_reason=(
                "length" if active.output_tokens >= grant.max_new_tokens else "stop"
            ),
            replica_inflight_at_start=grant.admitted_batch_size,
            started_at=active.started_at,
            finished_at=self.now,
            duration_sec=duration,
            status="success",
        )
        decision = self.core.complete(
            grant.task_id,
            report,
            acquire_id=acquire_id,
        )
        if not decision.emit_output or decision.retry_acquire:
            raise RuntimeError("successful agent completion was not terminal")
        task = self.core.tasks[grant.task_id]
        self._finish_node(
            grant.task_id,
            task.session_id,
            active.workflow_name,
            task.node_id,
        )

    def _finish_node(
        self,
        task_id: str,
        session_id: str,
        workflow_name: str,
        node_id: str,
    ) -> None:
        workflow = self.workload.workflows[workflow_name]
        terminal = node_id == workflow.graph.terminal_node
        self.core.finish_node(
            task_id,
            OutputReport(
                task_id=task_id,
                output_item_ids=[],
                persisted_terminal_result=terminal,
            ),
        )
        self.active_node_counts[(workflow_name, node_id)] -= 1
        if terminal:
            self.session_completions[session_id] = self.now
            return
        for successor in workflow.graph.adjacency[node_id]:
            task = self.core.tasks[self.core.sessions[session_id].task_ids[successor]]
            if task.state != NodeTaskState.PENDING:
                continue
            dependencies = workflow.graph.dependencies[successor]
            if all(
                self.core.tasks[
                    self.core.sessions[session_id].task_ids[dependency]
                ].state
                == NodeTaskState.COMPLETED
                for dependency in dependencies
            ):
                self._enqueue_node(session_id, workflow_name, successor)

    def _schedule_core(self) -> None:
        actions = self.core.tick_once(now=self.now)
        for action in actions:
            if isinstance(action, LoadReplicaAction):
                self._start_load(action)
            else:
                self._start_eviction(action)
        self._schedule_new_grants()
        if (
            self.policy == "trace_calibrated_cache_reclaim"
            and self.arrivals_remaining == 0
        ):
            self._schedule_reclaims()
        self._schedule_prefetch_tick()

    def _start_load(self, action: LoadReplicaAction) -> None:
        if action.replica_id in self.scheduled_loads:
            raise ValueError(f"replica load scheduled twice: {action.replica_id}")
        self.scheduled_loads.add(action.replica_id)
        cold = action.accelerator.accelerator_id not in self.accelerator_warm
        duration = self._load_duration(
            action.deployment.model_name,
            action.accelerator.gpu_kind,
            cold=cold,
        )
        self.replica_timelines[action.replica_id] = _ReplicaTimeline(
            model_key=action.deployment.model_key,
            model_name=action.deployment.model_name,
            gpu_kind=action.accelerator.gpu_kind,
            load_reason=action.reason,
            load_started=self.now,
        )
        self._push(self.now + duration, "load_done", action.replica_id)

    def _finish_load(self, replica_id: str) -> None:
        replica = self.core.replicas[replica_id]
        accelerator = self.core.accelerators[replica.accelerator_ids[0]].config
        timeline = self.replica_timelines[replica_id]
        timeline.load_finished = self.now
        self.accelerator_warm.add(accelerator.accelerator_id)
        self.core.complete_load(
            replica_id,
            ReplicaLoadResult(
                physical_gpu_id=accelerator.local_index,
                duration_sec=self.now - timeline.load_started,
                idle_vram_mb=1.0,
                max_num_seqs=replica.deployment.serving.max_num_seqs,
                block_size=16,
                num_gpu_blocks=1,
                gpu_kv_tokens=16,
            ),
            backend_handle=replica_id,
            now=self.now,
        )

    def _start_eviction(self, action: EvictReplicaAction) -> None:
        if action.replica_id in self.scheduled_evictions:
            raise ValueError(f"replica eviction scheduled twice: {action.replica_id}")
        self.scheduled_evictions.add(action.replica_id)
        timeline = self.replica_timelines[action.replica_id]
        timeline.eviction_started = self.now
        duration = self.workload.eviction_sec.get(timeline.gpu_kind)
        if duration is None:
            raise ValueError(
                f"eviction duration is missing for GPU {timeline.gpu_kind}"
            )
        self._push(self.now + duration, "evict_done", action.replica_id)

    def _finish_eviction(self, replica_id: str) -> None:
        self.replica_timelines[replica_id].eviction_finished = self.now
        self.core.complete_eviction(replica_id)

    def _schedule_new_grants(self) -> None:
        for acquire_id, grant in tuple(self.core.grants.items()):
            if acquire_id in self.scheduled_grants:
                continue
            self.scheduled_grants.add(acquire_id)
            self.grant_count += 1
            task = self.core.tasks[grant.task_id]
            workflow_name = self.core.sessions[task.session_id].workflow_name
            spec = self.workload.task_specs[(task.session_id, task.node_id)]
            output_tokens = spec.output_tokens
            if output_tokens is None:
                raise ValueError(f"agent output tokens are missing: {task.task_id}")
            duration = self._agent_duration(
                task.session_id,
                workflow_name,
                task.node_id,
                grant,
            )
            timeline = self.replica_timelines[grant.replica_id]
            timeline.executions.append((self.now, self.now + duration))
            if timeline.first_grant_at is None:
                timeline.first_grant_at = self.now
                timeline.first_acquire_at = self.acquire_requested_at[acquire_id]
            self.active_agents[acquire_id] = _ActiveAgent(
                grant=grant,
                workflow_name=workflow_name,
                started_at=self.now,
                output_tokens=output_tokens,
            )
            self._push(self.now + duration, "agent_done", acquire_id)

    def _schedule_reclaims(self) -> None:
        for replica in sorted(
            tuple(self.core.replicas.values()), key=lambda value: value.replica_id
        ):
            if replica.state != ModelReplicaState.IDLE:
                continue
            if self._has_unfinished_demand(replica.model_key):
                continue
            action = self.core.request_eviction(replica.replica_id)
            self.reclaimed_replica_count += 1
            self._start_eviction(action)

    def _has_unfinished_demand(self, model_key: str) -> bool:
        for session in self.core.sessions.values():
            if session.state.value != "active":
                continue
            workflow = self.workload.workflows[session.workflow_name]
            nodes = workflow.node_map()
            for node_id, task_id in session.task_ids.items():
                task = self.core.tasks[task_id]
                node = nodes[node_id]
                if (
                    isinstance(node, AgentNodeConfig)
                    and task.state != NodeTaskState.COMPLETED
                    and ModelDeploymentConfig.from_node(node).model_key == model_key
                ):
                    return True
        return False

    def _schedule_prefetch_tick(self) -> None:
        future = [
            candidate.prefetch_at
            for candidate in self.core.near_ready_tasks()
            if candidate.prefetch_at > self.now
        ]
        if not future:
            return
        wake_at = min(future)
        if self.next_policy_tick is not None and self.next_policy_tick <= wake_at:
            return
        self.next_policy_tick = wake_at
        self._push(wake_at, "policy_tick", None)

    def _agent_duration(
        self,
        session_id: str,
        workflow_name: str,
        node_id: str,
        grant: GrantInfo,
    ) -> float:
        batch_size = grant.admitted_batch_size
        table = self.workload.agent_durations
        exact = table.exact.get((session_id, node_id, batch_size))
        if exact is not None:
            self.duration_fallback_counts["exact"] += 1
            return exact
        node_value = table.node_batch.get((workflow_name, node_id, batch_size))
        if node_value is not None:
            self.duration_fallback_counts["node_batch"] += 1
            return node_value
        model_value = table.model_batch.get(
            (grant.model_key, grant.gpu_kind, batch_size)
        )
        if model_value is not None:
            self.duration_fallback_counts["model_batch"] += 1
            return model_value
        task_value = table.task_any_batch.get((session_id, node_id))
        if task_value is None:
            raise ValueError(
                "agent duration coverage is missing: "
                f"{session_id}/{node_id}/b{batch_size}"
            )
        self.duration_fallback_counts["task_any_batch"] += 1
        return task_value

    def _load_duration(self, model_name: str, gpu_kind: str, *, cold: bool) -> float:
        identity = (model_name, gpu_kind)
        primary = self.workload.cold_load_sec if cold else self.workload.warm_load_sec
        value = primary.get(identity)
        if value is not None:
            self.load_fallback_counts["cold" if cold else "warm"] += 1
            return value
        value = self.workload.any_load_sec.get(identity)
        if value is None:
            raise ValueError(f"load duration coverage is missing: {identity}")
        self.load_fallback_counts["any"] += 1
        return value

    def _metrics(self, business_finished: float) -> ReplayMetrics:
        active_gpu_seconds = 0.0
        loading_gpu_seconds = 0.0
        resident_gpu_seconds = 0.0
        idle_resident_gpu_seconds = 0.0
        evicting_gpu_seconds = 0.0
        prefetch_count = 0
        prefetch_ready = 0
        prefetch_overrun = 0.0
        eviction_count = 0
        used_load_count = 0
        for timeline in self.replica_timelines.values():
            load_end = min(
                timeline.load_finished or business_finished,
                business_finished,
            )
            loading_gpu_seconds += max(
                0.0, load_end - min(timeline.load_started, business_finished)
            )
            if (
                timeline.load_finished is None
                or timeline.load_finished > business_finished
            ):
                continue
            resident_end = min(
                timeline.eviction_started or business_finished, business_finished
            )
            resident = max(0.0, resident_end - timeline.load_finished)
            active = interval_union_duration(
                _clip_intervals(
                    timeline.executions,
                    timeline.load_finished,
                    resident_end,
                )
            )
            active_gpu_seconds += active
            resident_gpu_seconds += resident
            idle_resident_gpu_seconds += max(0.0, resident - active)
            if timeline.eviction_started is not None:
                eviction_count += 1
                eviction_end = min(
                    timeline.eviction_finished or business_finished,
                    business_finished,
                )
                evicting_gpu_seconds += max(
                    0.0, eviction_end - timeline.eviction_started
                )
            if timeline.first_grant_at is not None:
                used_load_count += 1
            if timeline.load_reason == "near_ready_prefetch":
                prefetch_count += 1
                if (
                    timeline.first_acquire_at is not None
                    and timeline.load_finished <= timeline.first_acquire_at
                ):
                    prefetch_ready += 1
                elif timeline.first_acquire_at is not None:
                    prefetch_overrun += (
                        timeline.load_finished - timeline.first_acquire_at
                    )

        latencies = [
            completion - self.workload.session_arrivals[session_id]
            for session_id, completion in self.session_completions.items()
        ]
        family_completion: dict[str, float] = {}
        for session_id, completion in self.session_completions.items():
            workflow_name = self.workload.session_workflows[session_id]
            family = re.sub(r"\d+$", "", workflow_name)
            family_completion[family] = max(
                family_completion.get(family, 0.0), completion
            )
        non_active = (
            loading_gpu_seconds
            + resident_gpu_seconds
            + evicting_gpu_seconds
            - active_gpu_seconds
        )
        return ReplayMetrics(
            policy=self.policy,
            replay_completion_span_sec=business_finished,
            mean_session_latency_sec=statistics.fmean(latencies),
            completion_by_family_sec=family_completion,
            loading_gpu_seconds=loading_gpu_seconds,
            resident_gpu_seconds=resident_gpu_seconds,
            idle_resident_gpu_seconds=idle_resident_gpu_seconds,
            evicting_gpu_seconds=evicting_gpu_seconds,
            non_active_lifecycle_gpu_seconds=non_active,
            pipeline_bubble_ratio=(
                idle_resident_gpu_seconds / resident_gpu_seconds
                if resident_gpu_seconds
                else 0.0
            ),
            model_load_count=len(self.replica_timelines),
            model_reuse_count=max(0, self.grant_count - used_load_count),
            model_eviction_count=eviction_count,
            prefetch_count=prefetch_count,
            prefetch_ready_before_acquire_count=prefetch_ready,
            prefetch_overrun_seconds=prefetch_overrun,
            reclaimed_replica_count=self.reclaimed_replica_count,
            duration_fallback_counts=dict(self.duration_fallback_counts),
            load_fallback_counts=dict(self.load_fallback_counts),
        )

    def _push(self, at: float, kind: str, payload: object) -> None:
        if at < self.now:
            raise ValueError(f"cannot schedule {kind} in the past")
        heapq.heappush(
            self.events,
            _Event(at=at, sequence=self.event_sequence, kind=kind, payload=payload),
        )
        self.event_sequence += 1


def run_cache_replay_experiment(
    *,
    calibration_root: Path,
    evaluation_root: Path,
    prediction_cache_path: Path,
    workflow_root: Path,
) -> dict[str, object]:
    profile_cache = load_resource_contract_cache(prediction_cache_path)
    calibration_paths = discover_complete_trace_paths(calibration_root)
    if not calibration_paths:
        raise ValueError(f"no complete calibration traces: {calibration_root}")
    calibration_runs = tuple(
        read_trace_run(path, expected_sessions=120) for path in calibration_paths
    )
    calibration = calibrate_prediction_cache(profile_cache, calibration_runs)

    cohort_rows = []
    evaluation_runs: dict[Path, TraceRun] = {}
    definitions = default_evaluation_cohorts(evaluation_root)
    for definition in definitions:
        runs = tuple(
            evaluation_runs.setdefault(
                path,
                read_trace_run(path, expected_sessions=120),
            )
            for path in definition.trace_paths
        )
        workflows = load_workflows_for_runs(runs, workflow_root)
        workload = build_replay_workload(definition, runs, workflows)
        metrics = {
            policy: ReplaySimulator(
                workload,
                policy=policy,
                profile_cache=profile_cache,
                calibrated_cache=calibration.cache,
            ).run()
            for policy in POLICIES
        }
        cohort_rows.append(
            {
                "cohort": definition.name,
                "source_traces": [str(path) for path in definition.trace_paths],
                "gpu_slots": dict(definition.gpu_slots),
                "strategies": {
                    policy: asdict(value) for policy, value in metrics.items()
                },
                "comparison": _cohort_comparison(metrics),
            }
        )

    evaluation_unique_runs = tuple(
        evaluation_runs[path] for path in sorted(evaluation_runs)
    )
    prediction_error = _prediction_error_summary(
        evaluation_unique_runs,
        profile_cache,
        calibration.cache,
    )
    acceptance = _acceptance_summary(cohort_rows, prediction_error)
    complete_evaluation_paths = set(evaluation_runs)
    all_evaluation_paths = set(evaluation_root.rglob("workflow_trace.jsonl"))
    excluded = sorted(all_evaluation_paths - complete_evaluation_paths)
    return {
        "manifest": {
            "version": 1,
            "calibration_root": str(calibration_root),
            "evaluation_root": str(evaluation_root),
            "prediction_cache": str(prediction_cache_path),
            "prediction_cache_sha256": file_sha256(prediction_cache_path),
            "workflow_root": str(workflow_root),
            "calibration_traces": _path_hashes(calibration_paths),
            "evaluation_traces": _path_hashes(
                path for definition in definitions for path in definition.trace_paths
            ),
            "excluded_evaluation_traces": [str(path) for path in excluded],
            "queue_model": "unbounded",
            "result_semantics": "trace_derived_counterfactual_replay",
        },
        "calibration": {
            "entry_count": len(calibration.records),
            "records": [
                _calibration_record_payload(record) for record in calibration.records
            ],
            "held_out_prediction_error": prediction_error,
        },
        "cohorts": cohort_rows,
        "summary": {
            "acceptance": acceptance,
            "claim_boundary": (
                "Replay completion span is a trace-derived counterfactual proxy, "
                "not measured system makespan."
            ),
        },
    }


def write_cache_replay_outputs(
    output_dir: Path,
    result: Mapping[str, object],
) -> None:
    if output_dir.exists():
        raise FileExistsError(output_dir)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent)
    )
    try:
        _write_json(temporary / "manifest.json", result["manifest"])
        _write_json(temporary / "calibration.json", result["calibration"])
        cohorts = _expect_sequence(result["cohorts"], "cohorts")
        (temporary / "cohort_results.jsonl").write_text(
            "".join(f"{canonical_json(row)}\n" for row in cohorts),
            encoding="utf-8",
        )
        _write_json(temporary / "summary.json", result["summary"])
        (temporary / "report.md").write_text(
            render_cache_replay_report(result),
            encoding="utf-8",
        )
        temporary.rename(output_dir)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def render_cache_replay_report(result: Mapping[str, object]) -> str:
    cohorts = _expect_sequence(result["cohorts"], "cohorts")
    summary = _expect_mapping(result["summary"], "summary")
    acceptance = _expect_mapping(summary["acceptance"], "acceptance")
    lines = [
        "# Trace-Calibrated Cache 离线回放",
        "",
        (
            "本报告是 trace-derived counterfactual replay; "
            "completion span 不是实机 makespan。"
        ),
        "",
        "| Cohort | Strategy | Completion span (s) | Mean E2E (s) | "
        "Non-active lifecycle GPU-s | Bubble | Loads |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for cohort_value in cohorts:
        cohort = _expect_mapping(cohort_value, "cohort")
        strategies = _expect_mapping(cohort["strategies"], "strategies")
        for policy in POLICIES:
            metrics = _expect_mapping(strategies[policy], policy)
            lines.append(
                "| {cohort} | {policy} | {span:.3f} | {latency:.3f} | "
                "{non_active:.3f} | {bubble:.4%} | {loads} |".format(
                    cohort=cohort["cohort"],
                    policy=policy,
                    span=_number(metrics["replay_completion_span_sec"]),
                    latency=_number(metrics["mean_session_latency_sec"]),
                    non_active=_number(metrics["non_active_lifecycle_gpu_seconds"]),
                    bubble=_number(metrics["pipeline_bubble_ratio"]),
                    loads=_integer(metrics["model_load_count"]),
                )
            )
    lines.extend(
        [
            "",
            "## 验收",
            "",
            f"- 总体通过: {bool(acceptance['passed'])}",
            (f"- Completion span 门槛: {bool(acceptance['completion_span_ok'])}"),
            (
                "- Non-active lifecycle 降幅: "
                f"{_number(acceptance['non_active_reduction_fraction']):.2%}"
            ),
            (
                "- Bubble 改善 cohort 数: "
                f"{_integer(acceptance['bubble_improved_cohort_count'])}"
            ),
            (f"- 留出预测 MAE 改善: {bool(acceptance['prediction_error_ok'])}"),
            "",
            "只有所有门槛通过时, 结果才可描述为离线 Cache 优化成功。",
            "",
        ]
    )
    return "\n".join(lines)


def _cohort_comparison(
    metrics: Mapping[PolicyName, ReplayMetrics],
) -> dict[str, float]:
    baseline = metrics["profile_cache"]
    candidate = metrics["trace_calibrated_cache_reclaim"]
    return {
        "completion_span_change_fraction": _relative_change(
            candidate.replay_completion_span_sec,
            baseline.replay_completion_span_sec,
        ),
        "non_active_lifecycle_reduction_fraction": _relative_reduction(
            candidate.non_active_lifecycle_gpu_seconds,
            baseline.non_active_lifecycle_gpu_seconds,
        ),
        "bubble_delta": (
            candidate.pipeline_bubble_ratio - baseline.pipeline_bubble_ratio
        ),
    }


def _prediction_error_summary(
    runs: Sequence[TraceRun],
    profile_cache: ResourceContractCache,
    calibrated_cache: ResourceContractCache,
) -> dict[str, float | int]:
    profile_errors = []
    calibrated_errors = []
    profile_relative = []
    calibrated_relative = []
    for run in runs:
        for task in run.tasks.values():
            key = task.prediction_key
            if key is None:
                continue
            actual = task.duration_sec
            if actual <= 0:
                raise ValueError(f"non-positive evaluation duration: {run.path}")
            profile = profile_cache.lookup(key).predicted_run_sec
            calibrated = calibrated_cache.lookup(key).predicted_run_sec
            profile_errors.append(abs(profile - actual))
            calibrated_errors.append(abs(calibrated - actual))
            profile_relative.append(abs(profile - actual) / actual)
            calibrated_relative.append(abs(calibrated - actual) / actual)
    if not profile_errors:
        raise ValueError("evaluation traces contain no agent tasks")
    return {
        "sample_count": len(profile_errors),
        "profile_mean_absolute_error_sec": statistics.fmean(profile_errors),
        "calibrated_mean_absolute_error_sec": statistics.fmean(calibrated_errors),
        "profile_median_absolute_relative_error": statistics.median(profile_relative),
        "calibrated_median_absolute_relative_error": statistics.median(
            calibrated_relative
        ),
    }


def _acceptance_summary(
    cohort_rows: Sequence[Mapping[str, object]],
    prediction_error: Mapping[str, float | int],
) -> dict[str, object]:
    completion_changes = []
    bubble_improvements = 0
    baseline_non_active = 0.0
    candidate_non_active = 0.0
    for row in cohort_rows:
        comparison = _expect_mapping(row["comparison"], "comparison")
        completion_changes.append(
            _number(comparison["completion_span_change_fraction"])
        )
        if _number(comparison["bubble_delta"]) < 0:
            bubble_improvements += 1
        strategies = _expect_mapping(row["strategies"], "strategies")
        baseline = _expect_mapping(strategies["profile_cache"], "profile_cache")
        candidate = _expect_mapping(
            strategies["trace_calibrated_cache_reclaim"],
            "trace_calibrated_cache_reclaim",
        )
        baseline_non_active += _number(baseline["non_active_lifecycle_gpu_seconds"])
        candidate_non_active += _number(candidate["non_active_lifecycle_gpu_seconds"])
    completion_ok = all(change <= 0.01 for change in completion_changes)
    non_active_reduction = _relative_reduction(
        candidate_non_active, baseline_non_active
    )
    non_active_ok = non_active_reduction >= 0.05
    bubble_ok = bubble_improvements >= 2
    prediction_ok = _number(
        prediction_error["calibrated_mean_absolute_error_sec"]
    ) < _number(prediction_error["profile_mean_absolute_error_sec"])
    return {
        "passed": completion_ok and non_active_ok and bubble_ok and prediction_ok,
        "completion_span_ok": completion_ok,
        "maximum_completion_span_change_fraction": max(completion_changes),
        "non_active_lifecycle_ok": non_active_ok,
        "non_active_reduction_fraction": non_active_reduction,
        "bubble_ok": bubble_ok,
        "bubble_improved_cohort_count": bubble_improvements,
        "prediction_error_ok": prediction_ok,
    }


def _load_samples(
    events: Sequence[Mapping[str, Any]],
    model_names: Mapping[str, str],
    path: Path,
) -> tuple[LoadSample, ...]:
    seen_accelerators: set[str] = set()
    samples = []
    for event in events:
        if event.get("event_type") != "model_load_finished":
            continue
        model_key = _required_str(event, "model_key", path)
        gpu_kind = _required_str(event, "gpu_kind", path)
        accelerator_id = _required_str(event, "accelerator_id", path)
        model_name = model_names.get(model_key)
        if model_name is None:
            raise ValueError(f"loaded model has no task metadata {model_key}: {path}")
        samples.append(
            LoadSample(
                model_name=model_name,
                model_key=model_key,
                gpu_kind=gpu_kind,
                duration_sec=_number(_payload(event, path)["duration_sec"]),
                cold_accelerator=accelerator_id not in seen_accelerators,
            )
        )
        seen_accelerators.add(accelerator_id)
    return tuple(samples)


def _eviction_durations(
    events: Sequence[Mapping[str, Any]],
    path: Path,
) -> dict[str, tuple[float, ...]]:
    starts: dict[str, tuple[float, str]] = {}
    durations: dict[str, list[float]] = defaultdict(list)
    for event in events:
        event_type = event.get("event_type")
        if event_type not in ("model_eviction_started", "model_evicted"):
            continue
        replica_id = _required_str(event, "replica_id", path)
        if event_type == "model_eviction_started":
            if replica_id in starts:
                raise ValueError(f"duplicate eviction start {replica_id}: {path}")
            starts[replica_id] = (
                _number(event["ts"]),
                _required_str(event, "gpu_kind", path),
            )
            continue
        start = starts.pop(replica_id, None)
        if start is None:
            raise ValueError(f"eviction finish has no start {replica_id}: {path}")
        started_at, gpu_kind = start
        duration = _number(event["ts"]) - started_at
        if duration < 0:
            raise ValueError(f"eviction finish precedes start {replica_id}: {path}")
        durations[gpu_kind].append(duration)
    if starts:
        raise ValueError(f"trace has unfinished evictions: {path}")
    return {key: tuple(values) for key, values in durations.items()}


def _calibration_ratio(
    key: WorkflowModelFeatureKey,
    exact: Mapping[WorkflowModelFeatureKey, Sequence[float]],
    model_batch: Mapping[tuple[str, str, int], Sequence[float]],
    model: Mapping[tuple[str, str], Sequence[float]],
) -> tuple[float, str, int]:
    tiers = (
        ("exact_prediction_key", exact.get(key, ())),
        (
            "model_gpu_batch",
            model_batch.get((key.model_name, key.gpu_name, key.batch_size), ()),
        ),
        ("model_gpu", model.get((key.model_name, key.gpu_name), ())),
    )
    for tier, values in tiers:
        if len(values) >= MIN_CALIBRATION_SAMPLES:
            return statistics.median(values), tier, len(values)
    return 1.0, "profile_fallback", 0


def _calibrated_load_sec(
    entry: ResourceContract,
    warm_loads: Mapping[tuple[str, str], Sequence[float]],
    all_loads: Mapping[tuple[str, str], Sequence[float]],
) -> tuple[float, str, int]:
    identity = (entry.key.model_name, entry.key.gpu_name)
    warm = warm_loads.get(identity, ())
    if len(warm) >= MIN_CALIBRATION_SAMPLES:
        return statistics.median(warm), "warm_model_gpu", len(warm)
    all_values = all_loads.get(identity, ())
    if len(all_values) >= MIN_CALIBRATION_SAMPLES:
        return statistics.median(all_values), "all_load_model_gpu", len(all_values)
    return entry.predicted_load_sec, "profile_fallback", 0


def _calibration_record_payload(
    record: CalibrationRecord,
) -> dict[str, object]:
    return {
        "prediction_key": record.prediction_key.model_dump(mode="json"),
        "run_tier": record.run_tier,
        "run_sample_count": record.run_sample_count,
        "load_tier": record.load_tier,
        "load_sample_count": record.load_sample_count,
        "original_run_sec": record.original_run_sec,
        "calibrated_run_sec": record.calibrated_run_sec,
        "original_load_sec": record.original_load_sec,
        "calibrated_load_sec": record.calibrated_load_sec,
    }


def _event_timestamps(
    events: Sequence[Mapping[str, Any]],
    event_type: str,
) -> list[float]:
    return [
        _number(event["ts"])
        for event in events
        if event.get("event_type") == event_type
    ]


def _payload(event: Mapping[str, Any], path: Path) -> Mapping[str, Any]:
    payload = event.get("payload")
    if not isinstance(payload, Mapping):
        raise TypeError(f"trace payload must be a mapping: {path}")
    return payload


def _required_str(
    value: Mapping[str, Any],
    key: str,
    path: Path,
) -> str:
    result = value.get(key)
    if not isinstance(result, str) or not result:
        raise TypeError(f"trace {key} must be a non-empty string: {path}")
    return result


def _optional_str(value: object, key: str, path: Path) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise TypeError(f"trace {key} must be a non-empty string: {path}")
    return value


def _optional_int(value: object) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("trace integer field has the wrong type")
    return value


def _optional_number(value: object) -> float | None:
    return None if value is None else _number(value)


def _number(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"expected a number, got {type(value).__name__}")
    return float(value)


def _integer(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"expected an integer, got {type(value).__name__}")
    return value


def _expect_str(value: object) -> str:
    if not isinstance(value, str):
        raise TypeError(f"expected a string, got {type(value).__name__}")
    return value


def _expect_mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    if any(not isinstance(key, str) for key in value):
        raise TypeError(f"{name} keys must be strings")
    return cast(Mapping[str, object], value)


def _expect_sequence(value: object, name: str) -> Sequence[object]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise TypeError(f"{name} must be a sequence")
    return value


def _median_required_int(
    values: Sequence[int | None],
    name: str,
    identity: tuple[str, str],
) -> int:
    if any(value is None for value in values):
        raise ValueError(f"{name} is missing for {identity}")
    typed = [value for value in values if value is not None]
    return statistics.median_low(typed)


def _median_mapping[KeyT](
    values: Mapping[KeyT, Sequence[float]],
) -> dict[KeyT, float]:
    return {
        key: statistics.median(samples) for key, samples in values.items() if samples
    }


def _clip_intervals(
    intervals: Iterable[tuple[float, float]],
    start: float,
    end: float,
) -> list[tuple[float, float]]:
    return [
        (max(left, start), min(right, end))
        for left, right in intervals
        if min(right, end) > max(left, start)
    ]


def _relative_change(candidate: float, baseline: float) -> float:
    if baseline <= 0:
        raise ValueError("relative change requires a positive baseline")
    return (candidate - baseline) / baseline


def _relative_reduction(candidate: float, baseline: float) -> float:
    return -_relative_change(candidate, baseline)


def _path_hashes(paths: Iterable[Path]) -> list[dict[str, str]]:
    return [
        {"path": str(path), "sha256": file_sha256(path)} for path in sorted(set(paths))
    ]


def _write_json(path: Path, value: object) -> None:
    path.write_text(f"{canonical_json(value)}\n", encoding="utf-8")
