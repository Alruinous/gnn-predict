"""Build detailed evaluation drafts from the unified-load serve_0728 replay."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import matplotlib
import yaml
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import Rectangle

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from experiment.workflow.analysis import interval_union_duration  # noqa: E402
from experiment.workflow.cache_replay import (  # noqa: E402
    GPU_MEMORY_MB,
    CohortDefinition,
    PolicyName,
    ReplaySimulator,
    ReplayWorkload,
    TraceRun,
    _IndexedPredictionCache,
    build_replay_workload,
    read_trace_run,
)
from workflow.artifacts import (  # noqa: E402
    AcceleratorConfig,
    ResourceContractCache,
    SchedulerConfig,
    load_resource_contract_cache,
)
from workflow.fusion import fuse_workflow  # noqa: E402
from workflow.replica import ModelDeploymentConfig  # noqa: E402
from workflow.scheduler import SchedulerCore  # noqa: E402
from workflow.schema import AgentNodeConfig, Workflow  # noqa: E402

OUTPUT_ROOT = ROOT / "output" / "serve_0728_evaluation_drafts"
METRICS_PATH = OUTPUT_ROOT / "analysis_metrics.json"
SERVE_ROOT = ROOT / "output" / "serve_0728"

HARDWARE = ("a1v2", "a1v3")
ARRIVALS = ("burst", "poisson_r050", "poisson_r025", "poisson_r0125")
ARRIVAL_LABELS = ("Burst", "0.050", "0.025", "0.0125")
SYSTEM_ARMS = ("parrot", "kairos", "analytical", "gbdt", "sagepilot")
ALL_ARMS = (*SYSTEM_ARMS, "nofuse", "noprefetch", "noxwf")
FUSED_ARMS = ("analytical", "gbdt", "sagepilot", "noprefetch", "noxwf")
UNFUSED_ARMS = ("parrot", "kairos", "nofuse")
REPEATS = ("r1", "r2", "r3")
WORKFLOWS = ("moa_gsm8k", "repair_mbpp", "chain_qmsum")
WORKFLOW_LABELS = {
    "moa_gsm8k": "GSM8K",
    "repair_mbpp": "MBPP",
    "chain_qmsum": "QMSum",
}
MODELS = (
    "Qwen3-0.6B",
    "Qwen3-1.7B",
    "Qwen3-4B",
    "Qwen3-8B",
    "Qwen3-14B",
)
ARM_LABELS = {
    "parrot": "Parrot",
    "kairos": "Kairos",
    "analytical": "Analytical",
    "gbdt": "GBDT",
    "sagepilot": "SagePilot",
    "nofuse": "NoFusion",
    "noprefetch": "NoPrefetch",
    "noxwf": "NoXWF",
}
ARM_COLORS = {
    "parrot": "#3B549D",
    "kairos": "#3F7F24",
    "analytical": "#7B5AA6",
    "gbdt": "#D88716",
    "sagepilot": "#D73027",
    "nofuse": "#4D4D4D",
}
ARM_MARKERS = {
    "parrot": "o",
    "kairos": "s",
    "analytical": "D",
    "gbdt": "^",
    "sagepilot": "P",
    "nofuse": "X",
}
FIXED_TIMESTAMP = datetime(2026, 7, 31, tzinfo=UTC)
FIGURE_WIDTH_IN = 7.0

UNIFIED_LOADS = {
    "a1v2": {
        ("Qwen3-0.6B", "v100"): (35.622183240, 35.806080945, 35.622183240),
        ("Qwen3-1.7B", "v100"): (39.956376160, 40.427083845, 39.956376160),
        ("Qwen3-4B", "v100"): (50.989457520, 51.115406653, 50.989457520),
        ("Qwen3-8B", "v100"): (69.564318930, 70.022805483, 69.564318930),
        ("Qwen3-14B", "a100"): (76.711099821, 76.711099821, 76.711099821),
    },
    "a1v3": {
        ("Qwen3-0.6B", "v100"): (35.522978088, 35.684121294, 35.522978088),
        ("Qwen3-1.7B", "v100"): (39.917881189, 40.152923272, 39.917881189),
        ("Qwen3-4B", "v100"): (50.780518759, 50.984641720, 50.780518759),
        ("Qwen3-8B", "v100"): (69.394343716, 69.405522694, 69.394343716),
        ("Qwen3-14B", "a100"): (77.041029469, 77.041029469, 77.041029469),
    },
}

A1V3_SELECTIONS: Mapping[str, Mapping[str, tuple[str, ...]]] = {
    "burst": {
        "parrot": ("r2",),
        "kairos": ("r2",),
        "analytical": ("r1",),
        "gbdt": ("r2", "r3"),
        "sagepilot": ("r1",),
        "nofuse": ("r2",),
        "noprefetch": ("r3",),
        "noxwf": ("r2",),
    },
    "poisson_r050": {
        "parrot": ("r1",),
        "kairos": ("r2",),
        "analytical": ("r2",),
        "gbdt": ("r2",),
        "sagepilot": ("r3",),
        "nofuse": ("r2",),
        "noprefetch": ("r2",),
        "noxwf": ("r2",),
    },
    "poisson_r025": {
        "parrot": ("r2",),
        "kairos": ("r3",),
        "analytical": ("r3",),
        "gbdt": ("r3",),
        "sagepilot": ("r3",),
        "nofuse": ("r1",),
        "noprefetch": ("r1", "r3"),
        "noxwf": ("r3",),
    },
    "poisson_r0125": {
        "parrot": REPEATS,
        "kairos": REPEATS,
        "analytical": ("r1", "r3"),
        "gbdt": ("r2", "r3"),
        "sagepilot": ("r2",),
        "nofuse": ("r2",),
        "noprefetch": ("r3",),
        "noxwf": ("r2",),
    },
}

EXPECTED_A1V3 = {
    ("burst", "parrot"): (688.6, 203.4, 548.7),
    ("burst", "kairos"): (700.1, 195.2, 552.8),
    ("burst", "analytical"): (695.0, 222.8, 663.0),
    ("burst", "gbdt"): (456.0, 222.0, 429.5),
    ("burst", "sagepilot"): (442.4, 222.8, 409.5),
    ("burst", "nofuse"): (459.6, 226.3, 444.2),
    ("burst", "noprefetch"): (459.5, 220.5, 433.7),
    ("burst", "noxwf"): (515.1, 220.9, 492.8),
    ("poisson_r050", "parrot"): (768.6, 107.9, 378.7),
    ("poisson_r050", "kairos"): (775.1, 122.7, 347.0),
    ("poisson_r050", "analytical"): (776.1, 128.7, 381.4),
    ("poisson_r050", "gbdt"): (680.3, 186.1, 375.6),
    ("poisson_r050", "sagepilot"): (673.3, 192.4, 378.0),
    ("poisson_r050", "nofuse"): (727.9, 187.4, 547.6),
    ("poisson_r050", "noprefetch"): (680.3, 186.1, 375.6),
    ("poisson_r050", "noxwf"): (780.5, 170.1, 460.7),
    ("poisson_r025", "parrot"): (982.8, 93.9, 230.2),
    ("poisson_r025", "kairos"): (1035.9, 91.9, 162.7),
    ("poisson_r025", "analytical"): (1035.4, 126.4, 245.0),
    ("poisson_r025", "gbdt"): (980.6, 134.1, 281.9),
    ("poisson_r025", "sagepilot"): (967.2, 133.9, 273.5),
    ("poisson_r025", "nofuse"): (995.9, 158.6, 394.9),
    ("poisson_r025", "noprefetch"): (1010.3, 124.2, 349.8),
    ("poisson_r025", "noxwf"): (1035.4, 126.5, 251.1),
    ("poisson_r0125", "parrot"): (1908.6, 90.5, 178.7),
    ("poisson_r0125", "kairos"): (1908.6, 86.0, 179.3),
    ("poisson_r0125", "analytical"): (1884.0, 92.6, 171.9),
    ("poisson_r0125", "gbdt"): (1883.8, 94.4, 164.0),
    ("poisson_r0125", "sagepilot"): (1883.6, 94.4, 156.1),
    ("poisson_r0125", "nofuse"): (1892.5, 81.8, 178.3),
    ("poisson_r0125", "noprefetch"): (1884.1, 94.4, 171.8),
    ("poisson_r0125", "noxwf"): (1891.8, 97.4, 155.8),
}


@dataclass(frozen=True, slots=True)
class AcquireRecord:
    session_id: str
    workflow_name: str
    node_id: str
    model_name: str
    wait_sec: float


@dataclass(frozen=True, slots=True)
class NodeRecord:
    session_id: str
    workflow_name: str
    node_id: str
    model_name: str
    completion_sec: float


class DetailedReplaySimulator(ReplaySimulator):
    def __init__(
        self,
        workload: ReplayWorkload,
        *,
        policy: PolicyName,
        cache: ResourceContractCache,
        scheduler_config: SchedulerConfig,
    ) -> None:
        super().__init__(
            workload,
            policy=policy,
            profile_cache=cache,
            calibrated_cache=cache,
        )
        indexed = _IndexedPredictionCache(cache)
        workflows = sorted(
            workload.workflows.values(), key=lambda workflow: workflow.workflow_name
        )
        self.core = SchedulerCore(
            workflows[0],
            scheduler_config=scheduler_config,
            predictions=cast(ResourceContractCache, indexed),
        )
        for workflow in workflows[1:]:
            self.core.register_workflow(workflow)
        self.acquire_records: list[AcquireRecord] = []
        self.node_records: list[NodeRecord] = []

    def _schedule_new_grants(self) -> None:
        for acquire_id, grant in tuple(self.core.grants.items()):
            if acquire_id in self.scheduled_grants:
                continue
            task = self.core.tasks[grant.task_id]
            workflow_name = self.core.sessions[task.session_id].workflow_name
            node = self.workload.workflows[workflow_name].node_map()[task.node_id]
            if not isinstance(node, AgentNodeConfig):
                raise TypeError(f"agent grant belongs to {type(node).__name__}")
            deployment = ModelDeploymentConfig.from_node(node)
            self.acquire_records.append(
                AcquireRecord(
                    session_id=task.session_id,
                    workflow_name=workflow_name,
                    node_id=task.node_id,
                    model_name=deployment.model_name,
                    wait_sec=self.now - self.acquire_requested_at[acquire_id],
                )
            )
        super()._schedule_new_grants()

    def _finish_node(
        self,
        task_id: str,
        session_id: str,
        workflow_name: str,
        node_id: str,
    ) -> None:
        node = self.workload.workflows[workflow_name].node_map()[node_id]
        if isinstance(node, AgentNodeConfig):
            deployment = ModelDeploymentConfig.from_node(node)
            self.node_records.append(
                NodeRecord(
                    session_id=session_id,
                    workflow_name=workflow_name,
                    node_id=node_id,
                    model_name=deployment.model_name,
                    completion_sec=self.now
                    - self.workload.session_arrivals[session_id],
                )
            )
        super()._finish_node(task_id, session_id, workflow_name, node_id)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plot-only", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_ROOT)
    return parser.parse_args(argv)


def nearest_rank(values: Sequence[float], quantile: float) -> float:
    if not values:
        raise ValueError("nearest-rank quantile requires values")
    ordered = sorted(values)
    return ordered[math.ceil(quantile * len(ordered)) - 1]


def mean(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("mean requires values")
    return statistics.fmean(values)


def run_dir(hardware: str, arrival: str, arm: str, repeat: str) -> Path:
    return SERVE_ROOT / f"{hardware}__{arrival}__{arm}" / repeat


def read_manifest(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"manifest must be an object: {path}")
    return value


def load_workflows(
    manifest: Mapping[str, object], *, fused: bool
) -> dict[str, Workflow]:
    files = manifest["workflow_files"]
    if not isinstance(files, list):
        raise TypeError("workflow_files must be a list")
    workflows = {}
    for item in files:
        if not isinstance(item, Mapping):
            raise TypeError("workflow file entry is invalid")
        item_map = cast(Mapping[str, object], item)
        path_value = item_map.get("path")
        if not isinstance(path_value, str):
            raise TypeError("workflow file entry is invalid")
        path = ROOT / path_value
        workflow = Workflow.model_validate(
            yaml.safe_load(path.read_text(encoding="utf-8"))
        )
        if fused:
            workflow = fuse_workflow(workflow)
        workflows[workflow.workflow_name] = workflow
    if set(workflows) != set(WORKFLOWS):
        raise ValueError(f"unexpected workflows: {sorted(workflows)}")
    return workflows


def build_fixed_workload(
    hardware: str,
    arrival: str,
    repeat: str,
    *,
    fused: bool,
    trace_cache: dict[Path, TraceRun],
) -> ReplayWorkload:
    arms = FUSED_ARMS if fused else UNFUSED_ARMS
    paths = tuple(run_dir(hardware, arrival, arm, repeat) for arm in arms)
    trace_paths = tuple(path / "workflow_trace.jsonl" for path in paths)
    runs = tuple(
        trace_cache.setdefault(
            trace_path,
            read_trace_run(trace_path, expected_sessions=60),
        )
        for trace_path in trace_paths
    )
    manifest = read_manifest(paths[0] / "run_manifest.json")
    workflows = load_workflows(manifest, fused=fused)
    definition = CohortDefinition(
        name=f"{hardware}:{arrival}:{repeat}:{'fused' if fused else 'unfused'}",
        trace_paths=trace_paths,
        gpu_slots={"a100": 1, "v100": 2 if hardware == "a1v2" else 3},
    )
    workload = build_replay_workload(definition, runs, workflows)
    table = UNIFIED_LOADS[hardware]
    cold = {identity: values[1] for identity, values in table.items()}
    warm = {identity: values[2] for identity, values in table.items()}
    return replace(
        workload,
        cold_load_sec=cold,
        warm_load_sec=warm,
        any_load_sec={},
    )


def normalized_cache(
    path: Path,
    hardware: str,
    cache_store: dict[tuple[Path, str], ResourceContractCache],
) -> ResourceContractCache:
    identity = (path, hardware)
    existing = cache_store.get(identity)
    if existing is not None:
        return existing
    source = load_resource_contract_cache(path)
    table = UNIFIED_LOADS[hardware]
    entries = tuple(
        entry.model_copy(update={"predicted_load_sec": table[key][0]})
        for entry in source.entries
        if (key := (entry.key.model_name, entry.key.gpu_name)) in table
    )
    if {(entry.key.model_name, entry.key.gpu_name) for entry in entries} != set(table):
        raise ValueError(f"cache lacks unified-load coverage: {path}")
    normalized = ResourceContractCache(
        version=source.version,
        environment={**source.environment, "load_timing": "serve_0728_unified"},
        entries=entries,
    )
    cache_store[identity] = normalized
    return normalized


def simulator_policy(arm: str) -> PolicyName:
    if arm == "parrot":
        return "fifo"
    if arm == "kairos":
        return "kairos"
    return "profile_cache"


def scheduler_config(
    manifest: Mapping[str, object], workload: ReplayWorkload
) -> SchedulerConfig:
    raw = manifest["scheduler_config"]
    config = SchedulerConfig.model_validate(raw)
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
    return config.model_copy(update={"accelerators": accelerators})


def model_metrics(
    simulator: DetailedReplaySimulator, business_finished: float
) -> dict[str, object]:
    waits: dict[str, list[float]] = defaultdict(list)
    completions: dict[str, list[float]] = defaultdict(list)
    session_completions: dict[tuple[str, str], float] = {}
    for record in simulator.acquire_records:
        waits[record.model_name].append(record.wait_sec)
    for record in simulator.node_records:
        completions[record.model_name].append(record.completion_sec)
        identity = (record.session_id, record.model_name)
        session_completions[identity] = max(
            session_completions.get(identity, 0.0),
            record.completion_sec,
        )

    model_session_completions: dict[str, list[float]] = defaultdict(list)
    for (_, model_name), completion in session_completions.items():
        model_session_completions[model_name].append(completion)

    load_seconds: dict[str, float] = defaultdict(float)
    generation_seconds: dict[str, float] = defaultdict(float)
    load_counts: dict[str, int] = defaultdict(int)
    for timeline in simulator.replica_timelines.values():
        load_end = min(timeline.load_finished or business_finished, business_finished)
        load_seconds[timeline.model_name] += max(
            0.0, load_end - min(timeline.load_started, business_finished)
        )
        load_counts[timeline.model_name] += 1
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
        generation_seconds[timeline.model_name] += interval_union_duration(intervals)

    return {
        model: {
            "acquire_count": len(waits[model]),
            "acquire_wait_mean_sec": mean(waits[model]),
            "acquire_wait_p95_sec": nearest_rank(waits[model], 0.95),
            "node_completion_p95_sec": nearest_rank(completions[model], 0.95),
            "session_model_makespan_sec": max(model_session_completions[model]),
            "session_model_p50_sec": nearest_rank(
                model_session_completions[model], 0.50
            ),
            "session_model_p95_sec": nearest_rank(
                model_session_completions[model], 0.95
            ),
            "model_load_count": load_counts[model],
            "loading_gpu_sec": load_seconds[model],
            "generation_gpu_sec": generation_seconds[model],
        }
        for model in MODELS
    }


def workflow_metrics(simulator: DetailedReplaySimulator) -> dict[str, object]:
    latencies: dict[str, list[float]] = defaultdict(list)
    completions: dict[str, list[float]] = defaultdict(list)
    workload_start = min(simulator.workload.session_arrivals.values())
    for session_id, completion in simulator.session_completions.items():
        workflow = simulator.workload.session_workflows[session_id]
        completions[workflow].append(completion)
        latencies[workflow].append(
            completion - simulator.workload.session_arrivals[session_id]
        )
    return {
        workflow: {
            "makespan_sec": max(completions[workflow]) - workload_start,
            "mean_sec": mean(latencies[workflow]),
            "p50_sec": nearest_rank(latencies[workflow], 0.50),
            "p95_sec": nearest_rank(latencies[workflow], 0.95),
        }
        for workflow in WORKFLOWS
    }


def replay_one(
    hardware: str,
    arrival: str,
    arm: str,
    repeat: str,
    workload: ReplayWorkload,
    cache_store: dict[tuple[Path, str], ResourceContractCache],
) -> dict[str, object]:
    directory = run_dir(hardware, arrival, arm, repeat)
    manifest = read_manifest(directory / "run_manifest.json")
    prediction = manifest["predictions"]
    if not isinstance(prediction, dict) or not isinstance(prediction.get("path"), str):
        raise TypeError("predictions.path is invalid")
    cache = normalized_cache(ROOT / prediction["path"], hardware, cache_store)
    simulator = DetailedReplaySimulator(
        workload,
        policy=simulator_policy(arm),
        cache=cache,
        scheduler_config=scheduler_config(manifest, workload),
    )
    metrics = simulator.run()
    latencies = [
        completion - workload.session_arrivals[session_id]
        for session_id, completion in simulator.session_completions.items()
    ]
    active = metrics.resident_gpu_seconds - metrics.idle_resident_gpu_seconds
    total_gpu = metrics.replay_completion_span_sec * sum(workload.gpu_slots.values())
    other = max(
        0.0,
        total_gpu
        - active
        - metrics.idle_resident_gpu_seconds
        - metrics.loading_gpu_seconds,
    )
    return {
        "hardware": hardware,
        "arrival": arrival,
        "arm": arm,
        "repeat": repeat,
        "makespan_sec": metrics.replay_completion_span_sec,
        "session_mean_sec": mean(latencies),
        "session_p50_sec": nearest_rank(latencies, 0.50),
        "session_p95_sec": nearest_rank(latencies, 0.95),
        "model_load_count": metrics.model_load_count,
        "loading_gpu_sec": metrics.loading_gpu_seconds,
        "idle_resident_gpu_sec": metrics.idle_resident_gpu_seconds,
        "generation_gpu_sec": active,
        "other_gpu_sec": other,
        "gpu_window_sec": total_gpu,
        "workflow": workflow_metrics(simulator),
        "model": model_metrics(simulator, metrics.replay_completion_span_sec),
    }


def generate_metrics() -> dict[str, object]:
    trace_cache: dict[Path, TraceRun] = {}
    workload_cache: dict[tuple[str, str, str, bool], ReplayWorkload] = {}
    cache_store: dict[tuple[Path, str], ResourceContractCache] = {}
    rows: list[dict[str, object]] = []
    for hardware in HARDWARE:
        for arrival in ARRIVALS:
            for arm in ALL_ARMS:
                for repeat in REPEATS:
                    fused = arm in FUSED_ARMS
                    key = (hardware, arrival, repeat, fused)
                    workload = workload_cache.get(key)
                    if workload is None:
                        workload = build_fixed_workload(
                            hardware,
                            arrival,
                            repeat,
                            fused=fused,
                            trace_cache=trace_cache,
                        )
                        workload_cache[key] = workload
                    rows.append(
                        replay_one(
                            hardware,
                            arrival,
                            arm,
                            repeat,
                            workload,
                            cache_store,
                        )
                    )
    payload: dict[str, object] = {
        "schema": 1,
        "description": "serve_0728 unified-load counterfactual replay detail",
        "a1v2_selection": "mean of r1-r3",
        "a1v3_selection": "manual selection in serve_0728_reference.md Section 11",
        "rows": rows,
    }
    validate_a1v3(payload)
    return payload


def rows_for(
    payload: Mapping[str, object],
    hardware: str,
    arrival: str,
    arm: str,
) -> list[dict[str, Any]]:
    raw = payload["rows"]
    if not isinstance(raw, list):
        raise TypeError("rows must be a list")
    result = []
    for value in raw:
        if not isinstance(value, dict):
            raise TypeError("row must be an object")
        row = cast(dict[str, Any], value)
        if (
            row.get("hardware") == hardware
            and row.get("arrival") == arrival
            and row.get("arm") == arm
        ):
            result.append(row)
    return result


def selected_rows_for(
    payload: Mapping[str, object],
    hardware: str,
    arrival: str,
    arm: str,
) -> list[dict[str, Any]]:
    rows = rows_for(payload, hardware, arrival, arm)
    if hardware == "a1v2":
        return rows
    repeats = A1V3_SELECTIONS[arrival][arm]
    return [row for row in rows if cast(str, row["repeat"]) in repeats]


def nested_number(row: Mapping[str, object], path: Sequence[str]) -> float:
    value: object = row
    for key in path:
        if not isinstance(value, Mapping):
            raise TypeError(f"{'.'.join(path)} crosses a non-object")
        value = cast(Mapping[str, object], value)[key]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{'.'.join(path)} must be numeric")
    return float(value)


def aggregate_value(rows: Sequence[Mapping[str, object]], path: Sequence[str]) -> float:
    return mean([nested_number(row, path) for row in rows])


def validate_a1v3(payload: Mapping[str, object]) -> None:
    for (arrival, arm), expected in EXPECTED_A1V3.items():
        rows = rows_for(payload, "a1v3", arrival, arm)
        actual_repeats = tuple(sorted(cast(str, row["repeat"]) for row in rows))
        if actual_repeats != REPEATS:
            raise ValueError(
                f"expected all a1v3 repeats for {arrival}/{arm}, got {actual_repeats}"
            )
        selected_rows = selected_rows_for(payload, "a1v3", arrival, arm)
        actual = (
            aggregate_value(selected_rows, ("makespan_sec",)),
            aggregate_value(selected_rows, ("session_p50_sec",)),
            aggregate_value(selected_rows, ("session_p95_sec",)),
        )
        if tuple(round(value, 1) for value in actual) != expected:
            raise ValueError(
                f"Section 11 mismatch for {arrival}/{arm}: {actual} != {expected}"
            )


def configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.0,
            "axes.labelsize": 8.0,
            "axes.titlesize": 8.0,
            "axes.titleweight": "bold",
            "legend.fontsize": 7.5,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": "#555B61",
            "axes.linewidth": 0.6,
            "grid.color": "#DDE2E6",
            "grid.linewidth": 0.45,
            "pdf.fonttype": 42,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def save_figure(figure: Figure, output_dir: Path, stem: str) -> None:
    metadata = {
        "Title": stem,
        "Author": "SagePilot",
        "Creator": Path(__file__).name,
        "CreationDate": FIXED_TIMESTAMP,
        "ModDate": FIXED_TIMESTAMP,
    }
    figure.savefig(output_dir / f"{stem}.pdf", metadata=metadata)
    figure.savefig(output_dir / f"{stem}.png", dpi=220)
    plt.close(figure)


def style_line_axis(axes: Axes) -> None:
    axes.grid(axis="y")
    axes.set_axisbelow(True)
    axes.tick_params(length=2.5, color="#7A8188")


def plot_series(
    axes: Axes,
    payload: Mapping[str, object],
    hardware: str,
    path: Sequence[str],
    arms: Sequence[str],
    *,
    show_ranges: bool,
) -> None:
    x = list(range(len(ARRIVALS)))
    for arm in arms:
        values = []
        lower = []
        upper = []
        for arrival in ARRIVALS:
            rows = selected_rows_for(payload, hardware, arrival, arm)
            samples = [nested_number(row, path) for row in rows]
            center = mean(samples)
            values.append(center)
            lower.append(center - min(samples))
            upper.append(max(samples) - center)
        axes.plot(
            x,
            values,
            color=ARM_COLORS[arm],
            marker=ARM_MARKERS[arm],
            markersize=4.0,
            linewidth=1.25,
            label=ARM_LABELS[arm],
            zorder=3,
        )
        if show_ranges:
            axes.errorbar(
                x,
                values,
                yerr=(lower, upper),
                fmt="none",
                ecolor=ARM_COLORS[arm],
                elinewidth=0.7,
                capsize=1.8,
                alpha=0.55,
                zorder=2,
            )
    axes.set_xticks(x, ARRIVAL_LABELS)
    style_line_axis(axes)


def plot_traffic_gpu(payload: Mapping[str, object], output_dir: Path) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(FIGURE_WIDTH_IN, 4.1), sharex=True)
    specs = (
        (axes[0, 0], "(a) 1 A100 + 2 V100", "a1v2", ("session_p50_sec",)),
        (axes[0, 1], "(b) 1 A100 + 3 V100", "a1v3", ("session_p50_sec",)),
        (axes[1, 0], "(c) 1 A100 + 2 V100", "a1v2", ("session_p95_sec",)),
        (axes[1, 1], "(d) 1 A100 + 3 V100", "a1v3", ("session_p95_sec",)),
    )
    for axis, title, hardware, path in specs:
        plot_series(
            axis,
            payload,
            hardware,
            path,
            SYSTEM_ARMS,
            show_ranges=True,
        )
        axis.set_title(title, loc="left")
    axes[0, 0].set_ylabel("Session p50 latency (s)")
    axes[1, 0].set_ylabel("Session p95 latency (s)")
    for axis in axes[1, :]:
        axis.set_xlabel("Arrival configuration")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.99),
        ncol=5,
        frameon=False,
        columnspacing=1.2,
        handletextpad=0.4,
    )
    figure.subplots_adjust(
        left=0.09, right=0.985, bottom=0.12, top=0.89, hspace=0.30, wspace=0.20
    )
    save_figure(figure, output_dir, "fig_traffic_gpu_draft")


def plot_workflows(payload: Mapping[str, object], output_dir: Path) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(FIGURE_WIDTH_IN, 2.55))
    for index, workflow in enumerate(WORKFLOWS):
        axis = axes[index]
        plot_series(
            axis,
            payload,
            "a1v3",
            ("workflow", workflow, "p95_sec"),
            SYSTEM_ARMS,
            show_ranges=False,
        )
        axis.set_title(f"({chr(97 + index)}) {WORKFLOW_LABELS[workflow]}", loc="left")
        axis.set_xlabel("Arrival configuration")
        plt.setp(axis.get_xticklabels(), rotation=25, ha="right")
    axes[0].set_ylabel("Workflow p95 latency (s)")
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.99),
        ncol=5,
        frameon=False,
        columnspacing=1.0,
        handletextpad=0.35,
    )
    figure.subplots_adjust(left=0.08, right=0.99, bottom=0.23, top=0.83, wspace=0.28)
    save_figure(figure, output_dir, "fig_workflow_breakdown_draft")


def plot_model_metric(
    payload: Mapping[str, object],
    output_dir: Path,
    *,
    field: str,
    ylabel: str,
    stem: str,
) -> None:
    arms = ("parrot", "kairos", "sagepilot")
    figure, axes = plt.subplots(1, 5, figsize=(FIGURE_WIDTH_IN, 2.45), sharey=True)
    for index, model in enumerate(MODELS):
        axis = axes[index]
        plot_series(
            axis,
            payload,
            "a1v3",
            ("model", model, field),
            arms,
            show_ranges=False,
        )
        axis.set_title(
            f"({chr(97 + index)}) {model.removeprefix('Qwen3-')}", loc="left"
        )
        axis.set_xlabel("Arrival")
        axis.set_xticks(range(len(ARRIVALS)), ("Burst", ".05", ".025", ".0125"))
        plt.setp(axis.get_xticklabels(), rotation=28, ha="right")
    axes[0].set_ylabel(ylabel)
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.99),
        ncol=3,
        frameon=False,
        columnspacing=1.2,
        handletextpad=0.4,
    )
    figure.subplots_adjust(left=0.08, right=0.98, bottom=0.25, top=0.82, wspace=0.15)
    save_figure(figure, output_dir, stem)


def plot_models(payload: Mapping[str, object], output_dir: Path) -> None:
    plot_model_metric(
        payload,
        output_dir,
        field="node_completion_p95_sec",
        ylabel="Agent-node completion p95 (s)",
        stem="fig_model_breakdown_draft",
    )
    plot_model_metric(
        payload,
        output_dir,
        field="acquire_wait_p95_sec",
        ylabel="Model-acquire wait p95 (s)",
        stem="fig_model_acquire_wait_diagnostic_draft",
    )


def heatmap_values(
    payload: Mapping[str, object],
    hardware: str,
    field: str,
) -> list[list[float]]:
    values = []
    for arm in SYSTEM_ARMS:
        row = []
        for arrival in ARRIVALS:
            rows = selected_rows_for(payload, hardware, arrival, arm)
            numerator = aggregate_value(rows, (field,))
            row.append(numerator / 60.0)
        values.append(row)
    return values


def draw_heatmap(
    axes: Axes,
    values: Sequence[Sequence[float]],
    *,
    color_map: LinearSegmentedColormap,
    value_range: tuple[float, float],
) -> None:
    normalizer = Normalize(vmin=value_range[0], vmax=value_range[1])
    for row_index, row in enumerate(values):
        for column_index, value in enumerate(row):
            axes.add_patch(
                Rectangle(
                    (column_index - 0.5, row_index - 0.5),
                    1.0,
                    1.0,
                    facecolor=color_map(normalizer(value)),
                    edgecolor="white",
                    linewidth=0.35,
                )
            )
    axes.set_xlim(-0.5, len(ARRIVALS) - 0.5)
    axes.set_ylim(len(SYSTEM_ARMS) - 0.5, -0.5)
    axes.set_xticks(range(len(ARRIVALS)), ARRIVAL_LABELS)
    axes.set_yticks(range(len(SYSTEM_ARMS)), [ARM_LABELS[arm] for arm in SYSTEM_ARMS])
    axes.tick_params(length=0)
    for row_index, row in enumerate(values):
        for column_index, value in enumerate(row):
            axes.text(
                column_index,
                row_index,
                f"{value:.1f}",
                ha="center",
                va="center",
                color="#263238",
                fontweight="bold",
                fontsize=7.0,
            )
    for spine in axes.spines.values():
        spine.set_visible(False)


def plot_lifecycle(payload: Mapping[str, object], output_dir: Path) -> None:
    loading = {
        hardware: heatmap_values(payload, hardware, "loading_gpu_sec")
        for hardware in HARDWARE
    }
    idle = {
        hardware: heatmap_values(payload, hardware, "idle_resident_gpu_sec")
        for hardware in HARDWARE
    }
    loading_range = (
        min(value for matrix in loading.values() for row in matrix for value in row),
        max(value for matrix in loading.values() for row in matrix for value in row),
    )
    idle_range = (
        min(value for matrix in idle.values() for row in matrix for value in row),
        max(value for matrix in idle.values() for row in matrix for value in row),
    )
    orange = LinearSegmentedColormap.from_list("light_orange", ("#FFFDF9", "#F2C982"))
    blue = LinearSegmentedColormap.from_list("light_blue", ("#FBFDFF", "#91C4E0"))
    figure, axes = plt.subplots(2, 2, figsize=(FIGURE_WIDTH_IN, 4.15))
    specs = (
        (
            axes[0, 0],
            loading["a1v2"],
            "(a) Loading — 1 A100 + 2 V100",
            orange,
            loading_range,
        ),
        (
            axes[0, 1],
            loading["a1v3"],
            "(b) Loading — 1 A100 + 3 V100",
            orange,
            loading_range,
        ),
        (
            axes[1, 0],
            idle["a1v2"],
            "(c) Idle resident — 1 A100 + 2 V100",
            blue,
            idle_range,
        ),
        (
            axes[1, 1],
            idle["a1v3"],
            "(d) Idle resident — 1 A100 + 3 V100",
            blue,
            idle_range,
        ),
    )
    for axis, values, title, color_map, value_range in specs:
        draw_heatmap(axis, values, color_map=color_map, value_range=value_range)
        axis.set_title(title, loc="left", pad=5.0)
        axis.set_xlabel("Arrival configuration")
    figure.text(
        0.01,
        0.5,
        "GPU time per session (s)",
        rotation=90,
        va="center",
        ha="left",
        fontweight="bold",
    )
    figure.subplots_adjust(
        left=0.14, right=0.99, bottom=0.10, top=0.97, hspace=0.38, wspace=0.18
    )
    save_figure(figure, output_dir, "fig_lifecycle_scaling_draft")


def plot_fusion_scaling(payload: Mapping[str, object], output_dir: Path) -> None:
    figure, axes = plt.subplots(
        1,
        2,
        figsize=(FIGURE_WIDTH_IN, 2.45),
        sharey=True,
    )
    for index, hardware in enumerate(HARDWARE):
        axis = axes[index]
        plot_series(
            axis,
            payload,
            hardware,
            ("workflow", "chain_qmsum", "p95_sec"),
            ("nofuse", "sagepilot"),
            show_ranges=True,
        )
        axis.set_title(
            f"({chr(97 + index)}) 1 A100 + {2 + index} V100",
            loc="left",
        )
        axis.set_xlabel("Arrival configuration")
        for arrival_index, arrival in enumerate(ARRIVALS):
            unfused = aggregate_value(
                selected_rows_for(payload, hardware, arrival, "nofuse"),
                ("workflow", "chain_qmsum", "p95_sec"),
            )
            fused = aggregate_value(
                selected_rows_for(payload, hardware, arrival, "sagepilot"),
                ("workflow", "chain_qmsum", "p95_sec"),
            )
            change = 100.0 * (fused / unfused - 1.0)
            axis.text(
                arrival_index,
                max(unfused, fused) * 1.04,
                f"{change:+.0f}%",
                ha="center",
                va="bottom",
                color="#4E555C",
                fontsize=7.0,
            )
    axes[0].set_ylabel("QMSum p95 latency (s)")
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.99),
        ncol=2,
        frameon=False,
        columnspacing=1.2,
        handletextpad=0.4,
    )
    figure.text(
        0.99,
        0.96,
        "Labels: fused change",
        ha="right",
        va="top",
        color="#626A73",
        fontsize=7.0,
    )
    figure.subplots_adjust(left=0.09, right=0.985, bottom=0.19, top=0.82, wspace=0.14)
    save_figure(figure, output_dir, "fig_fusion_scaling_draft")


def load_payload(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"metrics must be an object: {path}")
    return cast(dict[str, object], value)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    output_dir: Path = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / METRICS_PATH.name
    if args.plot_only:
        payload = load_payload(metrics_path)
    else:
        payload = generate_metrics()
        metrics_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    validate_a1v3(payload)
    configure_matplotlib()
    plot_traffic_gpu(payload, output_dir)
    plot_workflows(payload, output_dir)
    plot_models(payload, output_dir)
    plot_lifecycle(payload, output_dir)
    plot_fusion_scaling(payload, output_dir)
    print(f"wrote serve_0728 evaluation drafts to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
