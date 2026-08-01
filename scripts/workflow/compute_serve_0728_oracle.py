"""Build and verify the trace-conditioned serve_0728 offline Oracle reference."""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import json
import math
import sys
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import plot_serve_0728_counterfactual_drafts as replay  # noqa: E402

from experiment.workflow.artifacts import file_sha256  # noqa: E402
from experiment.workflow.cache_replay import TraceRun, read_trace_run  # noqa: E402
from workflow.replica import ModelDeploymentConfig  # noqa: E402
from workflow.schema import AgentNodeConfig  # noqa: E402

CALCULATION = "trace_time_indexed_offline_reference"
ORTOOLS_VERSION = "9.15.6755"
TARGET_MODELS = ("Qwen3-4B", "Qwen3-8B", "Qwen3-14B")
MODEL_POOLS = {
    "Qwen3-0.6B": "v100",
    "Qwen3-1.7B": "v100",
    "Qwen3-4B": "v100",
    "Qwen3-8B": "v100",
    "Qwen3-14B": "a100",
}
POOL_SIZES = {"v100": 3, "a100": 1}
MAX_REPLICAS = 2
REPLICA_CAPACITY = 3


@dataclass(frozen=True, slots=True)
class ModelReference:
    makespan_sec: float
    p95_sec: float
    max_slots: int
    p95_slots: int


@dataclass(frozen=True, slots=True)
class ReferenceSpec:
    arrival: str
    repeat: str
    trace_sha256: str
    manifest_sha256: str
    step_sec: int
    makespan_sec: float
    p95_sec: float
    makespan_slots: int
    p95_slots: int
    target_sum_slots: int
    solve_quality: str
    bounds: Mapping[str, int]
    models: Mapping[str, ModelReference]


@dataclass(frozen=True, slots=True)
class NodeSpec:
    session_id: str
    workflow_name: str
    node_id: str
    duration_slots: int
    model_name: str | None
    dependencies: tuple[tuple[str, str], ...]


REFERENCE_SPECS = {
    "burst": ReferenceSpec(
        arrival="burst",
        repeat="r1",
        trace_sha256="f6e980dbb08be24f369639274f6a001d7796f324e39f8724cab5aca667e0d496",
        manifest_sha256="c57dd572b13abd19af6bcbc42a7cc161b4c33993f1589f6ce0f975c2d6b6faf4",
        step_sec=5,
        makespan_sec=290.0,
        p95_sec=284.9929702281952,
        makespan_slots=58,
        p95_slots=57,
        target_sum_slots=253,
        solve_quality="finite_time_relaxed_candidate",
        bounds={"makespan_best_bound_slots": 44},
        models={
            "Qwen3-4B": ModelReference(269.95805978775024, 234.9890341758728, 54, 47),
            "Qwen3-8B": ModelReference(290.0, 284.9929702281952, 58, 57),
            "Qwen3-14B": ModelReference(94.99752497673035, 90.0, 19, 18),
        },
    ),
    "poisson_r050": ReferenceSpec(
        arrival="poisson_r050",
        repeat="r3",
        trace_sha256="5cf041befe66c7b4ca8683491bbe57e656834f2a80baf8b991e17ceba7cb795a",
        manifest_sha256="2ae7a6636329858e62edc184f7c38571fec05694556d2c6a344fbaed92743387",
        step_sec=4,
        makespan_sec=496.0,
        p95_sec=75.993971824646,
        makespan_slots=124,
        p95_slots=19,
        target_sum_slots=178,
        solve_quality="finite_time_relaxed_candidate",
        bounds={
            "makespan_best_bound_slots": 124,
            "p95_best_bound_slots": 18,
            "target_sum_best_bound_slots": 98,
        },
        models={
            "Qwen3-4B": ModelReference(220.2771077156067, 67.98814463615417, 56, 17),
            "Qwen3-8B": ModelReference(227.34009647369385, 75.993971824646, 57, 19),
            "Qwen3-14B": ModelReference(76.0, 39.752307653427124, 19, 10),
        },
    ),
    "poisson_r025": ReferenceSpec(
        arrival="poisson_r025",
        repeat="r3",
        trace_sha256="ed9d1511a0a1cd2cc470073f31a21ab753d7fb7b2312bf8d23a7b72518629829",
        manifest_sha256="91499bd463787e81835b475f3dfd5e8db28ccc58b4d5f5051ee81fe13262c5f8",
        step_sec=3,
        makespan_sec=954.0,
        p95_sec=74.99566626548767,
        makespan_slots=318,
        p95_slots=25,
        target_sum_slots=122,
        solve_quality="discretized_lexicographic_optimum",
        bounds={
            "makespan_best_bound_slots": 318,
            "p95_best_bound_slots": 25,
            "target_sum_best_bound_slots": 122,
        },
        models={
            "Qwen3-4B": ModelReference(71.99566626548767, 41.844141483306885, 24, 14),
            "Qwen3-8B": ModelReference(84.0, 74.99566626548767, 28, 25),
            "Qwen3-14B": ModelReference(81.0, 11.995666265487671, 27, 4),
        },
    ),
}


def source_paths(spec: ReferenceSpec) -> tuple[Path, Path]:
    directory = replay.run_dir("a1v3", spec.arrival, "sagepilot", spec.repeat)
    return directory / "workflow_trace.jsonl", directory / "run_manifest.json"


def validate_source(spec: ReferenceSpec) -> tuple[TraceRun, Mapping[str, object]]:
    selected = replay.A1V3_SELECTIONS[spec.arrival]["sagepilot"]
    if selected != (spec.repeat,):
        raise ValueError(f"stale Oracle repeat for {spec.arrival}: {selected}")
    trace_path, manifest_path = source_paths(spec)
    if file_sha256(trace_path) != spec.trace_sha256:
        raise ValueError(f"stale Oracle trace: {trace_path}")
    if file_sha256(manifest_path) != spec.manifest_sha256:
        raise ValueError(f"stale Oracle manifest: {manifest_path}")
    run = read_trace_run(trace_path, expected_sessions=60)
    manifest = replay.read_manifest(manifest_path)
    if manifest.get("fuse_nodes") is not True:
        raise ValueError(f"Oracle trace is not fused: {trace_path}")
    scheduler = manifest.get("scheduler_config")
    if not isinstance(scheduler, Mapping):
        raise TypeError(f"scheduler_config must be an object: {manifest_path}")
    if scheduler.get("max_replicas_per_model") != MAX_REPLICAS:
        raise ValueError(f"unexpected replica limit: {manifest_path}")
    accelerators = scheduler.get("accelerators")
    if not isinstance(accelerators, list):
        raise TypeError(f"accelerators must be a list: {manifest_path}")
    counts: dict[str, int] = defaultdict(int)
    for accelerator in accelerators:
        if not isinstance(accelerator, Mapping):
            raise TypeError(f"invalid accelerator: {manifest_path}")
        gpu_kind = accelerator.get("gpu_kind")
        if not isinstance(gpu_kind, str):
            raise TypeError(f"accelerator lacks gpu_kind: {manifest_path}")
        counts[gpu_kind] += 1
    if dict(counts) != {"a100": 1, "v100": 3}:
        raise ValueError(f"unexpected cluster shape: {dict(counts)}")
    return run, manifest


def trace_problem(
    spec: ReferenceSpec,
) -> tuple[list[NodeSpec], dict[str, float], dict[str, int]]:
    run, manifest = validate_source(spec)
    workflows = replay.load_workflows(manifest, fused=True)
    first_arrival = min(run.session_arrivals.values())
    arrivals = {
        session_id: timestamp - first_arrival
        for session_id, timestamp in run.session_arrivals.items()
    }
    nodes = []
    for session_id in sorted(run.session_arrivals):
        workflow_name = run.session_workflows[session_id]
        workflow = workflows[workflow_name]
        node_map = workflow.node_map()
        for node_id in workflow.graph.topological_order:
            node = node_map[node_id]
            task = run.tasks[(session_id, node_id)]
            model_name = (
                ModelDeploymentConfig.from_node(node).model_name
                if isinstance(node, AgentNodeConfig)
                else None
            )
            if (
                isinstance(node, AgentNodeConfig)
                and node.execution.serving.max_num_seqs != REPLICA_CAPACITY
            ):
                raise ValueError(
                    f"unexpected serving concurrency: {workflow_name}/{node_id}"
                )
            nodes.append(
                NodeSpec(
                    session_id=session_id,
                    workflow_name=workflow_name,
                    node_id=node_id,
                    duration_slots=math.floor(task.duration_sec / spec.step_sec),
                    model_name=model_name,
                    dependencies=tuple(
                        (session_id, dependency)
                        for dependency in workflow.graph.dependencies[node_id]
                    ),
                )
            )
    load_sec: dict[str, float] = {}
    for sample in run.loads:
        load_sec[sample.model_name] = min(
            load_sec.get(sample.model_name, sample.duration_sec),
            sample.duration_sec,
        )
    if set(load_sec) != set(MODEL_POOLS):
        raise ValueError(f"load coverage mismatch: {sorted(load_sec)}")
    load_slots = {
        model_name: math.floor(duration / spec.step_sec)
        for model_name, duration in load_sec.items()
    }
    return nodes, arrivals, load_slots


def reference_row(spec: ReferenceSpec) -> dict[str, object]:
    validate_source(spec)
    trace_path, manifest_path = source_paths(spec)
    return {
        "hardware": "a1v3",
        "arrival": spec.arrival,
        "repeat": spec.repeat,
        "source_trace": str(trace_path.relative_to(ROOT)),
        "source_manifest": str(manifest_path.relative_to(ROOT)),
        "makespan_sec": spec.makespan_sec,
        "session_p95_sec": spec.p95_sec,
        "model": {
            model_name: {
                "session_model_makespan_sec": value.makespan_sec,
                "session_model_p95_sec": value.p95_sec,
            }
            for model_name, value in spec.models.items()
        },
        "evidence": {
            "calculation": CALCULATION,
            "solver": f"OR-Tools CP-SAT {ORTOOLS_VERSION}",
            "solve_quality": spec.solve_quality,
            "time_step_sec": spec.step_sec,
            "time_rounding": "floor",
            "objective_order": ["makespan", "overall_p95", "target_model_metrics"],
            "empty_cluster_start": True,
            "gpu_counts": {"a100": 1, "v100": 3},
            "max_replicas_per_model": MAX_REPLICAS,
            "replica_concurrency": REPLICA_CAPACITY,
            "makespan_slots": spec.makespan_slots,
            "p95_slots": spec.p95_slots,
            "target_sum_slots": spec.target_sum_slots,
            "bounds": dict(spec.bounds),
            "target_model_slots": {
                model_name: {
                    "makespan": value.max_slots,
                    "p95": value.p95_slots,
                }
                for model_name, value in spec.models.items()
            },
            "trace_sha256": spec.trace_sha256,
            "manifest_sha256": spec.manifest_sha256,
        },
    }


def build_reference_oracle() -> dict[str, object]:
    return {
        "definition": (
            "Trace-only time-indexed offline reference derived from selected "
            "SagePilot traces; preserves arrivals, DAG dependencies, model-GPU "
            "compatibility, per-GPU residency, blocking loads, replica limits, and "
            "serving concurrency without replay or system execution"
        ),
        "rows": [
            reference_row(REFERENCE_SPECS[arrival]) for arrival in REFERENCE_SPECS
        ],
    }


def verify_reference(spec: ReferenceSpec, time_limit_sec: float) -> dict[str, object]:
    try:
        cp_model = importlib.import_module("ortools.sat.python.cp_model")
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "verification requires `uv run --with ortools==9.15.6755`"
        ) from error

    version = importlib.metadata.version("ortools")
    if version != ORTOOLS_VERSION:
        raise RuntimeError(f"expected ortools {ORTOOLS_VERSION}, got {version}")
    nodes, arrivals, load_slots = trace_problem(spec)
    horizon = spec.makespan_slots
    node_by_key = {(node.session_id, node.node_id): node for node in nodes}
    model = cp_model.CpModel()
    start_var: dict[tuple[str, str], Any] = {}
    completion_var: dict[tuple[str, str], Any] = {}
    model_active: dict[tuple[str, int], list[Any]] = defaultdict(list)
    node_active: dict[tuple[str, str, int], list[Any]] = defaultdict(list)
    earliest_start: dict[tuple[str, str], int] = {}
    for node in nodes:
        release = math.floor(arrivals[node.session_id] / spec.step_sec)
        earliest_start[(node.session_id, node.node_id)] = max(
            (
                earliest_start[dependency] + node_by_key[dependency].duration_slots
                for dependency in node.dependencies
            ),
            default=release,
        )
    for node in nodes:
        key = (node.session_id, node.node_id)
        start = model.new_int_var(0, horizon, f"s:{node.session_id}:{node.node_id}")
        completion = model.new_int_var(
            0, horizon, f"c:{node.session_id}:{node.node_id}"
        )
        start_var[key] = start
        completion_var[key] = completion
        model.add(completion == start + node.duration_slots)
        model.add(start >= math.floor(arrivals[node.session_id] / spec.step_sec))
        for dependency in node.dependencies:
            model.add(start >= completion_var[dependency])
        if node.model_name is None:
            continue
        choices = []
        for tick in range(earliest_start[key], horizon - node.duration_slots + 1):
            choice = model.new_bool_var(f"y:{node.session_id}:{node.node_id}:{tick}")
            choices.append((tick, choice))
            for active_tick in range(tick, tick + node.duration_slots):
                model_active[(node.model_name, active_tick)].append(choice)
                node_active[(node.workflow_name, node.node_id, active_tick)].append(
                    choice
                )
        model.add_exactly_one(choice for _, choice in choices)
        model.add(start == sum(tick * choice for tick, choice in choices))
    resident: dict[tuple[str, int], Any] = {}
    pool_occupancy: dict[tuple[str, int], list[Any]] = defaultdict(list)
    for model_name, pool in MODEL_POOLS.items():
        max_replicas = min(MAX_REPLICAS, POOL_SIZES[pool])
        duration = load_slots[model_name]
        load_start = {}
        for tick in range(horizon + 1):
            resident[(model_name, tick)] = model.new_int_var(
                0, max_replicas, f"r:{model_name}:{tick}"
            )
            pool_occupancy[(pool, tick)].append(resident[(model_name, tick)])
        for tick in range(horizon - duration + 1):
            load_start[tick] = model.new_int_var(
                0, max_replicas, f"l:{model_name}:{tick}"
            )
            for active_tick in range(tick, tick + duration):
                pool_occupancy[(pool, active_tick)].append(load_start[tick])
        model.add(resident[(model_name, 0)] == 0)
        for tick in range(1, horizon + 1):
            completed_load = load_start[tick - duration] if tick >= duration else 0
            model.add(
                resident[(model_name, tick)]
                <= resident[(model_name, tick - 1)] + completed_load
            )
    for pool, count in POOL_SIZES.items():
        for tick in range(horizon + 1):
            model.add(sum(pool_occupancy[(pool, tick)]) <= count)
    for model_name in MODEL_POOLS:
        for tick in range(horizon + 1):
            model.add(
                sum(model_active[(model_name, tick)])
                <= REPLICA_CAPACITY * resident[(model_name, tick)]
            )
    for terms in node_active.values():
        model.add(sum(terms) <= REPLICA_CAPACITY)
    successors: dict[tuple[str, str], list[tuple[str, str]]] = defaultdict(list)
    model_keys: dict[tuple[str, str], list[tuple[str, str]]] = defaultdict(list)
    for node in nodes:
        key = (node.session_id, node.node_id)
        for dependency in node.dependencies:
            successors[dependency].append(key)
        if node.model_name is not None:
            model_keys[(node.session_id, node.model_name)].append(key)
    terminal_keys = {
        node.session_id: (node.session_id, node.node_id)
        for node in nodes
        if not successors[(node.session_id, node.node_id)]
    }
    overall_p95 = model.new_int_var(0, horizon, "overall_p95")
    overall_outliers = []
    for session_id, terminal_key in terminal_keys.items():
        outlier = model.new_bool_var(f"overall_outlier:{session_id}")
        overall_outliers.append(outlier)
        release = math.floor(arrivals[session_id] / spec.step_sec)
        model.add(
            completion_var[terminal_key] - release <= overall_p95 + horizon * outlier
        )
    model.add(sum(overall_outliers) <= 3)
    model.add(overall_p95 <= spec.p95_slots)
    target_metrics = []
    for model_name, reference in spec.models.items():
        latencies = []
        outliers = []
        for session_id in arrivals:
            keys = model_keys.get((session_id, model_name), [])
            if not keys:
                continue
            completion = model.new_int_var(
                0, horizon, f"model_completion:{session_id}:{model_name}"
            )
            model.add_max_equality(completion, [completion_var[key] for key in keys])
            latency = model.new_int_var(
                0, horizon, f"model_latency:{session_id}:{model_name}"
            )
            model.add(
                latency == completion - math.floor(arrivals[session_id] / spec.step_sec)
            )
            latencies.append(latency)
            outliers.append(
                model.new_bool_var(f"model_outlier:{session_id}:{model_name}")
            )
        maximum = model.new_int_var(0, horizon, f"model_max:{model_name}")
        p95 = model.new_int_var(0, horizon, f"model_p95:{model_name}")
        model.add_max_equality(maximum, latencies)
        for latency, outlier in zip(latencies, outliers, strict=True):
            model.add(latency <= p95 + horizon * outlier)
        model.add(sum(outliers) <= len(latencies) - math.ceil(0.95 * len(latencies)))
        model.add(maximum <= reference.max_slots)
        model.add(p95 <= reference.p95_slots)
        target_metrics.extend((maximum, p95))
    model.add(sum(target_metrics) <= spec.target_sum_slots)
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit_sec
    solver.parameters.num_search_workers = 8
    solver.parameters.random_seed = 0
    status = solver.solve(model)
    if status not in (cp_model.FEASIBLE, cp_model.OPTIMAL):
        raise RuntimeError(
            f"Oracle reference verification failed for {spec.arrival}: "
            f"{solver.status_name(status)}"
        )
    return {
        "arrival": spec.arrival,
        "status": solver.status_name(status),
        "wall_time_sec": solver.wall_time,
        "overall_p95_slots": solver.value(overall_p95),
        "target_sum_slots": sum(solver.value(value) for value in target_metrics),
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--arrival",
        choices=("all", *REFERENCE_SPECS),
        default="all",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="verify the frozen reference with the time-indexed CP-SAT model",
    )
    parser.add_argument("--time-limit-sec", type=float, default=300.0)
    parser.add_argument("--output", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    arrivals = tuple(REFERENCE_SPECS) if args.arrival == "all" else (args.arrival,)
    payload = build_reference_oracle()
    if args.verify:
        payload["verification"] = [
            verify_reference(REFERENCE_SPECS[arrival], args.time_limit_sec)
            for arrival in arrivals
        ]
    text = json.dumps(payload, indent=2) + "\n"
    if args.output is None:
        print(text, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
