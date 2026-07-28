"""Plot node-fusion-addressable acquire waits from unfused baseline traces."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.patches import FancyBboxPatch, Patch, Rectangle

REPO_ROOT = Path(__file__).resolve().parents[2]
WORKFLOW_NAME = "qmsum_lane1"
EXPECTED_SESSIONS_PER_RUN = 10
EXPECTED_AGENT_NODES = frozenset(
    {
        "lane_a_0",
        "lane_a_1",
        "lane_a_2",
        "lane_b_0",
        "lane_b_1",
        "lane_b_2",
        "merge",
        "expand",
        "finalize",
    }
)
ORDERINGS = (
    ("Parrot", ("fifo_base_r1", "fifo_base_r2", "fifo_base_r3")),
    ("Kairos", ("kairos_base_r1", "kairos_base_r2", "kairos_base_r3")),
)

FIG_WIDTH_IN = 3.33
FIG_HEIGHT_IN = 2.85
FONT_SIZE = 8.0

C_CHAIN = "#26456E"
C_ACQUIRE = "#8F3B4F"
C_TEXT = "#202124"
C_MUTED = "#50545A"
C_GRID = "#CDD1D5"
C_CHAIN_FILL = "#E7EDF4"
C_OTHER = "#D9DEE5"


@dataclass(frozen=True)
class NodeExecution:
    started_at: float
    finished_at: float
    acquire_id: str
    task_id: str
    replica_id: str
    model_key: str


@dataclass(frozen=True)
class AcquireInterval:
    requested_at: float
    granted_at: float
    session_id: str
    node_id: str
    task_id: str
    replica_id: str
    model_key: str


@dataclass(frozen=True)
class SessionFusionCost:
    run_id: str
    session_id: str
    latency_sec: float
    intermediate_acquire_sec: float
    reload_seconds: float


def _required_text(record: Mapping[str, Any], key: str) -> str:
    value = record.get(key)
    if not isinstance(value, str) or not value:
        raise TypeError(f"{key} must be a non-empty string")
    return value


def _required_number(record: Mapping[str, Any], key: str) -> float:
    value = record.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{key} must be numeric")
    return float(value)


def _payload(event: Mapping[str, Any]) -> Mapping[str, Any]:
    value = event.get("payload")
    if not isinstance(value, dict):
        raise TypeError("trace payload must be an object")
    return value


def read_events(path: Path) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            value = json.loads(line)
            if not isinstance(value, dict):
                raise TypeError(f"trace row must be an object: {path}")
            events.append(value)
    if not events:
        raise ValueError(f"trace is empty: {path}")
    return events


def nearest_rank(values: Sequence[float], quantile: float) -> float:
    if not values:
        raise ValueError("percentile requires at least one value")
    if not 0.0 < quantile <= 1.0:
        raise ValueError("quantile must be within (0, 1]")
    ordered = sorted(values)
    return ordered[math.ceil(quantile * len(ordered)) - 1]


def extract_session_costs(
    events: Sequence[Mapping[str, Any]],
    run_id: str,
) -> list[SessionFusionCost]:
    executions: dict[str, dict[str, NodeExecution]] = defaultdict(dict)
    latencies: dict[str, float] = {}
    acquire_requests: dict[str, tuple[float, str, str, str]] = {}
    acquire_grants: dict[str, tuple[float, str, str, str, str, str]] = {}
    load_started_at: dict[str, float] = {}
    load_finished_at: dict[str, float] = {}
    load_duration_sec: dict[str, float] = {}
    load_reason: dict[str, str] = {}
    eviction_started_at: dict[str, float] = {}
    evicted_at: dict[str, float] = {}
    eviction_reason: dict[str, str] = {}

    for event in events:
        event_type = event.get("event_type")
        if (
            event_type == "acquire_requested"
            and event.get("workflow_name") == WORKFLOW_NAME
        ):
            acquire_id = _required_text(event, "acquire_id")
            if acquire_id in acquire_requests:
                raise ValueError(f"duplicate acquire request: {acquire_id}")
            acquire_requests[acquire_id] = (
                _required_number(event, "ts"),
                _required_text(event, "session_id"),
                _required_text(event, "node_id"),
                _required_text(event, "task_id"),
            )
            continue
        if (
            event_type == "acquire_granted"
            and event.get("workflow_name") == WORKFLOW_NAME
        ):
            acquire_id = _required_text(event, "acquire_id")
            if acquire_id in acquire_grants:
                raise ValueError(f"duplicate acquire grant: {acquire_id}")
            acquire_grants[acquire_id] = (
                _required_number(event, "ts"),
                _required_text(event, "session_id"),
                _required_text(event, "node_id"),
                _required_text(event, "task_id"),
                _required_text(event, "replica_id"),
                _required_text(event, "model_key"),
            )
            continue
        if event_type == "model_load_started":
            replica_id = _required_text(event, "replica_id")
            if replica_id in load_started_at:
                raise ValueError(f"duplicate model load start: {replica_id}")
            load_started_at[replica_id] = _required_number(event, "ts")
            load_reason[replica_id] = _required_text(_payload(event), "reason")
            continue
        if event_type == "model_load_finished":
            replica_id = _required_text(event, "replica_id")
            if replica_id in load_finished_at:
                raise ValueError(f"duplicate model load completion: {replica_id}")
            load_finished_at[replica_id] = _required_number(event, "ts")
            duration_sec = _required_number(_payload(event), "duration_sec")
            if duration_sec <= 0:
                raise ValueError(f"non-positive model load duration: {replica_id}")
            load_duration_sec[replica_id] = duration_sec
            continue
        if event_type == "model_eviction_started":
            replica_id = _required_text(event, "replica_id")
            if replica_id in eviction_started_at:
                raise ValueError(f"duplicate model eviction start: {replica_id}")
            eviction_started_at[replica_id] = _required_number(event, "ts")
            eviction_reason[replica_id] = _required_text(_payload(event), "reason")
            continue
        if event_type == "model_evicted":
            replica_id = _required_text(event, "replica_id")
            if replica_id in evicted_at:
                raise ValueError(f"duplicate model eviction completion: {replica_id}")
            evicted_at[replica_id] = _required_number(event, "ts")
            continue
        if (
            event_type == "session_completed"
            and event.get("workflow_name") == WORKFLOW_NAME
        ):
            session_id = _required_text(event, "session_id")
            if session_id in latencies:
                raise ValueError(f"duplicate session completion: {session_id}")
            latency_sec = _required_number(_payload(event), "latency_sec")
            if latency_sec <= 0:
                raise ValueError(f"non-positive session latency: {session_id}")
            latencies[session_id] = latency_sec
            continue
        if (
            event_type != "task_execution_finished"
            or event.get("workflow_name") != WORKFLOW_NAME
            or not event.get("acquire_id")
        ):
            continue

        session_id = _required_text(event, "session_id")
        node_id = _required_text(event, "node_id")
        if "+" in node_id:
            raise ValueError(f"fused node found in baseline trace: {node_id}")
        if node_id in executions[session_id]:
            raise ValueError(f"duplicate node execution: {session_id}/{node_id}")
        payload = _payload(event)
        execution = NodeExecution(
            started_at=_required_number(payload, "started_at"),
            finished_at=_required_number(payload, "finished_at"),
            acquire_id=_required_text(event, "acquire_id"),
            task_id=_required_text(event, "task_id"),
            replica_id=_required_text(event, "replica_id"),
            model_key=_required_text(event, "model_key"),
        )
        if execution.finished_at < execution.started_at:
            raise ValueError(
                f"node execution ends before it starts: {session_id}/{node_id}"
            )
        executions[session_id][node_id] = execution

    if set(acquire_requests) != set(acquire_grants):
        raise ValueError("QMSum acquire event identities are unmatched")
    acquires: dict[str, AcquireInterval] = {}
    for acquire_id, request in acquire_requests.items():
        grant = acquire_grants[acquire_id]
        requested_at, session_id, node_id, task_id = request
        granted_at, grant_session, grant_node, grant_task, replica_id, model_key = grant
        if (session_id, node_id, task_id) != (
            grant_session,
            grant_node,
            grant_task,
        ):
            raise ValueError(f"acquire identity changes at grant: {acquire_id}")
        if granted_at < requested_at:
            raise ValueError(f"acquire granted before request: {acquire_id}")
        acquires[acquire_id] = AcquireInterval(
            requested_at=requested_at,
            granted_at=granted_at,
            session_id=session_id,
            node_id=node_id,
            task_id=task_id,
            replica_id=replica_id,
            model_key=model_key,
        )

    if set(executions) != set(latencies):
        raise ValueError(
            f"QMSum execution/completion mismatch: "
            f"executions={sorted(executions)}, completions={sorted(latencies)}"
        )
    execution_acquire_ids = [
        execution.acquire_id
        for nodes in executions.values()
        for execution in nodes.values()
    ]
    if len(set(execution_acquire_ids)) != len(execution_acquire_ids):
        raise ValueError("baseline nodes must use independent acquire operations")
    if set(execution_acquire_ids) != set(acquires):
        raise ValueError("QMSum acquire/execution identities are unmatched")

    costs: list[SessionFusionCost] = []
    for session_id, nodes in sorted(executions.items()):
        if set(nodes) != EXPECTED_AGENT_NODES:
            missing = sorted(EXPECTED_AGENT_NODES - set(nodes))
            extra = sorted(set(nodes) - EXPECTED_AGENT_NODES)
            raise ValueError(
                f"incomplete QMSum session {session_id}: "
                f"missing={missing}, extra={extra}"
            )
        node_acquires: dict[str, AcquireInterval] = {}
        for node_id, execution in nodes.items():
            acquire = acquires.get(execution.acquire_id)
            if acquire is None:
                raise ValueError(
                    f"node execution has no acquire interval: {session_id}/{node_id}"
                )
            if (
                acquire.session_id,
                acquire.node_id,
                acquire.task_id,
                acquire.replica_id,
                acquire.model_key,
            ) != (
                session_id,
                node_id,
                execution.task_id,
                execution.replica_id,
                execution.model_key,
            ):
                raise ValueError(
                    f"node execution does not match acquire: {session_id}/{node_id}"
                )
            if not (acquire.requested_at <= acquire.granted_at <= execution.started_at):
                raise ValueError(f"invalid acquire interval: {session_id}/{node_id}")
            node_acquires[node_id] = acquire

        edges = (
            ("lane_a_0", "lane_a_1"),
            ("lane_a_1", "lane_a_2"),
            ("lane_b_0", "lane_b_1"),
            ("lane_b_1", "lane_b_2"),
            ("merge", "expand"),
            ("expand", "finalize"),
        )
        waits: dict[tuple[str, str], float] = {}
        reloads: dict[tuple[str, str], float] = {}
        for source, target in edges:
            predecessor = nodes[source]
            successor = nodes[target]
            if predecessor.model_key != successor.model_key:
                raise ValueError(f"boundary changes model: {source}->{target}")
            acquire = node_acquires[target]
            if predecessor.finished_at > acquire.requested_at:
                raise ValueError(
                    f"invalid intermediate acquire interval: "
                    f"{session_id}/{source}->{target}"
                )
            edge = (source, target)
            waits[edge] = acquire.granted_at - acquire.requested_at
            reloads[edge] = 0.0
            if predecessor.replica_id == successor.replica_id:
                continue
            load_start = load_started_at.get(successor.replica_id)
            load_finish = load_finished_at.get(successor.replica_id)
            load_duration = load_duration_sec.get(successor.replica_id)
            evict_start = eviction_started_at.get(predecessor.replica_id)
            evict_finish = evicted_at.get(predecessor.replica_id)
            if (
                load_start is None
                or load_finish is None
                or load_duration is None
                or evict_start is None
                or evict_finish is None
                or load_reason.get(successor.replica_id) != "ready_load"
                or eviction_reason.get(predecessor.replica_id) != "ready_load"
                or not (
                    predecessor.finished_at
                    <= load_start
                    <= load_finish
                    <= successor.started_at
                )
                or not (
                    predecessor.finished_at
                    <= evict_start
                    <= evict_finish
                    <= successor.started_at
                )
                or evict_finish > load_start
                or load_duration > load_finish - load_start
            ):
                raise ValueError(
                    f"replica change lacks complete ready-load evidence: "
                    f"{session_id}/{source}->{target}"
                )
            load_overlap = max(
                0.0,
                min(load_finish, acquire.granted_at)
                - max(load_start, acquire.requested_at),
            )
            reloads[edge] = min(load_duration, load_overlap)

        lane_waits = {
            lane: sum(
                waits[(f"{lane}_{index}", f"{lane}_{index + 1}")] for index in (0, 1)
            )
            for lane in ("lane_a", "lane_b")
        }
        lane_reloads = {
            lane: sum(
                reloads[(f"{lane}_{index}", f"{lane}_{index + 1}")] for index in (0, 1)
            )
            for lane in ("lane_a", "lane_b")
        }
        lane_finishes = {
            lane: nodes[f"{lane}_2"].finished_at for lane in ("lane_a", "lane_b")
        }
        original_join = max(lane_finishes.values())
        adjusted_join = max(
            lane_finishes[lane] - lane_waits[lane] for lane in ("lane_a", "lane_b")
        )
        reload_adjusted_join = max(
            lane_finishes[lane] - lane_reloads[lane] for lane in ("lane_a", "lane_b")
        )
        serial_edges = (("merge", "expand"), ("expand", "finalize"))
        intermediate_acquire_sec = (
            original_join - adjusted_join + sum(waits[edge] for edge in serial_edges)
        )
        reload_seconds = (
            original_join
            - reload_adjusted_join
            + sum(reloads[edge] for edge in serial_edges)
        )
        latency_sec = latencies[session_id]
        if intermediate_acquire_sec > latency_sec:
            raise ValueError(
                f"intermediate acquire exceeds session latency: {session_id}"
            )
        if reload_seconds > intermediate_acquire_sec:
            raise ValueError(
                f"reload duration exceeds intermediate acquire: {session_id}"
            )
        costs.append(
            SessionFusionCost(
                run_id=run_id,
                session_id=session_id,
                latency_sec=latency_sec,
                intermediate_acquire_sec=intermediate_acquire_sec,
                reload_seconds=reload_seconds,
            )
        )
    if not costs:
        raise ValueError(f"no {WORKFLOW_NAME} sessions in {run_id}")
    return costs


def latency_quantile_session(
    costs: Sequence[SessionFusionCost],
    quantile: float,
) -> SessionFusionCost:
    if not costs:
        raise ValueError("session quantile requires at least one value")
    if not 0.0 < quantile <= 1.0:
        raise ValueError("quantile must be within (0, 1]")
    latency_sec = nearest_rank(
        [cost.latency_sec for cost in costs],
        quantile,
    )
    return min(
        (cost for cost in costs if cost.latency_sec == latency_sec),
        key=lambda cost: (cost.run_id, cost.session_id),
    )


def load_baseline_run(root: Path, run_id: str) -> list[SessionFusionCost]:
    if "fuse" in run_id:
        raise ValueError(f"motivation analysis cannot read a fusion run: {run_id}")
    trace_path = root / run_id / "workflow_trace.jsonl"
    costs = extract_session_costs(read_events(trace_path), run_id)
    if len(costs) != EXPECTED_SESSIONS_PER_RUN:
        raise ValueError(
            f"{run_id} has {len(costs)} QMSum sessions, "
            f"expected {EXPECTED_SESSIONS_PER_RUN}"
        )
    return costs


def collect(root: Path) -> dict[str, list[SessionFusionCost]]:
    return {
        label: [cost for run_id in run_ids for cost in load_baseline_run(root, run_id)]
        for label, run_ids in ORDERINGS
    }


def configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": FONT_SIZE,
            "axes.labelsize": FONT_SIZE,
            "axes.titlesize": FONT_SIZE,
            "xtick.labelsize": FONT_SIZE,
            "ytick.labelsize": FONT_SIZE,
            "legend.fontsize": FONT_SIZE,
            "axes.edgecolor": C_MUTED,
            "axes.linewidth": 0.6,
            "text.color": C_TEXT,
            "axes.labelcolor": C_TEXT,
            "xtick.color": C_MUTED,
            "ytick.color": C_MUTED,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.bbox": None,
        }
    )


def plot_scheduling_units(axes: Axes) -> None:
    axes.set_xlim(0.0, 1.0)
    axes.set_ylim(0.0, 1.0)
    axes.axis("off")
    axes.text(
        0.0,
        1.0,
        "(a) Node fusion removes re-acquires",
        ha="left",
        va="top",
    )

    axes.text(0.0, 0.55, "Unfused\nnode-boundary", ha="left", va="center")
    for x in (0.38, 0.53, 0.68):
        axes.text(x, 0.55, "Q", ha="center", va="center", color=C_ACQUIRE)
    for x, label in zip((0.42, 0.57, 0.72), ("A", "B", "C"), strict=True):
        axes.add_patch(
            Rectangle(
                (x, 0.42),
                0.065,
                0.26,
                facecolor="white",
                edgecolor=C_MUTED,
                linewidth=0.6,
            )
        )
        axes.text(x + 0.0325, 0.55, label, ha="center", va="center")
    axes.text(0.80, 0.55, "3 acquires", ha="left", va="center")

    axes.text(0.0, 0.15, "Fused nodes", ha="left", va="center")
    axes.text(0.35, 0.15, "Q", ha="center", va="center", color=C_ACQUIRE)
    axes.add_patch(
        FancyBboxPatch(
            (0.40, 0.02),
            0.35,
            0.26,
            boxstyle="round,pad=0.015,rounding_size=0.02",
            facecolor=C_CHAIN_FILL,
            edgecolor=C_CHAIN,
            linewidth=0.7,
        )
    )
    axes.text(0.575, 0.15, "A  →  B  →  C", ha="center", va="center", color=C_CHAIN)
    axes.text(0.80, 0.15, "1 acquire", ha="left", va="center")


def plot_e2e_breakdown(
    axes: Axes,
    by_ordering: Mapping[str, Sequence[SessionFusionCost]],
) -> None:
    rows: list[tuple[str, str, SessionFusionCost]] = []
    for label, _ in ORDERINGS:
        costs = list(by_ordering[label])
        if len(costs) != 3 * EXPECTED_SESSIONS_PER_RUN:
            raise ValueError(f"{label} has {len(costs)} sessions, expected 30")
        rows.extend(
            (
                (label, "p50", latency_quantile_session(costs, 0.50)),
                (label, "p95", latency_quantile_session(costs, 0.95)),
            )
        )

    y_positions = (3.0, 2.1, 0.9, 0.0)
    for y, (_, _, cost) in zip(y_positions, rows, strict=True):
        other_sec = cost.latency_sec - cost.intermediate_acquire_sec
        axes.barh(
            y,
            other_sec,
            height=0.56,
            color=C_OTHER,
            edgecolor="white",
            linewidth=0.6,
            zorder=2,
        )
        axes.barh(
            y,
            cost.intermediate_acquire_sec,
            left=other_sec,
            height=0.56,
            color=C_ACQUIRE,
            edgecolor="white",
            linewidth=0.6,
            zorder=3,
        )
        axes.text(
            5.0,
            y,
            f"{cost.latency_sec:.0f}s total",
            ha="left",
            va="center",
            fontsize=FONT_SIZE,
        )
        fraction = cost.intermediate_acquire_sec / cost.latency_sec
        if cost.intermediate_acquire_sec >= 50.0:
            acquire_label = f"{cost.intermediate_acquire_sec:.0f}s ({fraction:.0%})"
            axes.text(
                other_sec + cost.intermediate_acquire_sec / 2.0,
                y,
                acquire_label,
                ha="center",
                va="center",
                color="white",
                fontsize=FONT_SIZE,
            )
        else:
            acquire_label = f"{cost.intermediate_acquire_sec:.1f}s ({fraction:.0%})"
            axes.text(
                cost.latency_sec + 5.0,
                y,
                acquire_label,
                ha="left",
                va="center",
                color=C_ACQUIRE,
                fontsize=FONT_SIZE,
            )

    axes.set_title("(b) Re-acquire wait in E2E latency", loc="left", pad=4)
    axes.set_xlim(0.0, 420.0)
    axes.set_ylim(-0.45, 4.15)
    axes.set_xticks((0, 100, 200, 300, 400))
    axes.set_yticks(y_positions)
    axes.set_yticklabels([f"{label} {quantile}" for label, quantile, _ in rows])
    axes.set_xlabel("End-to-end latency (s)")
    axes.grid(axis="x", color=C_GRID, linewidth=0.6, zorder=0)
    axes.spines[["top", "right"]].set_visible(False)
    axes.tick_params(axis="y", length=0)
    axes.legend(
        handles=[
            Patch(facecolor=C_OTHER, label="other E2E"),
            Patch(facecolor=C_ACQUIRE, label="re-acquire wait"),
        ],
        loc="upper left",
        frameon=False,
        ncols=2,
        handlelength=1.0,
        handletextpad=0.3,
        columnspacing=0.8,
        borderpad=0.0,
    )


def build(root: Path, output: Path) -> None:
    configure_matplotlib()
    by_ordering = collect(root)
    figure = plt.figure(figsize=(FIG_WIDTH_IN, FIG_HEIGHT_IN))
    grid = figure.add_gridspec(2, 1, height_ratios=(0.85, 1.85))
    scheduling_axes = figure.add_subplot(grid[0])
    plot_scheduling_units(scheduling_axes)
    plot_e2e_breakdown(figure.add_subplot(grid[1]), by_ordering)
    figure.subplots_adjust(left=0.28, right=0.985, top=0.97, bottom=0.15, hspace=0.48)
    scheduling_position = scheduling_axes.get_position()
    scheduling_axes.set_position(
        (0.06, scheduling_position.y0, 0.925, scheduling_position.height)
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output.with_suffix(".pdf"))
    figure.savefig(output.with_suffix(".png"), dpi=600)
    plt.close(figure)

    for label, _ in ORDERINGS:
        costs = by_ordering[label]
        p50 = latency_quantile_session(costs, 0.50)
        p95 = latency_quantile_session(costs, 0.95)
        print(
            f"{label:7s} sessions={len(costs):2d} "
            f"p50_e2e={p50.latency_sec:5.1f}s "
            f"p50_acquire={p50.intermediate_acquire_sec:5.1f}s "
            f"p50_fraction={p50.intermediate_acquire_sec / p50.latency_sec:4.0%} "
            f"p95_e2e={p95.latency_sec:5.1f}s "
            f"p95_acquire={p95.intermediate_acquire_sec:5.1f}s "
            f"p95_fraction={p95.intermediate_acquire_sec / p95.latency_sec:4.0%} "
            f"p95_reload={p95.reload_seconds:5.1f}s"
        )
    print(f"wrote {output.with_suffix('.pdf')} and {output.with_suffix('.png')}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, default=REPO_ROOT / "output" / "fusion_exp"
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=REPO_ROOT / "output" / "fusion_exp" / "fig" / "queueing_cost",
    )
    args = parser.parse_args()
    build(args.root, args.out)


if __name__ == "__main__":
    main()
