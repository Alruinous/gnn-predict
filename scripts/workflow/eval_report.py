"""Compare evaluation runs: scheduling method, and prediction cache.

Reads every run directory under --root and prints two tables plus a validity check.
Metrics come from the existing analysis modules; nothing is recomputed here:

- ``experiment.workflow.analysis.summarize_trial`` for makespan, session latency and
  the residency/active/idle GPU-second split.
- ``experiment.workflow.motivation_residency.analyze_residency`` for load cost and the
  attribution of request waiting.

Runs are grouped by stripping a trailing ``_r<N>`` repeat suffix, and each cell reports
the mean over repeats with the observed range, because a single run of this system is
not reproducible enough to rank on.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
from collections.abc import Sequence
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC = REPO_ROOT / "src"
if str(SRC) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(SRC))

from experiment.workflow.analysis import summarize_trial  # noqa: E402
from experiment.workflow.motivation_residency import analyze_residency  # noqa: E402

REPEAT_SUFFIX = re.compile(r"_r\d+$")

# Order the tables so the published orderings come first and the system's own
# prediction caches follow, which is how the two comparisons are read.
GROUP_ORDER = (
    "parrot",
    "kairos",
    "sys_profile_v2",
    "sys_gnn_v2",
    "sys_static",
    "sys_tabular",
)


def group_of(run_id: str) -> str:
    return REPEAT_SUFFIX.sub("", run_id)


def collect(root: Path) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for directory in sorted(root.iterdir()):
        trace = directory / "workflow_trace.jsonl"
        if not directory.is_dir() or not trace.exists():
            continue
        if not (directory / "run_summary.json").exists():
            print(f"skipping unfinished run: {directory.name}")
            continue
        summary = summarize_trial(directory)
        residency = analyze_residency(trace)
        events = [json.loads(line) for line in trace.open(encoding="utf-8")]
        groups.setdefault(group_of(directory.name), []).append(
            {"run_id": directory.name, "summary": summary, "residency": residency}
            | mechanism_counts(events)
        )
    return groups


def mechanism_counts(events: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Whether the system's mechanisms actually fired, not just whether they were on."""
    fused_executions = 0
    scale_out = drain = 0
    infeasible = 0
    live: dict[str, int] = {}
    peak_replicas = 0
    for event in events:
        kind = event["event_type"]
        payload = event.get("payload") or {}
        if kind == "task_execution_finished" and payload.get("stages"):
            fused_executions += 1
        elif kind == "scheduler_decision":
            if payload.get("reason") == "scale_out":
                scale_out += 1
            elif payload.get("action_type") == "drain_replica":
                drain += 1
        elif kind == "request_infeasible":
            infeasible += 1
        elif kind == "model_load_finished":
            key = event["model_key"]
            live[key] = live.get(key, 0) + 1
            peak_replicas = max(peak_replicas, live[key])
        elif kind == "model_evicted":
            key = event["model_key"]
            live[key] = live.get(key, 0) - 1
    return {
        "fused_executions": fused_executions,
        "scale_out": scale_out,
        "drain": drain,
        "infeasible": infeasible,
        "peak_replicas_per_model": peak_replicas,
    }


def pct(values: Sequence[float], digits: int = 1) -> str:
    return spread([value * 100 for value in values], digits) + "%"


def spread(values: Sequence[float], digits: int = 0) -> str:
    mean = statistics.fmean(values)
    if len(values) == 1:
        return f"{mean:.{digits}f}"
    return f"{mean:.{digits}f} [{min(values):.{digits}f}-{max(values):.{digits}f}]"


def pick(runs: Sequence[dict[str, Any]], *path: str) -> list[float]:
    values = []
    for run in runs:
        node: Any = run
        for key in path:
            node = node[key]
        values.append(float(node))
    return values


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="eval_report")
    parser.add_argument(
        "--root", type=Path, default=REPO_ROOT / "output" / "eval_20260727"
    )
    args = parser.parse_args(argv)
    groups = collect(args.root)
    if not groups:
        raise SystemExit(f"no completed run under {args.root}")
    ordered = [name for name in GROUP_ORDER if name in groups]
    ordered += [name for name in sorted(groups) if name not in GROUP_ORDER]

    print("\n== validity ==")
    header = f"{'config':16s}{'runs':>6s}{'completed/failed':>20s}{'infeasible':>12s}"
    print(header)
    print("-" * len(header))
    for name in ordered:
        runs = groups[name]
        completed = {int(r["summary"]["completed_session_count"]) for r in runs}
        failed = {int(r["summary"].get("failed_session_count", 0)) for r in runs}
        infeasible = sum(r["infeasible"] for r in runs)
        status = f"{sorted(completed)}/{sorted(failed)}"
        print(f"{name:16s}{len(runs):>6d}{status:>20s}{infeasible:>12d}")

    print("\n== comparison 1: scheduling method (makespan and model residency) ==")
    header = (
        f"{'config':16s}{'makespan':>18s}{'latency p50':>16s}{'latency p95':>16s}"
        f"{'resident GPUs':>18s}{'idle resident':>16s}{'loads':>14s}{'load GPUs':>18s}"
    )
    print(header)
    print("-" * len(header))
    for name in ordered:
        runs = groups[name]
        idle_share = [
            r["summary"]["idle_resident_gpu_seconds"]
            / max(r["summary"]["resident_gpu_seconds"], 1e-9)
            * 100
            for r in runs
        ]
        print(
            f"{name:16s}"
            f"{spread(pick(runs, 'summary', 'makespan_sec')):>18s}"
            f"{spread(pick(runs, 'summary', 'session_latency_sec', 'p50')):>16s}"
            f"{spread(pick(runs, 'summary', 'session_latency_sec', 'p95')):>16s}"
            f"{spread(pick(runs, 'summary', 'resident_gpu_seconds')):>18s}"
            f"{spread(idle_share, 1) + '%':>16s}"
            f"{spread(pick(runs, 'residency', 'loading', 'model_load_count')):>14s}"
            f"{spread(pick(runs, 'residency', 'loading', 'loading_gpu_seconds')):>18s}"
        )

    print("\n== comparison 2: load cost and where request waiting goes ==")
    header = (
        f"{'config':16s}{'load/generate':>16s}{'wait: own load':>18s}"
        f"{'wait: other load':>20s}{'distinct models':>18s}"
    )
    print(header)
    print("-" * len(header))
    for name in ordered:
        runs = groups[name]
        wait = ("residency", "wait_breakdown")
        ratio = pick(runs, "residency", "loading", "loading_over_generation")
        print(
            f"{name:16s}"
            f"{spread(ratio, 2):>16s}"
            f"{pct(pick(runs, *wait, 'own_model_load_fraction')):>18s}"
            f"{pct(pick(runs, *wait, 'head_of_line_other_key_fraction')):>20s}"
            f"{spread(pick(runs, 'residency', 'demand', 'distinct_model_keys')):>18s}"
        )

    print("\n== did each mechanism actually fire ==")
    header = (
        f"{'config':16s}{'fused execs':>14s}{'scale-out':>12s}"
        f"{'drain':>8s}{'peak replicas':>16s}"
    )
    print(header)
    print("-" * len(header))
    for name in ordered:
        runs = groups[name]
        print(
            f"{name:16s}"
            f"{spread([float(r['fused_executions']) for r in runs]):>14s}"
            f"{spread([float(r['scale_out']) for r in runs]):>12s}"
            f"{spread([float(r['drain']) for r in runs]):>8s}"
            f"{spread([float(r['peak_replicas_per_model']) for r in runs]):>16s}"
        )
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
