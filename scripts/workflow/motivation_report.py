"""Report for the model-residency motivation experiment.

Reads every run directory under a root, runs the residency analysis on each trace, and
writes the main table plus one CSV per panel. Panels map to the experiment's four
checkpoints: demand concurrency, optimal-replacement lower bound, loading cost and
wait breakdown, and preloading opportunity.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from experiment.workflow.analysis import read_jsonl  # noqa: E402
from experiment.workflow.motivation_residency import analyze_residency  # noqa: E402

POOL_SIZES = (1, 2, 3, 4, 5, 6)


def arrival_digest(trace_path: Path) -> str:
    """Digest of the submitted-session order, so arms can be shown to be paired."""
    submissions = [
        str(event.get("session_id"))
        for event in read_jsonl(trace_path)
        if event.get("event_type") == "session_submitted"
    ]
    return hashlib.md5("\n".join(submissions).encode()).hexdigest()[:10]


def session_counts(trace_path: Path) -> dict[str, int]:
    counts = {"completed": 0, "failed": 0}
    for event in read_jsonl(trace_path):
        if event.get("event_type") == "session_completed":
            counts["completed"] += 1
        elif event.get("event_type") == "session_failed":
            counts["failed"] += 1
    return counts


def collect(root: Path) -> dict[str, dict[str, Any]]:
    runs: dict[str, dict[str, Any]] = {}
    for trace_path in sorted(root.glob("*/workflow_trace.jsonl")):
        run_id = trace_path.parent.name
        runs[run_id] = {
            "analysis": analyze_residency(trace_path, pool_sizes=POOL_SIZES),
            "sessions": session_counts(trace_path),
            "arrival_digest": arrival_digest(trace_path),
        }
    if not runs:
        raise FileNotFoundError(f"no */workflow_trace.jsonl under {root}")
    return runs


def write_csv(path: Path, header: Sequence[str], rows: Sequence[Sequence[Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(header)
        writer.writerows(rows)


def main_table(runs: dict[str, dict[str, Any]]) -> str:
    header = (
        "| run | sessions | arrival | keys | min_static_cards | loads | "
        "load_gpu_s | load/gen | wait_own | wait_hol | lead>=d | lead&cap |"
    )
    lines = [header, "|" + "---|" * 11]
    for run_id, data in runs.items():
        analysis = data["analysis"]
        demand = analysis["demand"]
        loading = analysis["loading"]
        wait = analysis["wait_breakdown"]
        preload = analysis["preload"]
        sessions = data["sessions"]
        lines.append(
            f"| {run_id} "
            f"| {sessions['completed']}c/{sessions['failed']}f "
            f"| {data['arrival_digest']} "
            f"| {demand['distinct_model_keys']} "
            f"| {demand['min_static_cards']} "
            f"| {loading['model_load_count']} "
            f"| {loading['loading_gpu_seconds']:.0f} "
            f"| {loading['loading_over_generation']:.3f} "
            f"| {wait['own_model_load_fraction']:.3f} "
            f"| {wait['head_of_line_other_key_fraction']:.3f} "
            f"| {preload['lead_sufficient_fraction']:.3f} "
            f"| {preload['lead_and_capacity_fraction']:.3f} |"
        )
    return "\n".join(lines)


def write_panels(root: Path, runs: dict[str, dict[str, Any]]) -> list[Path]:
    written: list[Path] = []

    concurrency_rows = [
        (run_id, level, seconds)
        for run_id, data in runs.items()
        for level, seconds in data["analysis"]["demand"][
            "seconds_at_concurrency"
        ].items()
    ]
    path = root / "panel1_demand_concurrency.csv"
    write_csv(path, ("run_id", "concurrent_models", "seconds"), concurrency_rows)
    written.append(path)

    belady_rows = [
        (run_id, size, loads)
        for run_id, data in runs.items()
        for size, loads in data["analysis"]["belady"]["loads_by_pool_size"].items()
    ]
    path = root / "panel1b_belady_lower_bound.csv"
    write_csv(path, ("run_id", "pool_size", "optimal_loads"), belady_rows)
    written.append(path)

    cost_rows = [
        (
            run_id,
            data["analysis"]["loading"]["model_load_count"],
            data["analysis"]["loading"]["loading_gpu_seconds"],
            data["analysis"]["loading"]["generation_gpu_seconds"],
            data["analysis"]["loading"]["loading_over_generation"],
            data["analysis"]["wait_breakdown"]["total_sec"],
            data["analysis"]["wait_breakdown"]["own_model_load_sec"],
            data["analysis"]["wait_breakdown"]["head_of_line_other_key_sec"],
            data["analysis"]["wait_breakdown"]["other_sec"],
        )
        for run_id, data in runs.items()
    ]
    path = root / "panel2_loading_cost.csv"
    write_csv(
        path,
        (
            "run_id",
            "model_load_count",
            "loading_gpu_seconds",
            "generation_gpu_seconds",
            "loading_over_generation",
            "wait_total_sec",
            "wait_own_model_load_sec",
            "wait_head_of_line_sec",
            "wait_other_sec",
        ),
        cost_rows,
    )
    written.append(path)

    preload_rows = [
        (
            run_id,
            record["model_key"][:8],
            record["workflow_name"],
            record["node_id"],
            record["load_duration_sec"],
            record["information_lead_sec"],
            int(record["lead_sufficient"]),
            int(record["capacity_sufficient"]),
            record["hideable_sec"],
        )
        for run_id, data in runs.items()
        for record in data["analysis"]["preload"]["records"]
    ]
    path = root / "panel4_preload_opportunity.csv"
    write_csv(
        path,
        (
            "run_id",
            "model_key",
            "workflow_name",
            "node_id",
            "load_duration_sec",
            "information_lead_sec",
            "lead_sufficient",
            "capacity_sufficient",
            "hideable_sec",
        ),
        preload_rows,
    )
    written.append(path)
    return written


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="motivation_report")
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args(argv)

    runs = collect(args.root)
    table = main_table(runs)

    (args.root / "motivation_summary.md").write_text(table + "\n", encoding="utf-8")
    (args.root / "motivation_summary.json").write_text(
        json.dumps(
            {
                run_id: {
                    key: value
                    for key, value in data["analysis"].items()
                    if key != "preload"
                }
                | {
                    "preload": {
                        key: value
                        for key, value in data["analysis"]["preload"].items()
                        if key != "records"
                    },
                    "sessions": data["sessions"],
                    "arrival_digest": data["arrival_digest"],
                }
                for run_id, data in runs.items()
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    panels = write_panels(args.root, runs)

    print(table)
    digests = {data["arrival_digest"] for data in runs.values()}
    print(
        f"\narrival digests: {sorted(digests)} "
        f"({'paired' if len(digests) == 1 else 'NOT PAIRED'})"
    )
    print(f"wrote {len(panels) + 2} files to {args.root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
