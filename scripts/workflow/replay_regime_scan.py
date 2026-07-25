"""Scan arrival intensity and GPU budget for regimes where lifecycle policy matters.

Measured per-task durations are held fixed (they come from the traces); only the
arrival compression and the GPU inventory vary. This locates the operating points
where the scheduler actually faces competing model-load decisions, which is a
precondition for any residency policy to separate from LRU.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from experiment.workflow.cache_replay import (  # noqa: E402
    CohortDefinition,
    PolicyName,
    ReplaySimulator,
    ReplayWorkload,
    build_replay_workload,
    load_workflows_for_runs,
    read_trace_run,
)
from workflow.artifacts import load_resource_contract_cache  # noqa: E402

SWEEP_POLICIES: tuple[PolicyName, ...] = ("fifo", "kairos", "history", "profile_cache")


def compress(workload: ReplayWorkload, scale: float, slots: dict[str, int]):
    return replace(
        workload,
        session_arrivals={
            session_id: arrival * scale
            for session_id, arrival in workload.session_arrivals.items()
        },
        gpu_slots=slots,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cohort-dir", type=Path, default=ROOT / "output/serve_0724/6w_poisson"
    )
    parser.add_argument("--run-prefix", default="hetero_r100_")
    parser.add_argument(
        "--prediction-cache",
        type=Path,
        default=ROOT / "cache/profile_v2/predictions.yaml",
    )
    parser.add_argument(
        "--workflow-root", type=Path, default=ROOT / "config/workflow/serve"
    )
    parser.add_argument(
        "--arrival-scales",
        default="1.0,0.5,0.25,0.125",
        help="multipliers on arrival timestamps; smaller = more concurrency",
    )
    parser.add_argument(
        "--pools",
        default="a100=1,v100=2;a100=1,v100=1;v100=2;a100=2,v100=2",
        help="semicolon-separated GPU inventories",
    )
    parser.add_argument("--json-out", type=Path, default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    traces = tuple(
        sorted(
            path
            for path in args.cohort_dir.glob("*/workflow_trace.jsonl")
            if path.parent.name.startswith(args.run_prefix)
        )
    )
    if not traces:
        raise SystemExit(f"no traces matching {args.run_prefix}")
    runs = tuple(read_trace_run(path, expected_sessions=120) for path in traces)
    workflows = load_workflows_for_runs(runs, args.workflow_root.resolve())
    base = build_replay_workload(
        CohortDefinition(
            name="scan", trace_paths=traces, gpu_slots={"a100": 1, "v100": 2}
        ),
        runs,
        workflows,
    )
    cache = load_resource_contract_cache(args.prediction_cache.resolve())
    pools = []
    for spec in args.pools.split(";"):
        entry = {}
        for part in spec.split(","):
            kind, _, count = part.partition("=")
            entry[kind] = int(count)
        pools.append(entry)
    scales = [float(value) for value in args.arrival_scales.split(",")]

    rows = []
    print(
        f"{'pool':16s} {'scale':>6s} {'policy':14s} {'span_s':>8s} {'meanE2E':>8s} "
        f"{'loads':>6s} {'load_gpu_s':>10s} {'idle_res_s':>10s} {'bubble':>7s} "
        f"{'pf':>3s} {'vs_fifo':>8s}"
    )
    for slots in pools:
        label = ",".join(f"{k}{v}" for k, v in sorted(slots.items()))
        for scale in scales:
            workload = compress(base, scale, slots)
            fifo_mean = None
            for policy in SWEEP_POLICIES:
                try:
                    metrics = ReplaySimulator(
                        workload,
                        policy=policy,
                        profile_cache=cache,
                        calibrated_cache=cache,
                    ).run()
                except (RuntimeError, ValueError) as error:
                    print(f"{label:16s} {scale:6.3f} {policy:14s} FAILED: {error}")
                    continue
                if policy == "fifo":
                    fifo_mean = metrics.mean_session_latency_sec
                delta = (
                    100.0 * (metrics.mean_session_latency_sec / fifo_mean - 1.0)
                    if fifo_mean
                    else 0.0
                )
                rows.append(
                    {
                        "pool": label,
                        "arrival_scale": scale,
                        "policy": policy,
                        "span_sec": metrics.replay_completion_span_sec,
                        "mean_e2e_sec": metrics.mean_session_latency_sec,
                        "loads": metrics.model_load_count,
                        "loading_gpu_sec": metrics.loading_gpu_seconds,
                        "idle_resident_gpu_sec": metrics.idle_resident_gpu_seconds,
                        "bubble": metrics.pipeline_bubble_ratio,
                        "prefetch": metrics.prefetch_count,
                        "pct_vs_fifo": delta,
                    }
                )
                print(
                    f"{label:16s} {scale:6.3f} {policy:14s} "
                    f"{metrics.replay_completion_span_sec:8.1f} "
                    f"{metrics.mean_session_latency_sec:8.1f} "
                    f"{metrics.model_load_count:6d} "
                    f"{metrics.loading_gpu_seconds:10.1f} "
                    f"{metrics.idle_resident_gpu_seconds:10.1f} "
                    f"{metrics.pipeline_bubble_ratio * 100:6.1f}% "
                    f"{metrics.prefetch_count:3d} {delta:+7.1f}%"
                )
            print()
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(rows, indent=2), encoding="utf-8")
        print(f"wrote {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
