"""Replay one serve cohort through every scheduler policy on identical measured work.

The serve sweep runs each policy against its own live GPU session, so arrival jitter
and generation-length noise are confounded with the policy. This driver instead takes
the per-task durations already measured in the traces, medians them across the cohort,
and replays that single fixed workload through fifo / kairos / history / cache. Any
difference in the output is therefore attributable to scheduling alone.

Pass --workflow-root and --prediction-cache to compare a candidate config or cache
against the committed one on the same workload.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from experiment.workflow.cache_replay import (  # noqa: E402
    CohortDefinition,
    PolicyName,
    ReplaySimulator,
    build_replay_workload,
    load_workflows_for_runs,
    read_trace_run,
)
from workflow.artifacts import load_resource_contract_cache  # noqa: E402
from workflow.replica import ModelDeploymentConfig  # noqa: E402
from workflow.schema import AgentNodeConfig  # noqa: E402

SWEEP_POLICIES: tuple[PolicyName, ...] = ("fifo", "kairos", "history", "profile_cache")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cohort-dir",
        type=Path,
        default=ROOT / "output/serve/6w_poisson",
        help="directory holding one run subdirectory per policy",
    )
    parser.add_argument(
        "--run-prefix",
        default="hetero_r100_",
        help="only replay runs whose directory name starts with this",
    )
    parser.add_argument(
        "--prediction-cache",
        type=Path,
        default=ROOT / "cache/profile_v2/predictions.yaml",
    )
    parser.add_argument(
        "--workflow-root", type=Path, default=ROOT / "config/workflow/serve"
    )
    parser.add_argument("--expected-sessions", type=int, default=120)
    parser.add_argument("--label", default="candidate")
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
        raise SystemExit(
            f"no traces under {args.cohort_dir} matching {args.run_prefix}"
        )
    runs = tuple(
        read_trace_run(path, expected_sessions=args.expected_sessions)
        for path in traces
    )
    workflows = load_workflows_for_runs(runs, args.workflow_root.resolve())
    definition = CohortDefinition(
        name=f"{args.cohort_dir.name}:{args.run_prefix}",
        trace_paths=traces,
        gpu_slots={"a100": 1, "v100": 2},
    )
    workload = build_replay_workload(definition, runs, workflows)
    cache = load_resource_contract_cache(args.prediction_cache.resolve())

    print(f"# {args.label}  cohort={definition.name}  traces={len(traces)}")
    print(f"#   workflows={args.workflow_root}  cache={args.prediction_cache}")
    keys = {
        ModelDeploymentConfig.from_node(node).model_key
        for workflow in workflows.values()
        for node in workflow.node_map().values()
        if isinstance(node, AgentNodeConfig)
    }
    print(f"#   distinct model_key = {len(keys)}")
    header = (
        f"{'policy':16s} {'span_s':>9s} {'meanE2E_s':>10s} {'loads':>6s} "
        f"{'reuse':>6s} {'load_gpu_s':>11s} {'idle_res_s':>11s} {'bubble':>8s} "
        f"{'prefetch':>9s} {'pf_intime':>10s}"
    )
    print(header)
    rows: dict[str, dict[str, object]] = {}
    for policy in SWEEP_POLICIES:
        metrics = ReplaySimulator(
            workload,
            policy=policy,
            profile_cache=cache,
            calibrated_cache=cache,
        ).run()
        rows[policy] = asdict(metrics)
        print(
            f"{policy:16s} {metrics.replay_completion_span_sec:9.1f} "
            f"{metrics.mean_session_latency_sec:10.1f} {metrics.model_load_count:6d} "
            f"{metrics.model_reuse_count:6d} {metrics.loading_gpu_seconds:11.1f} "
            f"{metrics.idle_resident_gpu_seconds:11.1f} "
            f"{metrics.pipeline_bubble_ratio * 100:7.2f}% "
            f"{metrics.prefetch_count:9d} "
            f"{metrics.prefetch_ready_before_acquire_count:10d}"
        )
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(
            json.dumps({"label": args.label, "policies": rows}, indent=2),
            encoding="utf-8",
        )
        print(f"wrote {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
