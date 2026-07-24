from __future__ import annotations

import argparse
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import cast

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from experiment.workflow.cache_replay import (  # noqa: E402
    run_cache_replay_experiment,
    write_cache_replay_outputs,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the trace-calibrated Cache counterfactual replay."
    )
    parser.add_argument(
        "--calibration-root",
        type=Path,
        default=ROOT / "output/serve_0723",
    )
    parser.add_argument(
        "--evaluation-root",
        type=Path,
        default=ROOT / "output/serve",
    )
    parser.add_argument(
        "--prediction-cache",
        type=Path,
        default=ROOT / "cache/profile/predictions.yaml",
    )
    parser.add_argument(
        "--workflow-root",
        type=Path,
        default=ROOT / "config/workflow/serve",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "output/trace_replay/cache_optimization_20260724",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = run_cache_replay_experiment(
        calibration_root=args.calibration_root.resolve(),
        evaluation_root=args.evaluation_root.resolve(),
        prediction_cache_path=args.prediction_cache.resolve(),
        workflow_root=args.workflow_root.resolve(),
    )
    output_dir = args.output_dir.resolve()
    write_cache_replay_outputs(output_dir, result)
    summary = cast(Mapping[str, object], result["summary"])
    acceptance = cast(Mapping[str, object], summary["acceptance"])
    print(f"wrote trace replay results to {output_dir}")
    print(f"acceptance passed: {acceptance['passed']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
