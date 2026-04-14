from __future__ import annotations

import argparse
from pathlib import Path

from gnn_model.runner import run_experiment


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the migrated gnn_model fake-data training workflow.",
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to the gnn_model YAML config.",
    )
    parser.add_argument(
        "--output_dir",
        default="output",
        help="Root directory for logs, checkpoints, and result JSON.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Optional torch device override such as cpu or cuda:0.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    run_experiment(
        config_path=Path(args.config),
        output_dir=Path(args.output_dir),
        requested_device=args.device,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
