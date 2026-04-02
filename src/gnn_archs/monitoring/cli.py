from __future__ import annotations

import argparse
from pathlib import Path

from .config import load_monitor_settings
from .service import run_monitoring


def parse_models_arg(raw_models: str) -> tuple[str, ...]:
    normalized_models: list[str] = []
    seen_models: set[str] = set()

    for raw_model in raw_models.split(","):
        model_name = raw_model.strip()
        if not model_name or model_name in seen_models:
            continue
        seen_models.add(model_name)
        normalized_models.append(model_name)

    if not normalized_models:
        raise argparse.ArgumentTypeError(
            "models must contain at least one non-empty target name"
        )
    return tuple(normalized_models)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect fail-fast Prometheus metrics for result documents.",
    )
    parser.add_argument(
        "--config",
        default="config/monitor/monitor.yaml",
        help="Monitor YAML config file to process.",
    )
    parser.add_argument(
        "--models",
        type=parse_models_arg,
        help=(
            "Comma-separated target names from monitor.yaml, for example "
            '"densenet121,bert-large-cased".'
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    settings = load_monitor_settings(Path(args.config), target_names=args.models)
    written_paths = run_monitoring(settings)
    for path in written_paths:
        print(path)
    return 0
