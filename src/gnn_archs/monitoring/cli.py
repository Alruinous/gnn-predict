from __future__ import annotations

import argparse
import logging
from pathlib import Path

from .config import load_monitor_settings
from .service import run_monitoring

logger = logging.getLogger(__name__)

def parse_models_arg(raw_models: str) -> tuple[str, ...]:
    normalized_models: list[str] = []
    seen_models: set[str] = set()

    for raw_model in raw_models.split(","):
        model_name = raw_model.strip()
        if not model_name or model_name in seen_models:
            continue
        seen_models.add(model_name)
        normalized_models.append(model_name)

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
            '"densenet121,bert-large-cased". Leave empty to use all enabled '
            "targets from the config file."
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO)
    args = parse_args(argv)
    target_names = args.models
    if target_names == ():
        logger.info(
            "Empty --models value provided; reading all enabled targets from %s.",
            args.config,
        )
        target_names = None

    settings = load_monitor_settings(Path(args.config), target_names=target_names)
    written_paths = run_monitoring(settings, logger=logger)
    for path in written_paths:
        print(path)
    return 0
