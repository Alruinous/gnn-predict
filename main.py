from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING

import torch
import yaml

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

if TYPE_CHECKING:
    from gnn_archs.config import ArchConfig
    from gnn_archs.variant_runner import OutputLayout


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run migrated gnn_archs variant generation workflows.",
    )
    parser.add_argument(
        "--config",
        nargs="+",
        required=True,
        help="One or more YAML config files to process.",
    )
    parser.add_argument(
        "--output_dir",
        default="output",
        help=(
            "Parent directory that will contain one artifact directory per config "
            "dataset."
        ),
    )
    parser.add_argument(
        "--gpu_node",
        required=True,
        help="GPU node label for the current run.",
    )
    parser.add_argument(
        "--gpu_ids",
        required=True,
        help='Comma-separated GPU ids, for example "0" or "0,1".',
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    from gnn_archs.result import ResultDocument, TimeWindow, write_result_document
    from gnn_archs.util.variant_expander import expand_arch_config
    from gnn_archs.variant_runner import (
        RunContext,
        build_time_window,
        format_timestamp,
        prepare_output_layout,
        run_variant,
        summarize_variant_results,
    )

    args = parse_args(argv)
    gpu_ids = parse_gpu_ids(args.gpu_ids)
    output_root = Path(args.output_dir).resolve()
    run_timestamp = time.time()
    run_timestamp_text = format_timestamp(run_timestamp)
    device = resolve_device(gpu_ids)

    for config_path_str in args.config:
        config_path = Path(config_path_str).resolve()
        config_started_at = time.time()
        output_layout = prepare_output_layout(output_root, config_path)
        logger = configure_logging(
            output_layout,
            args.gpu_node,
            run_timestamp,
            logger_name=f"gnn_predict.{output_layout.root.name}",
        )
        config = load_arch_config(config_path)
        variants = expand_arch_config(config)
        logger.info("starting migrated gnn_archs run")
        logger.info("config files: %s", ", ".join(args.config))
        logger.info("gpu_node=%s gpu_ids=%s device=%s", args.gpu_node, gpu_ids, device)
        logger.info("output_root=%s", output_root)
        logger.info("dataset_artifacts_dir=%s", output_layout.root)
        logger.info("processing %s with %s variants", config_path, len(variants))

        context = RunContext(
            config_path=config_path,
            output_layout=output_layout,
            device=device,
            gpu_node=args.gpu_node,
            gpu_ids=gpu_ids,
            logger=logger,
        )
        variant_results = [run_variant(variant, context) for variant in variants]
        document = ResultDocument(
            config_path=str(config_path),
            gpu_node=args.gpu_node,
            gpu_ids=gpu_ids,
            timings={
                "full": build_time_window(config_started_at, time.time()),
                "run_started": TimeWindow(
                    started_at_ts=run_timestamp,
                    ended_at_ts=run_timestamp,
                    started_at_text=run_timestamp_text,
                    ended_at_text=run_timestamp_text,
                ),
            },
            variants=variant_results,
            summary=summarize_variant_results(variant_results),
        )
        result_path = (
            output_layout.results_dir
            / (
                f"{config_path.stem}_{args.gpu_node}_"
                f"{int(config_started_at)}_results.json"
            )
        )
        write_result_document(result_path, document)
        logger.info("wrote result document to %s", result_path)

    return 0


def configure_logging(
    output_layout: OutputLayout,
    gpu_node: str,
    run_timestamp: float,
    *,
    logger_name: str,
) -> logging.Logger:
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()
    logger.propagate = False

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    log_path = output_layout.logs_dir / f"run_{gpu_node}_{int(run_timestamp)}.log"

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


def load_arch_config(config_path: Path) -> ArchConfig:
    from gnn_archs.config import ArchConfig

    with config_path.open(encoding="utf-8") as file:
        raw_config = yaml.safe_load(file)
    return ArchConfig.model_validate(raw_config)


def parse_gpu_ids(raw_gpu_ids: str) -> list[int]:
    gpu_ids = [item.strip() for item in raw_gpu_ids.split(",") if item.strip()]
    if not gpu_ids:
        raise ValueError("gpu_ids must contain at least one GPU id")
    return [int(item) for item in gpu_ids]


def resolve_device(gpu_ids: list[int]) -> torch.device:
    if torch.cuda.is_available():
        return torch.device(f"cuda:{gpu_ids[0]}")
    return torch.device("cpu")


if __name__ == "__main__":
    raise SystemExit(main())
