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
    from gnn_archs.result import VariantFailure, VariantResult
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
        "--device_backend",
        choices=("auto", "cpu", "nvidia", "corex"),
        default="auto",
        help="Accelerator backend; auto identifies NVIDIA or CoreX from cuda:0.",
    )
    parser.add_argument(
        "--continue_on_variant_error",
        action="store_true",
        help=(
            "Record a failed variant and continue with the next one. "
            "The result JSON is checkpointed after each variant."
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    from gnn_archs.device_runtime import resolve_runtime
    from gnn_archs.result import ResultDocument, TimeWindow, write_result_document
    from gnn_archs.util.variant_expander import expand_arch_config
    from gnn_archs.variant_runner import (
        RunContext,
        build_time_window,
        format_timestamp,
        prepare_output_layout,
        run_variants,
        summarize_variant_results,
    )

    args = parse_args(argv)
    output_root = Path(args.output_dir).resolve()
    run_timestamp = time.time()
    run_timestamp_text = format_timestamp(run_timestamp)
    runtime = resolve_runtime(args.device_backend)
    device = runtime.device
    any_failed_variants = False

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
        logger.info(
            "gpu_node=%s device=%s backend=%s device_name=%s",
            args.gpu_node,
            device,
            runtime.backend,
            runtime.device_name,
        )
        logger.info("output_root=%s", output_root)
        logger.info("dataset_artifacts_dir=%s", output_layout.root)
        logger.info("processing %s with %s variants", config_path, len(variants))

        context = RunContext(
            config_path=config_path,
            output_layout=output_layout,
            device=device,
            gpu_node=args.gpu_node,
            logger=logger,
            device_backend=runtime.backend,
            device_name=runtime.device_name,
        )
        result_path = output_layout.results_dir / (
            f"{config_path.stem}_{args.gpu_node}_{int(config_started_at)}_results.json"
        )
        failed_variants: list[VariantFailure] = []

        def checkpoint(
            variant_results: list[VariantResult],
            failures: list[VariantFailure],
        ) -> None:
            nonlocal failed_variants
            failed_variants = failures
            document = ResultDocument(
                config_path=str(config_path),
                gpu_node=args.gpu_node,
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
                failures=failures,
                summary={
                    **summarize_variant_results(variant_results),
                    "planned_variant_count": len(variants),
                    "failed_variant_count": len(failures),
                },
            )
            write_result_document(result_path, document)

        variant_results = run_variants(
            variants,
            context,
            continue_on_error=args.continue_on_variant_error,
            on_progress=checkpoint,
        )
        checkpoint(variant_results, failed_variants)
        logger.info("wrote result document to %s", result_path)
        if failed_variants:
            any_failed_variants = True
            logger.warning(
                "completed %s/%s variants; %s failed (see result JSON and log)",
                len(variant_results),
                len(variants),
                len(failed_variants),
            )

    return 1 if any_failed_variants else 0


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


def resolve_device() -> torch.device:
    from gnn_archs.device_runtime import resolve_runtime

    return resolve_runtime().device


if __name__ == "__main__":
    raise SystemExit(main())
