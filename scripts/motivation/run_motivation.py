from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from dataset.mbpp import load_mbpp_samples
from dataset.summarization import load_qmsum_split
from scripts.motivation.analysis import (
    summarize_mbpp,
    summarize_qmsum,
    write_combined_summary,
)
from scripts.motivation.common import write_jsonl
from scripts.motivation.plotting import write_plots
from scripts.motivation.report import write_manifest, write_report
from scripts.motivation.runner import run_mbpp_chain, run_qmsum_3way
from scripts.motivation.sampling import (
    mbpp_sample_rows,
    qmsum_sample_rows,
    stratified_by_input_length,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("output/motivation"))
    parser.add_argument("--image-dir", type=Path, default=Path("docs/workflow/img"))
    parser.add_argument(
        "--report-path",
        type=Path,
        default=Path("docs/workflow/motivation_20260704.md"),
    )
    parser.add_argument(
        "--qmsum-path",
        type=Path,
        default=Path("dataset/QMSum/data/ALL"),
    )
    parser.add_argument(
        "--mbpp-path",
        type=Path,
        default=Path("dataset/mbpp/sanitized-mbpp.json"),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--qmsum-count", type=int, default=60)
    parser.add_argument("--qmsum-per-stratum", type=int, default=20)
    parser.add_argument("--qmsum-shards", type=int, default=3)
    parser.add_argument("--mbpp-count", type=int, default=60)
    parser.add_argument("--judge", choices=["llm", "rouge"], default="llm")
    parser.add_argument("--qmsum-max-input-tokens", type=int, default=8192)
    parser.add_argument("--qmsum-max-new-tokens", type=int, default=512)
    parser.add_argument("--mbpp-max-input-tokens", type=int, default=4096)
    parser.add_argument("--mbpp-max-new-tokens", type=int, default=512)
    parser.add_argument("--skip-preflight", action="store_true")
    parser.add_argument("--skip-qmsum", action="store_true")
    parser.add_argument("--skip-mbpp", action="store_true")
    parser.add_argument("--make-report", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    assert args.qmsum_shards == 3, args.qmsum_shards
    assert args.qmsum_count == args.qmsum_per_stratum * 3, (
        args.qmsum_count,
        args.qmsum_per_stratum,
    )
    validate_cuda()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    qmsum_samples, mbpp_samples = load_experiment_samples(args)
    write_jsonl(
        args.output_dir / "qmsum_3way_samples.jsonl",
        qmsum_sample_rows(qmsum_samples),
    )
    write_jsonl(
        args.output_dir / "mbpp_chain_samples.jsonl",
        mbpp_sample_rows(mbpp_samples),
    )
    write_manifest(
        args.output_dir / "run_manifest.json",
        {
            "seed": args.seed,
            "qmsum_count": args.qmsum_count,
            "qmsum_per_stratum": args.qmsum_per_stratum,
            "qmsum_shards": args.qmsum_shards,
            "mbpp_count": args.mbpp_count,
            "judge": args.judge,
        },
    )
    if args.skip_qmsum:
        qmsum_samples = []
    if args.skip_mbpp:
        mbpp_samples = []
    if not args.skip_preflight:
        run_preflight(args, qmsum_samples, mbpp_samples)
    if not args.skip_qmsum:
        run_qmsum_3way(
            qmsum_samples,
            output_dir=args.output_dir,
            judge_mode=args.judge,
            max_input_tokens=args.qmsum_max_input_tokens,
            max_new_tokens=args.qmsum_max_new_tokens,
        )
    if not args.skip_mbpp:
        run_mbpp_chain(
            mbpp_samples,
            output_dir=args.output_dir,
            max_input_tokens=args.mbpp_max_input_tokens,
            max_new_tokens=args.mbpp_max_new_tokens,
        )
    qmsum_summary = safe_summarize_qmsum(args.output_dir)
    mbpp_summary = safe_summarize_mbpp(args.output_dir)
    write_combined_summary(
        args.output_dir / "summary.jsonl",
        qmsum_summary,
        mbpp_summary,
    )
    image_paths = write_plots(
        qmsum_summary=qmsum_summary,
        mbpp_summary=mbpp_summary,
        image_dir=args.image_dir,
    )
    if args.make_report:
        write_report(
            output_path=args.report_path,
            qmsum_summary=qmsum_summary,
            mbpp_summary=mbpp_summary,
            image_paths=image_paths,
            output_dir=args.output_dir,
        )


def load_experiment_samples(args: argparse.Namespace) -> tuple[list[Any], list[Any]]:
    qmsum_all = load_qmsum_split(args.qmsum_path, "test")
    mbpp_all = [
        sample for sample in load_mbpp_samples(args.mbpp_path) if sample.split == "test"
    ]
    qmsum_samples = stratified_by_input_length(
        qmsum_all,
        total_count=args.qmsum_count,
        seed=args.seed,
        strata_count=3,
    )
    mbpp_samples = stratified_by_input_length(
        mbpp_all,
        total_count=args.mbpp_count,
        seed=args.seed,
        strata_count=3,
    )
    return qmsum_samples, mbpp_samples


def safe_summarize_qmsum(output_dir: Path) -> dict[str, Any]:
    trace_path = output_dir / "qmsum_3way_trace.jsonl"
    summary_path = output_dir / "qmsum_3way_summary.json"
    if trace_path.exists():
        return summarize_qmsum(trace_path, summary_path)
    return json.loads(summary_path.read_text())


def safe_summarize_mbpp(output_dir: Path) -> dict[str, Any]:
    trace_path = output_dir / "mbpp_chain_trace.jsonl"
    summary_path = output_dir / "mbpp_chain_summary.json"
    if trace_path.exists():
        return summarize_mbpp(trace_path, summary_path)
    return json.loads(summary_path.read_text())


def run_preflight(
    args: argparse.Namespace,
    qmsum_samples: list[Any],
    mbpp_samples: list[Any],
) -> None:
    preflight_dir = args.output_dir / "preflight"
    if not args.skip_qmsum and qmsum_samples:
        qmsum_probe = max(qmsum_samples, key=lambda sample: len(sample.input_text))
        run_qmsum_3way(
            [qmsum_probe],
            output_dir=preflight_dir,
            judge_mode=args.judge,
            max_input_tokens=args.qmsum_max_input_tokens,
            max_new_tokens=args.qmsum_max_new_tokens,
        )
    if not args.skip_mbpp and mbpp_samples:
        run_mbpp_chain(
            [mbpp_samples[0]],
            output_dir=preflight_dir,
            max_input_tokens=args.mbpp_max_input_tokens,
            max_new_tokens=args.mbpp_max_new_tokens,
        )


def validate_cuda() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the motivation experiment")
    if torch.cuda.device_count() < 4:
        raise RuntimeError("QMSum 3-way experiment requires at least 4 CUDA devices")


if __name__ == "__main__":
    main()
