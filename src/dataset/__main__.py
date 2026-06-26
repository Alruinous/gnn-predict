from __future__ import annotations

import argparse
import json
from pathlib import Path

from dataset.gsm8k import load_gsm8k_split
from dataset.mbpp import load_mbpp_samples
from dataset.schema import TaskSample


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Normalize GSM8K and MBPP task samples into JSONL files.",
    )
    parser.add_argument(
        "--gsm8k_dir",
        default="dataset/gsm8k/data",
        help="Directory containing GSM8K train.jsonl and test.jsonl.",
    )
    parser.add_argument(
        "--mbpp_path",
        default="dataset/mbpp/sanitized-mbpp.json",
        help="Path to sanitized-mbpp.json.",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Directory for normalized task JSONL files.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    gsm8k_dir = Path(args.gsm8k_dir)
    write_jsonl(
        output_dir / "gsm8k_train.jsonl",
        load_gsm8k_split(gsm8k_dir / "train.jsonl", split="train"),
    )
    write_jsonl(
        output_dir / "gsm8k_test.jsonl",
        load_gsm8k_split(gsm8k_dir / "test.jsonl", split="test"),
    )
    write_jsonl(
        output_dir / "mbpp_sanitized.jsonl",
        load_mbpp_samples(Path(args.mbpp_path)),
    )
    return 0


def write_jsonl(path: Path, samples: list[TaskSample]) -> None:
    lines = [
        json.dumps(sample.model_dump(), ensure_ascii=False, sort_keys=True)
        for sample in samples
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
