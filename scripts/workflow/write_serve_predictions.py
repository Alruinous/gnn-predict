"""Write a synthetic v100+a100 ResourceContractCache for the serve models."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from experiment.workflow.cache import write_serve_prediction_cache  # noqa: E402


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="write_serve_predictions")
    parser.add_argument(
        "output", type=Path, help="destination JSON (fails if it exists)"
    )
    parser.add_argument(
        "--gpu-kinds", default="v100,a100", help="comma-separated gpu kinds"
    )
    args = parser.parse_args(argv)
    gpu_kinds = tuple(
        kind.strip().casefold() for kind in args.gpu_kinds.split(",") if kind.strip()
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    digest = write_serve_prediction_cache(args.output, gpu_kinds)
    print(f"wrote {args.output} (gpu_kinds={list(gpu_kinds)}) sha256={digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
