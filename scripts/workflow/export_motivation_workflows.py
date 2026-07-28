"""Export the motivation workflows (one file per heterogeneity level) to serve YAMLs."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import yaml  # noqa: E402

from experiment.workflow.motivation_workflows import (  # noqa: E402
    HETEROGENEITY_LEVELS,
    build_all,
)

DEFAULT_OUTPUT_DIR = ROOT / "config" / "workflow" / "serve" / "motivation_20260727"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="export_motivation_workflows")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    written: list[str] = []
    for heterogeneity in HETEROGENEITY_LEVELS:
        for workflow in build_all(heterogeneity):
            name = f"h{heterogeneity}_{workflow.workflow_name}.yaml"
            with (args.output_dir / name).open("w") as stream:
                yaml.safe_dump(
                    workflow.model_dump(mode="json"),
                    stream,
                    sort_keys=False,
                    allow_unicode=True,
                )
            written.append(name)
    print(f"wrote {len(written)} workflows to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
