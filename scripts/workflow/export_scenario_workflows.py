"""Export the QMSum + MBPP workflow builders to committed serve YAMLs."""

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

from experiment.workflow.gsm8k import build_gsm8k_workflow  # noqa: E402
from experiment.workflow.mbpp import build_mbpp_workflow  # noqa: E402
from experiment.workflow.qmsum import build_qmsum_workflow  # noqa: E402
from workflow.schema import Workflow  # noqa: E402

DEFAULT_OUTPUT_DIR = ROOT / "config" / "workflow" / "serve"
GSM8K_VARIANT_COUNT = 4


def export_workflow(workflow: Workflow, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as stream:
        yaml.safe_dump(
            workflow.model_dump(mode="json"),
            stream,
            sort_keys=False,
            allow_unicode=True,
        )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="export_scenario_workflows")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args(argv)
    export_workflow(build_qmsum_workflow(), args.output_dir / "qmsum.yaml")
    export_workflow(build_mbpp_workflow(), args.output_dir / "mbpp.yaml")
    for index in range(1, GSM8K_VARIANT_COUNT + 1):
        export_workflow(
            build_gsm8k_workflow(workflow_name=f"gsm8k{index}"),
            args.output_dir / f"gsm8k{index}.yaml",
        )
    print(
        f"wrote qmsum.yaml, mbpp.yaml and gsm8k1..{GSM8K_VARIANT_COUNT} "
        f"to {args.output_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
