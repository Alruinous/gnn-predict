from __future__ import annotations

from pathlib import Path

import yaml

from workflow.baseline.types import BaselineWorkflow


def load_baseline_workflow(path: str | Path) -> BaselineWorkflow:
    with open(path) as f:
        raw = yaml.safe_load(f)
    return BaselineWorkflow.model_validate(raw)
