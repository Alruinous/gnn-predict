from __future__ import annotations

from pathlib import Path

import yaml

from workflow.schema import Workflow
from workflow.validation import validate_workflow


def load_workflow(path: str | Path) -> Workflow:
    workflow_path = Path(path)
    if workflow_path.suffix != ".yaml":
        raise ValueError(f"workflow config must be a YAML .yaml file: {workflow_path}")
    payload = yaml.safe_load(workflow_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"workflow YAML must contain a mapping: {workflow_path}")
    workflow = Workflow.model_validate(payload)
    return validate_workflow(workflow)


def load_workflows(path: str | Path) -> list[Workflow]:
    workflow_path = Path(path)
    if workflow_path.is_file():
        return [load_workflow(workflow_path)]

    yaml_paths = sorted(workflow_path.glob("*.yaml"))
    if not yaml_paths:
        raise ValueError(f"workflow directory contains no YAML files: {workflow_path}")
    return [load_workflow(yaml_path) for yaml_path in yaml_paths]
