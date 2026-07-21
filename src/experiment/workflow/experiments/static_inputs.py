"""Generic experiment: submit an explicit list of sessions from config."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from experiment.workflow.experiments.driver import (
    SessionInvocation,
    submit_sessions,
    wait_terminal,
)
from workflow.fleet import WorkflowFleet


def run(
    fleet: WorkflowFleet,
    *,
    output_dir: Path,
    run_id: str,
    config: Mapping[str, Any],
) -> None:
    sessions = tuple(
        SessionInvocation(
            workflow_name=str(entry["workflow"]),
            session_id=str(entry["session_id"]),
            inputs=dict(entry["inputs"]),
            arrival_offset_sec=float(entry.get("arrival_offset_sec", 0.0)),
        )
        for entry in config["sessions"]
    )
    submit_sessions(fleet, sessions)
    wait_terminal(fleet, sessions, float(config.get("timeout_sec", 300.0)))
