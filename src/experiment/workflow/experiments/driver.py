"""Shared submit/wait helpers for master experiments over a running WorkflowFleet."""

from __future__ import annotations

import time
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from pydantic import JsonValue

from workflow.fleet import WorkflowFleet
from workflow.types import SessionState
from workflow.worker import message_text


@dataclass(frozen=True, slots=True)
class SessionInvocation:
    workflow_name: str
    session_id: str
    inputs: dict[str, JsonValue]
    arrival_offset_sec: float = 0.0


def submit_sessions(
    fleet: WorkflowFleet, sessions: Sequence[SessionInvocation]
) -> None:
    if not sessions:
        return
    started = time.monotonic()

    def submit(session: SessionInvocation) -> None:
        remaining = started + session.arrival_offset_sec - time.monotonic()
        if remaining > 0:
            time.sleep(remaining)
        fleet.submit(session.workflow_name, session.session_id, session.inputs)

    with ThreadPoolExecutor(max_workers=len(sessions)) as executor:
        futures = [executor.submit(submit, session) for session in sessions]
        for future in futures:
            future.result()


def wait_terminal(
    fleet: WorkflowFleet,
    sessions: Sequence[SessionInvocation],
    timeout_sec: float,
    *,
    poll_interval_sec: float = 0.05,
) -> None:
    pending = {(session.workflow_name, session.session_id) for session in sessions}
    deadline = time.monotonic() + timeout_sec
    while pending:
        if time.monotonic() >= deadline:
            raise TimeoutError("workflow sessions timed out")
        completed: list[tuple[str, str]] = []
        for workflow_name, session_id in pending:
            state = fleet.get_session_state(workflow_name, session_id)
            if state == SessionState.FAILED:
                raise RuntimeError(
                    f"workflow session failed: {workflow_name}/{session_id}"
                )
            if state == SessionState.COMPLETED:
                completed.append((workflow_name, session_id))
        pending.difference_update(completed)
        if pending:
            time.sleep(poll_interval_sec)


def collect_outputs(
    fleet: WorkflowFleet,
    sessions: Sequence[SessionInvocation],
) -> dict[tuple[str, str], str]:
    return {
        (session.workflow_name, session.session_id): message_text(
            fleet.get_result(session.workflow_name, session.session_id)
        )
        for session in sessions
    }
